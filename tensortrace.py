import numpy as np
import tensorflow as tf
sess = tf.Session()
import pickle
from relaxflow.reparam import CategoricalReparam
import time
dtype = 'float64'

import tqdm
import pandas as pd

from relaxflow.relax import RELAX
from collapsedclustering import CollapsedStochasticBlock, KLcorrectedBound
import tensornets as tn

from itertools import product

from AMSGrad.optimizers import AMSGrad as amsgrad

from networkx import karate_club_graph, adjacency_matrix

karate = karate_club_graph()
X = adjacency_matrix(karate).toarray().astype('float64')
N = 9
X = X[:N,:N]

#FLAGS
name = 'shadowvsrelax' 
version = 1
Ntest = 0 #number of edges to use for testing
K = 2 #number of communities to look for
folder = name + 'V{}K{}'.format(version, K)

#factors variations to run experiments over
random_restarts = 2
nsteps = 10000

rate = 1e-1#[1e-1,1e-2,1e-3]
decay = 0.1
decay_steps = nsteps/2.
optimizer = 'ams' #options: ams
nsample = 1000
coretype = 'canon'
marginal = False
timeit = False

objectives = ['shadow', 'relax-learned']
maxranks = [4]


factor_code = ['R','S','L']
factor_names = ['rank','restarts','objective']
factors = [maxranks, range(random_restarts), objectives]
short_key = False
active_factors = [len(factor)>1 for factor in factors]
all_config = list(product(*factors))
config_count = np.prod([len(factor) for factor in factors])
config_full_name = ''.join([code + '-'.join([str(fact) for fact in factor]) for code, factor in zip(factor_code, factors)])
copy_writer = []
        
np.random.seed(1)
tf.set_random_seed(1.)
#tf.reset_default_graph()



#generate mask of observed edges
if Ntest > 0:
    mask = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
    mask = mask < np.sort(mask.ravel())[Ntest]
    mask = np.logical_not(np.logical_or(mask,mask.T))
    mask = np.triu(mask, 1)
    predictionmask = np.triu(~mask, 1)
    mask = tf.convert_to_tensor(mask.astype(dtype))
    predictionmask = tf.convert_to_tensor(predictionmask.astype(dtype))
else:
    mask = np.ones((N, N), dtype=bool)
    mask = np.triu(mask, 1)
    mask = tf.convert_to_tensor(mask.astype(dtype))

concentration = 1.
a = 1. + (concentration-1.)*np.eye(K)
b = concentration - (concentration-1.)*np.eye(K) 

all_anchors = tf.stack([np.eye(K)[list(index)] for index in np.ndindex((K,)*N)])

p = CollapsedStochasticBlock(N, K, alpha=1, a=a, b=b)
logp = lambda sample: p.batch_logp(sample, X, observed=mask)
logp_all = p.populatetensor(X, observed=mask,sess=sess).ravel()
ptensor = np.exp(logp_all - np.logaddexp.reduce(logp_all))

beta1=tf.Variable(0.9,dtype='float64')
beta2=tf.Variable(0.999,dtype='float64')
epsilon=tf.Variable(0.999,dtype='float64')

q = {}
Xt = {}
cores = {}
loss = {}
dloss = {}
trueloss = {}
step = {}
var_step = {}
cvweight = {}
qtensor = {}
update = {}
decay_stage = tf.Variable(1, name='decay_stage', trainable=False, dtype=tf.int32)
increment_decay_stage_op = tf.assign(decay_stage, decay_stage+1)

var_reset = []
coregroup = [[] for _ in range(random_restarts)]

def flattengrad(grad_and_vars):
    grads = []
    for grad, var in grad_and_vars:
        grads += [tf.reshape(grad,(-1,))]
    return tf.concat(grads, axis=0)

def flattenlist(alist):
    elems = []
    for elem in alist:
        elems += [tf.reshape(elem,(-1,))]
    return tf.concat(elems, axis=0)


with tf.name_scope("model"):
    print("building models...")
    for config in tqdm.tqdm(all_config, total=config_count):
        R, restart_ind, objective = config
        if short_key:
            config_name = ''.join([''.join([key, str(value)]) for key, value, active in zip(factor_code, config, active_factors) if active])
        else:
            config_name = ''.join([''.join([key, str(value)]) for key, value in zip(factor_code, config)])
                
        with tf.name_scope(config_name):
            Xt[config] = tf.placeholder(dtype, shape=(N, N))
            
            #set coretype
            if coretype == 'canon':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                cores[config] = tn.Canonical(N, K, ranks, orthogonalstyle=tn.CayleyOrthogonal)
            elif coretype == 'perm':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                repranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
                cores[config] = tn.PermutationCore_augmented(N, K, repranks, ranks)
            elif coretype == 'standard':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                cores[config] = tn.Core(N, K, ranks)
            else:
                raise(ValueError)

            coregroup[restart_ind] += [cores[config]]

            #build q model
            q[config] = tn.MPS(N, K, ranks, cores=cores[config])
            
        tfrate = tf.convert_to_tensor(rate, dtype=dtype)
        if decay < 1.:
            learningrate = tf.train.exponential_decay(tfrate, decay_stage, decay_steps, decay)
        else:
            learningrate = tfrate
        
        if optimizer == 'ams':
            stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)
            var_stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        control_samples = q[config].shadowrelax(nsample)
        qtensor[config] = tf.reshape(q[config].populatetensor(),(-1,))
        logq = q[config].batch_logp(all_anchors)
        
        elbo = lambda sample: -q[config].elbo(sample, logp, marginal=marginal)
        loss[config] = tf.reduce_mean(elbo(control_samples[0]))
        dloss[config] = tf.reduce_mean(elbo(control_samples[1]))
        trueloss[config] = -tf.reduce_sum(tf.exp(logq)*(logp_all-logq))
        if objective == 'shadow':
            grad = stepper.compute_gradients(dloss[config], var_list=cores[config].params())
            var_grad = None
            var_reset += [q[config].set_nu(1.), q[config].set_temperature(0.5)]
        elif objective == 'shadow-tight':
            grad = stepper.compute_gradients(dloss[config], var_list=cores[config].params())
            var_grad = None
            var_reset += [q[config].set_nu(1.), q[config].set_temperature(0.1)]
        elif objective == 'relax':
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, _ = RELAX(*relax_params, hard_params=cores[config].params(), var_params=[], weight=q[config].nu)
            var_grad = None
            var_reset += [q[config].set_nu(1.), q[config].set_temperature(0.1)]
        elif objective == 'score':
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, _ = RELAX(*relax_params, hard_params=cores[config].params(), var_params=[], weight=0.)
            var_grad = None
            var_reset += [q[config].set_nu(1.), q[config].set_temperature(0.1)]
        elif objective == 'relax-varreduce':
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, var_grad = RELAX(*relax_params, hard_params=cores[config].params(), var_params=q[config].var_params(), weight=q[config].nu)
            var_reset += [q[config].set_nu(1.), q[config].set_temperature(0.1)]
        elif objective == 'relax-learned':
            control_scale = tf.Variable(0., dtype=dtype)
            control_R = 2
            control_ranks = tuple(min(K**min(r, N-r), control_R) for r in range(N+1))
            control_cores = tn.Core(N, K, control_ranks) 
            control_mps = tn.MPS(N, K, control_ranks, cores=control_cores, normalized=False)
            control = lambda sample: elbo(sample) + control_scale*control_mps.batch_root(sample)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo, fhat=control)
            grad, var_grad = RELAX(*relax_params, hard_params=cores[config].params(), var_params=q[config].var_params() + [control_scale] + control_cores.params(), weight=q[config].nu)
            var_reset += [tf.assign(control_scale, 0.), tf.initialize_variables(control_cores.params())]
        else:
            raise(ValueError)

        
        #step[config] = stepper.apply_gradients(var_grad)
        #residual[config] = tf.linalg.norm(grad-truegrad[config])
        #variance[config] =  
        #bias[config] = residual[config] - variance[config]
        if var_grad is not None:
            var_step[config] = var_stepper.apply_gradients(var_grad)
        else:
            var_step[config] = tf.no_op()

        step[config] = stepper.apply_gradients(grad)

        update[config] = tf.group([step[config], var_step[config]])

        
    
    var_reset += [tf.assign(decay_stage, 0)]
    var_reset = tf.group(var_reset)
    all_steps = tf.group(list(step.values()) + list(var_step.values()) + [increment_decay_stage_op])
    initializers = []
    for index in range(random_restarts):
        initializers += [tn.Initializer(list(coregroup[index]))]
    
    randomize = tf.group([initializer.randomize() for initializer in initializers])
    reset = tf.group([initializer.match() for initializer in initializers])
    def checkpoint(label, sess=None):
        for initializer in initializers:
            initializer.checkpoint_init(label, sess)
    #checkpoint = lambda label: tf.group([initializer.checkpoint_init(label) for initializer in initializers])
    restore = lambda label: tf.group([initializer.restore_init(label) for initializer in initializers])
    init = tf.global_variables_initializer()

    #run all configurations
    column_names = ['loss','dloss','trueloss']
    
    index_c = pd.MultiIndex.from_product(factors + [range(nsteps)], names=factor_names + ['iteration'])
    index_q = pd.MultiIndex.from_product(factors + [range(nsteps),range(K**N)], names=factor_names + ['iteration','state'])
    df_c = pd.DataFrame(np.zeros((config_count*nsteps,len(column_names))), index=index_c, columns=column_names)
    
    qtrace = pd.DataFrame(np.zeros((config_count*nsteps*(K**N),1)), index=index_q, columns=["probability"])

    train_writer = tf.summary.FileWriter('./train', sess.graph)
  
    sess.run(init)
    sess.run(randomize)
    sess.run(reset)
    sess.run(var_reset)
    with sess.as_default():
        checkpoint("initial")
        with tf.name_scope("optimization"):    
            for config in all_config:
                configc = config + (0,)
                lossit, dlossit, truelossit, qtensorit = sess.run([loss[config], dloss[config], trueloss[config], qtensor[config]])
                df_c.loc[configc, 'loss'] = lossit
                df_c.loc[configc, 'dloss'] = dlossit
                df_c.loc[configc, 'trueloss'] = truelossit
                qtrace.loc[configc + (slice(None),), 'probability'] = qtensorit
        
            for it in tqdm.trange(1,nsteps):
                for config in all_config:
                    configc = config + (it,)
                    if timeit:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, lossit, dlossit, truelossit, qtensorit = sess.run([update[config], loss[config], dloss[config], trueloss[config], qtensor[config]],
                                                                             options=run_options, run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step {}, obj {}'.format(it, config[-1]))
                    else:
                        _, lossit, dlossit, truelossit, qtensorit = sess.run([update[config], loss[config], dloss[config], trueloss[config], qtensor[config]])
                    df_c.loc[configc, 'loss'] = lossit
                    df_c.loc[configc, 'dloss'] = dlossit
                    df_c.loc[configc, 'trueloss'] = truelossit
                    qtrace.loc[configc + (slice(None),), 'probability'] = qtensorit
                sess.run(increment_decay_stage_op)
                #residualnorm = np.square(np.linalg.norm(residuals, ord=2, axis=1)).mean()
                    #variance = (1./(ngsamples-1))*np.square(np.linalg.norm(deviations, ord=2, axis=1)).sum()
                    #bias = residualnorm - variance

                    #df_c['residual'][configc] = residual
                    #df_c['variance'][configc] = variance
                    #df_c['bias'][configc] = bias
                    #df_c['obsbias'][configc] = np.linalg.norm(grad0-mean, ord=2)
            
train_writer.close()
save_name = folder + config_full_name + '_tensortrace.pkl'
qdict = {key:tn.packmps("q", val, sess=sess) for key, val in q.items()}
meta = {'name': save_name, 'N': N, 'K': K, 'nsamples': nsample, 'random_restarts': random_restarts, 'coretype': coretype, 'optimizer': optimizer, 'rate': rate, 'decay': decay}
supdict = {'meta': meta, 'ptensor': ptensor, 'df_c':df_c, 'qtrace': qtrace, 'q': qdict, 'init_checkpoints': [initializer.init_checkpoints for initializer in initializers], 'checkpoints': [initializer.checkpoints for initializer in initializers]}
with open(folder + config_full_name + '_tensortrace.pkl','wb') as handle:
    pickle.dump(supdict, handle, protocol=pickle.HIGHEST_PROTOCOL)