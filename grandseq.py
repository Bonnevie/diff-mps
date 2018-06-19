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
N = 34
X = X[:N,:N]

#FLAGS
name = 'permcore' 
version = 1
Ntest = 1 #number of edges to use for testing
K = 2 #number of communities to look for
folder = name + 'V{}K{}'.format(version, K)

#factors variations to run experiments over
random_restarts = 1
nsteps = 20000

rate = 1e-1#[1e-1,1e-2,1e-3]
decay = 1.
decay_steps = nsteps/2.
optimizer = 'ams' #options: ams
nsample = 100
marginal = False
timeit = False

objectives = ['relax-learned']
maxranks = [1,2]
marginals = [False]
unimixes = [False,True]
coretypes = ['canon','perm'] 
inits = ['random']

factor_code = ['R','S','L','M','U','C']
factor_names = ['rank','restarts','objective','marginal','unimix','coretype','init']
factors = [maxranks, range(random_restarts), objectives, marginals, unimixes,coretypes,inits]
short_key = False
active_factors = [len(factor)>1 for factor in factors]
all_config = list(product(*factors))
config_count = np.prod([len(factor) for factor in factors])
config_full_name = ''.join([code + '-'.join([str(fact) for fact in factor]) for code, factor in zip(factor_code, factors)])
copy_writer = []
        
np.random.seed(1)
#tf.reset_default_graph()



#generate mask of observed edges
mask = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
mask = mask < np.sort(mask.ravel())[Ntest]
mask = np.logical_not(np.logical_or(mask,mask.T))
mask = np.triu(mask, 1)
predictionmask = np.triu(~mask, 1)

concentration = 1.
a = 1. + (concentration-1.)*np.eye(K)
b = concentration - (concentration-1.)*np.eye(K) 

q = {}
cores = {}
loss = {}
trueloss = {}
step = {}
var_step = {}
cvweight = {}
qtensor = {}
update = {}
init_opt = {}
predloss = {}

var_reset = []

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

#mask = tf.convert_to_tensor(mask.astype(dtype))
#predictionmask = tf.convert_to_tensor(predictionmask.astype(dtype))

def buildq(config, logp, predlogp, decay_stage):
    R, restart_ind, objective, marginal, unimix, coretype, init = config
    configok = True
    if short_key:
        config_name = ''.join([''.join([key, str(value)]) for key, value, active in zip(factor_code, config, active_factors) if active])
    else:
        config_name = ''.join([''.join([key, str(value)]) for key, value in zip(factor_code, config)])
            
    with tf.name_scope(config_name):
        #set coretype
        if coretype == 'canon':
            ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
            cores = tn.Canonical(N, K, ranks, orthogonalstyle=tn.CayleyOrthogonal)
        elif coretype == 'perm':
            if R == 1:
                return (False,) + 7 * (None,)
            #ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
            ranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
            cores = tn.CanonicalPermutationCore(N, K, ranks)
        elif coretype == 'standard':
            ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
            cores = tn.Core(N, K, ranks)
        else:
            raise(ValueError)

        #build q model
        if unimix:
            q = tn.MPS(N, K, ranks, cores=cores)
        else:
            q = tn.unimix(N, K, ranks, cores=cores)

    
    tfrate = tf.convert_to_tensor(rate, dtype=dtype)
    if decay < 1.:
        learningrate = tf.train.exponential_decay(tfrate, decay_stage, decay_steps, decay)
    else:
        learningrate = tfrate
    
    if optimizer == 'ams':
        beta1=tf.Variable(0.9,dtype='float64', trainable=False)
        beta2=tf.Variable(0.999,dtype='float64', trainable=False)
        epsilon=tf.Variable(0.999,dtype='float64', trainable=False)

        stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        var_stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    control_samples = q.shadowrelax(nsample)
    
    elbo = lambda sample: -q.elbo(sample, logp, marginal=marginal)
    loss = tf.reduce_mean(elbo(control_samples[0]))
    predloss = tf.reduce_mean(predlogp(control_samples[0]))
    dloss = tf.reduce_mean(elbo(control_samples[1]))

    if objective == 'shadow':
        grad = stepper.compute_gradients(dloss, var_list=cores.params())
        var_grad = None
        var_reset = [q.set_nu(1.), q.set_temperature(0.5)]
    elif objective == 'shadow-tight':
        grad = stepper.compute_gradients(dloss, var_list=cores.params())
        var_grad = None
        var_reset = [q.set_nu(1.), q.set_temperature(0.1)]
    elif objective == 'relax':
        relax_params = tn.buildcontrol(control_samples, q.batch_logp, elbo)
        grad, _ = RELAX(*relax_params, hard_params=cores.params(), var_params=[], weight=q.nu)
        var_grad = None
        var_reset = [q.set_nu(1.), q.set_temperature(0.1)]
    elif objective == 'score':
        relax_params = tn.buildcontrol(control_samples, q.batch_logp, elbo)
        grad, _ = RELAX(*relax_params, hard_params=cores.params(), var_params=[], weight=0.)
        var_grad = None
        var_reset = [q.set_nu(1.), q.set_temperature(0.1)]
    elif objective == 'relax-varreduce':
        relax_params = tn.buildcontrol(control_samples, q.batch_logp, elbo)
        grad, var_grad = RELAX(*relax_params, hard_params=cores.params(), var_params=q.var_params(), weight=q.nu)
        var_reset = [q.set_nu(1.), q.set_temperature(0.1)]
    elif objective == 'relax-learned':
        control_scale = tf.Variable(0., dtype=dtype)
        control_R = 2
        control_ranks = tuple(min(K**min(r, N-r), control_R) for r in range(N+1))
        control_cores = tn.Core(N, K, control_ranks) 
        control_mps = tn.MPS(N, K, control_ranks, cores=control_cores, normalized=False)
        control = lambda sample: elbo(sample) + control_scale*control_mps.batch_root(sample)
        relax_params = tn.buildcontrol(control_samples, q.batch_logp, elbo, fhat=control)
        grad, var_grad = RELAX(*relax_params, hard_params=cores.params(), var_params=q.var_params() + [control_scale] + control_cores.params(), weight=q.nu)
        var_reset = [tf.assign(control_scale, 0.), tf.initialize_variables(control_cores.params())]
    else:
        raise(ValueError)

    if init is 'random':
        mode_loss = tf.convert_to_tensor(np.array(0.).astype('float64'))
    elif init is 'rank1':
        mode_loss = tf.reduce_sum(tn.norm_rank1(q, tf.nn.softmax(Z)))
    elif init is 'entropy':
        mode_loss = -(q.marginalentropy())
    elif init is 'expectation':
        mode_loss = -tf.reduce_sum(tf.log(q.batch_contraction(tf.nn.softmax(Z))))

    init_opt = tf.contrib.opt.ScipyOptimizerInterface(mode_loss, var_list=cores.params(),method='CG')
            
    
    #step = stepper.apply_gradients(var_grad)
    #residual = tf.linalg.norm(grad-truegrad)
    #variance =  
    #bias = residual - variance
    if var_grad is not None:
        var_step = var_stepper.apply_gradients(var_grad)
    else:
        var_step = tf.no_op()

    step = stepper.apply_gradients(grad)

    update = tf.group([step, var_step])
    
    return (configok, q, cores, update, loss, predloss, init_opt, var_reset)
        
    
#var_reset += [tf.assign(decay_stage, 0)]
#var_reset = tf.group(var_reset)
#all_steps = tf.group(list(step.values()) + list(var_step.values()) + [increment_decay_stage_op])
#initializers = []
#for index in range(random_restarts):
#    initializers += [tn.Initializer(list(coregroup[index]))]

# randomize = tf.group([initializer.randomize() for initializer in initializers])
# reset = tf.group([initializer.match() for initializer in initializers])
# def checkpoint(label, sess=None):
#     for initializer in initializers:
#         initializer.checkpoint_init(label, sess)
# #checkpoint = lambda label: tf.group([initializer.checkpoint_init(label) for initializer in initializers])
# restore = lambda label: tf.group([initializer.restore_init(label) for initializer in initializers])
# init = tf.global_variables_initializer()

#run all configurations
column_names = ['loss','predloss']

index_c = pd.MultiIndex.from_product(factors + [range(nsteps)], names=factor_names + ['iteration'])
df_c = pd.DataFrame(np.zeros((config_count*nsteps,len(column_names))), index=index_c, columns=column_names)

#train_writer = tf.summary.FileWriter('./train', sess.graph)

for config in all_config:
    tf.reset_default_graph()
    with tf.Session() as sess:        
        tf.set_random_seed(1.)
        p = CollapsedStochasticBlock(N, K, alpha=1, a=a, b=b)
        logp = lambda sample: p.batch_logp(sample, X, observed=mask)
        predlogp = lambda sample: p.batch_logpred(sample, X, observed=predictionmask)
        decay_stage = tf.Variable(1, name='decay_stage', trainable=False, dtype=tf.int32)
        configok, q, cores, update, loss, predloss, init_opt, var_reset = buildq(config, logp, predlogp, decay_stage)            
        if configok:    
            increment_decay_stage_op = tf.assign(decay_stage, decay_stage+1)
            var_reset += [increment_decay_stage_op]
            init = tf.global_variables_initializer()
            sess.graph.finalize()
            sess.run(init)
            sess.run(var_reset)
            configc = config + (0,)
            init_opt.minimize()
            lossit, predlossit = sess.run([loss, predloss])
            df_c.loc[configc, 'loss'] = lossit
            df_c.loc[configc, 'predloss'] = predlossit    
            for it in tqdm.trange(1,nsteps):
                configc = config + (it,)
                _, lossit, predlossit = sess.run([update, loss, predloss])
                df_c.loc[configc, 'loss'] = lossit
                df_c.loc[configc, 'predloss'] = predlossit
                sess.run(increment_decay_stage_op)
                
#train_writer.close()
save_name = folder + config_full_name + '_grandseq.pkl'
qdict = {key:tn.packmps("q", val, sess=sess) for key, val in q.items()}
meta = {'name': save_name, 'N': N, 'K': K, 'nsamples': nsample, 'random_restarts': random_restarts, 'coretype': coretype, 'optimizer': optimizer, 'rate': rate, 'decay': decay}
supdict = {'meta': meta, 'df_c':df_c, 'q': qdict, 'init_checkpoints': [initializer.init_checkpoints for initializer in initializers], 'checkpoints': [initializer.checkpoints for initializer in initializers]}
with open(folder + config_full_name + '_grandseq.pkl','wb') as handle:
    pickle.dump(supdict, handle, protocol=pickle.HIGHEST_PROTOCOL)