import numpy as np
import tensorflow as tf
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

#karate = karate_club_graph()
#X = adjacency_matrix(karate).toarray().astype('float64')
X = np.ones((4,4))
for i in range(4):
    X[3-i,i] = 0.
N = 4
X = X[:N,:N]

#FLAGS
name = 'trackbias' 
version = 1
Ntest = 0 #number of edges to use for testing
K = 2 #number of communities to look for
folder = name + 'V{}K{}'.format(version, K)

#factors variations to run experiments over
random_restarts = 3
varrelaxsteps = 0
ngsamples = 20000


decay_steps = 20000
optimizer = 'ams' #options: ams
nsample = 100
#nsample_vargrad = 100
coretype = 'canon'
train_temp = True
flow_stages = 10
flow_length = 5

learningrates = [1e-1]#[1e-1,1e-2,1e-3]
decays = [1.]
starttemps = [0.5]
objectives = ['relax']#['shadow','shadow-tight', 'score', 'relax', 'relax-tight', 'relax-varreduce','relax-learned']#['shadow', 'shadow-tight', 'score', 'relax', 'relax-marginal', 'relax-varreduce', 'relax-marginal-varreduce', 'relax-learned', 'relax-marginal-learned']
#Options are: '' for ordinary cores, canon' for canonical, and 'perm' for permutation-free
maxranks = [2]#,12,15,18]


factor_code = ['R','S','L','A','D','T']
factor_names = ['rank','restarts','objective','learningrate','decay','starttemp']
factors = [maxranks, range(random_restarts), objectives,learningrates, decays, starttemps]
short_key = False
active_factors = [len(factor)>1 for factor in factors]
all_config = list(product(*factors))
config_count = np.prod([len(factor) for factor in factors])
config_full_name = ''.join([code + '-'.join([str(fact) for fact in (factor if len(factor)<4 else [len(factor)])]) for code, factor in zip(factor_code, factors)])
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
logp_all = logp(all_anchors)


beta1=tf.Variable(0.9,dtype='float64')
beta2=tf.Variable(0.999,dtype='float64')
epsilon=tf.Variable(0.999,dtype='float64')

q = {}
Xt = {}
cores = {}
step = {}
trueloss = {}
truegrad = {}
truegradnorm = {}
flatgrad = {}
cvweight = {}
control_scale = {}
control_weight = {}

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
        R, restart_ind, objective, learningrate0, decay, starttemp = config
        if short_key:
            config_name = ''.join([''.join([key, str(value)]) for key, value, active in zip(factor_code, config, active_factors) if active])
        else:
            config_name = ''.join([''.join([key, str(value)]) for key, value in zip(factor_code, config)])
                
        with tf.name_scope(config_name):
            Xt[config] = tf.placeholder(dtype, shape=(N, N))
            
            #set coretype
            if coretype is 'canon':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                cores[config] = tn.Canonical(N, K, ranks, orthogonalstyle=tn.OrthogonalMatrix)
            elif coretype is 'perm':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                repranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
                cores[config] = tn.PermutationCore_augmented(N, K, repranks, ranks)
            else:
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                cores[config] = tn.Core(N, K, ranks)

            coregroup[restart_ind] += [cores[config]]

            #build q model
            q[config] = tn.MPS(N, K, ranks, cores=cores[config])
            
        tfrate = tf.convert_to_tensor(learningrate0, dtype=dtype)
        if decay < 1.:
            learningrate = tf.train.exponential_decay(tfrate, decay_stage, decay_steps, decay)
        else:
            learningrate = tfrate#/tf.sqrt(tf.cast(decay_stage, 'float64'))
        
        if optimizer == 'ams':
            stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)
        
        if train_temp:
            var_params = q[config].var_params()
        else:
            var_params = [q[config]._nuvar]

        control_samples = q[config].shadowrelax(nsample)
        #control_samples_var = q[config].shadowrelax(nsample_vargrad)
        
        logq = q[config].batch_logp(all_anchors)
        trueloss[config] =  -tf.reduce_mean(tf.exp(logq)*(logp_all-logq))
        truegrad[config] = flattenlist(tf.gradients(trueloss[config], cores[config].params()))
        truegradnorm[config] = tf.linalg.norm(truegrad[config])
        var_reset += [q[config].set_temperature(starttemp)]
        if objective == 'shadow':
            loss = -tf.reduce_mean(q[config].elbo(control_samples[1], logp, marginal=False))
            grad = stepper.compute_gradients(loss, var_list=q[config].params())
            var_grad = None
            var_reset += [q[config].set_nu(1.)]
        elif objective == 'relax':
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=False)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, _ = RELAX(*relax_params, hard_params=q[config].params(), var_params=[], weight=q[config].nu)
            var_grad = None
            var_reset += [q[config].set_nu(1.)]
        elif objective == 'score':
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=False)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, _ = RELAX(*relax_params, hard_params=q[config].params(), var_params=[], weight=0.)
            var_grad = None
            var_reset += [q[config].set_nu(1.)]
        elif objective == 'relax-varreduce':
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=False)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, var_grad = RELAX(*relax_params, hard_params=q[config].params(), var_params=var_params, weight=q[config].nu)
            var_reset += [q[config].set_nu(1.)]
        elif objective == 'relax-marginal':
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=True)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, _ = RELAX(*relax_params, hard_params=q[config].params(), var_params=[], weight=q[config].nu)
            var_grad = None
            var_reset += [q[config].set_nu(1.)]
        elif objective == 'relax-marginal-varreduce':
            cvweight[config] = tf.Variable(1., dtype=dtype)
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=True, cvweight=cvweight[config])
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo)
            grad, var_grad = RELAX(*relax_params, hard_params=q[config].params(), var_params=var_params + [cvweight[config]], weight=q[config].nu)
            var_reset += [q[config].set_nu(1.), tf.assign(cvweight[config], 1.)]
        elif objective == 'relax-learned':
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=False)
            control_weight[config] = tf.Variable(0., dtype=dtype)
            control_R = 2
            control_ranks = tuple(min(K**min(r, N-r), control_R) for r in range(N+1))
            control_cores = tn.Canonical(N, K, control_ranks) 
            control_mps = tn.MPS(N, K, control_ranks, cores=control_cores, normalized=False)
            control_scale[config] = control_mps._scale()
            control = lambda sample: elbo(sample) + control_weight[config]*control_mps.batch_root(sample)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo, fhat=control)
            grad, var_grad = RELAX(*relax_params, hard_params=q[config].params(), var_params=var_params + [control_weight[config]] + control_cores.params(), weight=q[config].nu)
            var_reset += [q[config].set_nu(1.), tf.assign(control_weight[config], 0.), tf.initialize_variables(control_cores.params())]
        elif objective == 'relax-marginal-learned':
            cvweight[config] = tf.Variable(1., dtype=dtype)
            elbo = lambda sample: -q[config].elbo(sample, logp, marginal=True, cvweight=cvweight[config])
            control_weight[config] = tf.Variable(0., dtype=dtype)
            control_R = 2
            control_ranks = tuple(min(K**min(r, N-r), control_R) for r in range(N+1))
            control_cores = tn.Canonical(N, K, control_ranks) 
            control_mps = tn.MPS(N, K, control_ranks, cores=control_cores, normalized=False)
            control_scale[config] = control_mps._scale()
            control = lambda sample: elbo(sample) + control_weight[config]*control_mps.batch_root(sample)
            relax_params = tn.buildcontrol(control_samples, q[config].batch_logp, elbo, fhat=control)
            grad, var_grad = RELAX(*relax_params, hard_params=q[config].params(), var_params=var_params + [control_weight[config], cvweight[config]] + control_cores.params(), weight=q[config].nu)
            var_reset += [q[config].set_nu(1.), tf.assign(cvweight[config], 1.), tf.assign(control_weight[config], 0.), tf.initialize_variables(control_cores.params())]
        else:
            raise(ValueError)

        flatgrad[config] = flattengrad(grad)
        
        #step[config] = stepper.apply_gradients(var_grad)
        #residual[config] = tf.linalg.norm(grad-truegrad[config])
        #variance[config] =  
        #bias[config] = residual[config] - variance[config]
        if var_grad is not None:
            step[config] = stepper.apply_gradients(var_grad)
        else:
            step[config] = tf.no_op()

        
    
    var_reset += [tf.assign(decay_stage, 1)]
    var_reset = tf.group(var_reset)
    all_steps = tf.group(list(step.values()) + [increment_decay_stage_op])
    initializers = []
    for index in range(random_restarts):
        initializers += [tn.Initializer(list(coregroup[index]))]
    
    print("building baselines...")
    baselines = {}
    baseopt = {}
    baseopt_flat = []
    for index, initializer in enumerate(initializers):
        baselines[index] = {}
        baseopt[index] = {}
        for key, core in initializer.init_cores.items():
            N, K, ranks, _ = key
            baselines[index][key] = tn.MPS(N, K, ranks, cores=core)
            logq = baselines[index][key].batch_logp(all_anchors)
            loss = -tf.reduce_mean(tf.exp(logq)*(logp_all-logq))
            baseopt[index][key] = lambda it: tf.contrib.opt.ScipyOptimizerInterface(loss, core.params(), options={'maxiter': it})
            baseopt_flat += [baseopt[index][key]]
    
    randomize = tf.group([initializer.randomize() for initializer in initializers])
    reset = tf.group([initializer.match() for initializer in initializers])
    def checkpoint(label, sess=None):
        for initializer in initializers:
            initializer.checkpoint_init(label, sess)
    #checkpoint = lambda label: tf.group([initializer.checkpoint_init(label) for initializer in initializers])
    restore = lambda label: tf.group([initializer.restore_init(label) for initializer in initializers])
    init = tf.global_variables_initializer()

    #run all configurations
    column_names = ['residual', 'variance', 'bias','obsbias']
    stages = ['initial'] + ['flow{}'.format(i) for i in range(flow_stages)]# + ['converged']
    num_vars = np.prod(flatgrad[config].get_shape()).value
    index_c = pd.MultiIndex.from_product(factors + [stages] + [np.arange(num_vars)], names=factor_names + ['stage', 'parameter'])
    df_c = pd.DataFrame(np.zeros((config_count*len(stages)*num_vars,len(column_names))), index=index_c, columns=column_names)
    
    
    vartrace = {}

    sess = tf.Session()
    sess.run(init)
    sess.run(randomize)
    sess.run(reset)
    with sess.as_default():
        with tf.name_scope("optimization"):    
            checkpoint("initial")
            for stage in range(flow_stages):
                for bopt in baseopt_flat:
                    bopt(flow_length).minimize()
                checkpoint("flow{}".format(stage))
            for bopt in baseopt_flat:
                bopt(1000).minimize()
            checkpoint("converged")
            
            for checkin in tqdm.tqdm(stages):
                print("at {}".format(checkin))
                sess.run(restore(checkin))
                sess.run(reset)
                sess.run(var_reset)
                for it in tqdm.trange(varrelaxsteps):
                    sess.run(all_steps)
                    for config in all_config:
                        objective = config[2]
                        configc = config + (checkin, )
                        try:
                            vartrace[configc]['temp'] += [sess.run(q[config].temperatures)[0]]
                            vartrace[configc]['nu'] += [sess.run(q[config].nu)]
                            if objective == 'relax-marginal-varreduce':
                                vartrace[configc]['cvweight'] += [sess.run(cvweight[config])]
                            if objective == 'relax-learned':
                                vartrace[configc]['control_weight'] += [sess.run(control_weight[config])]
                                vartrace[configc]['control_scale'] += [sess.run(control_scale[config])]
                        except KeyError: 
                            vartrace[configc] = {}
                            vartrace[configc]['temp'] = [sess.run(q[config].temperatures)[0]]
                            vartrace[configc]['nu'] = [sess.run(q[config].nu)]
                            if objective == 'relax-marginal-varreduce':
                                vartrace[configc]['cvweight'] = [sess.run(cvweight[config])]
                            if objective == 'relax-learned':
                                vartrace[configc]['control_weight'] = [sess.run(control_weight[config])]
                                vartrace[configc]['control_scale'] = [sess.run(control_scale[config])]
                for config in tqdm.tqdm(all_config, total=config_count):
                    configc = config + (checkin, slice(None))
                    gsamples = np.stack([sess.run(flatgrad[config]) for _ in range(ngsamples)])
                    grad0 = sess.run(truegrad[config])
                    residuals = gsamples - grad0[None,:]
                    mean = np.mean(gsamples, axis=0)
                    deviations = gsamples - mean[None, :]
                    
                    residual = np.square(residuals).mean(axis=0)
                    variance = (1./(ngsamples-1))*np.square(deviations).sum(axis=0)
                    bias = residual - variance
                    
                    df_c.loc[configc, 'residual'] = residual
                    df_c.loc[configc, 'variance'] = variance
                    df_c.loc[configc, 'bias'] = bias
                    df_c.loc[configc, 'obsbias'] = grad0 - mean
                    #residualnorm = np.square(np.linalg.norm(residuals, ord=2, axis=1)).mean()
                    #variance = (1./(ngsamples-1))*np.square(np.linalg.norm(deviations, ord=2, axis=1)).sum()
                    #bias = residualnorm - variance

                    #df_c['residual'][configc] = residual
                    #df_c['variance'][configc] = variance
                    #df_c['bias'][configc] = bias
                    #df_c['obsbias'][configc] = np.linalg.norm(grad0-mean, ord=2)
            
save_name = folder + config_full_name + '_varreduce.pkl'
qdict = {key:tn.packmps("q", val, sess=sess) for key, val in q.items()}
meta = {'name': save_name, 'N': N, 'K': K, 'stages': stages, 'flow_stages': flow_stages, 'flow_length': flow_length, 'config_count': config_count, 'ngsamples': ngsamples, 'factors': factors, 'factor_names': factor_names, 'nsamples': nsample, 'random_restarts': random_restarts, 'coretype': coretype, 'optimizer': optimizer}
supdict = {'meta': meta, 'df_c':df_c, 'q': qdict, 'init_checkpoints': [initializer.init_checkpoints for initializer in initializers], 'checkpoints': [initializer.checkpoints for initializer in initializers], 'traces': vartrace}
with open(folder + config_full_name + '_varreduce.pkl','wb') as handle:
    pickle.dump(supdict, handle, protocol=pickle.HIGHEST_PROTOCOL)