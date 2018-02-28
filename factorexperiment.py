import numpy as np
import tensorflow as tf
from relaxflow.reparam import CategoricalReparam
import time
dtype = 'float64'

import tqdm

import matplotlib.pyplot as plt

from collapsedclustering import CollapsedStochasticBlock, KLcorrectedBound
import tensornets as tn

from itertools import product

#FLAGS
timeit = False #log compute time of every op for Tensorboard visualization (costly)
calculate_true = True #calculate true tensor and MPS (WARNING: can cause OOM for N>>14)
do_anneal = False #do entropy annealing
name = 'lowrank' 
datatype = 'random' #'random', 'blocked', or 'karate'
version = 3
N = 7 #number of vertices in graph
Ntest = 0 #number of edges to use for testing
K = 3 #number of communities to look for
nsamples=100 #number of gradient samples per iteration
steps = 10000 #number of optimization iterations
decay_steps = 600 #number of steps between learning rate decay
anneal_decay_steps = 100 #number of steps beteween anneal decay
projection_steps = 1 #how often to project 
rate = 0.001 #learning rate used in optimizer
anneal_rate = 70. #initial annealing inverse temperature
nmodes = 20 #modes to find using standard VB to use in initialization schemes

folder = name + 'D{}V{}N{}K{}S{}'.format(datatype,version, N, K, nsamples)

#factors variations to run experiments over
copies = 1
coretypes = [''] #types of cores to try 
#Options are: '' for ordinary cores, canon' for canonical, and 'perm' for permutation-free
inittypes = ['random'] #whether to run an initialization routine
#Options are: 'random' for random values, 'entropy' for maximum marginal entropy, 'rank1' for minimum norm to tensor mixture of 10 modes the model,
#'expectation' for maximum expectation of the the previous tensor mixture under q.
maxranks = [9]
optimizers = ['Adam'] #optimizer to use for the stochastic gradients
#Options are: 'SGD', 'Adadelta', 'Adam'
objective = ['']
projection = [False]
decay_rates = [0.95] #decay rates for learning rate
anneal_rates = [1.] #decay rates for entropy annealing schedule

factor_code = ['T','I','O','B', 'P', 'L','A','R']
factors = [coretypes, inittypes, optimizers, objective, projection, decay_rates, anneal_rates, maxranks]
short_key = True
active_factors = [len(factor)>1 for factor in factors]
all_config = product(*factors)
config_count = np.prod([len(factor) for factor in factors])
copy_writer = []
        
np.random.seed(1)
tf.reset_default_graph()

#generate mask of observed edges
if Ntest > 0:
    mask = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
    mask = mask < np.sort(mask.ravel())[Ntest]
    mask = np.logical_not(np.logical_or(mask,mask.T))
else:
    mask = np.ones((N, N), dtype=bool)
mask = np.triu(mask, 1)
predictionmask = ~mask
mask = tf.convert_to_tensor(mask.astype(dtype))
predictionmask = tf.convert_to_tensor(predictionmask.astype(dtype))
if datatype is 'random':
    X = np.random.randn(*(N, N)) > 0.5
    X =(1.-np.eye(N))*(np.tril(X) + np.tril(X).T)
if datatype is 'blocked':
    split = N//(K+1)
    remainder = N - K*split
    strengths = 0.1*np.one((N-remainder, N-remainder)) + 0.8*np.eye(N-remainder)
    strengths = np.bmat([[strengths, 0.5*np.ones((N-remainder,remainder))],[0.5*np.ones((remainder,N-remainder)), 0.5*np.ones((remainder, remainder))]])
    X = strengths > np.random.randn(N, N)
    X = (1.-np.eye(N))*(np.tril(X) + np.tril(X).T)
elif datatype is 'karate':
    Akarate = np.array([   [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1., 1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
        [ 1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.],
        [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1., 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0., 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1., 0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  0., 0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0., 1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0., 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.]])
    X = Akarate[:N, :N]


cores = {}
q = {}
softelbo = {}
softpred = {}
softobj = {}
pred = {}
opt = {}
step = {}
equalize_ops = []
init_opt = {}
proj_opt = {}
alpha_update = {}
qtensor = {}

copy_summary = [[] for copy in range(copies)]

global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
increment_global_step_op = tf.assign(global_step, global_step+1)
with tf.name_scope("model"):
    p = CollapsedStochasticBlock(N, K, alpha=1, a=1, b=1)
    #p = CollapsedStochasticBlock(N, K, alpha=100, a=10, b=10)
    Z = tf.Variable(2.*tf.random_normal((nmodes, N, K), dtype=dtype))
    antiZ = 10.*tf.random_normal((nmodes, N, K), dtype=dtype)
    anchors = tf.concat([tf.nn.softmax(Z),tf.nn.softmax(antiZ)],axis=0)
    bounds = -tf.reduce_mean([KLcorrectedBound(p, X, [z]).bound for z in tf.unstack(Z, axis=0)])
    mode_opt = tf.contrib.opt.ScipyOptimizerInterface(bounds, var_list=[Z])

    Xt = tf.constant(X)
    print("building models...")
    for config in tqdm.tqdm(all_config, total=config_count):
        coretype, init, opt_name, objective, do_project, decay_rate, anneal_decay_rate, R = config
        if short_key:
            config_name = ''.join([''.join([key, str(value)]) for key, value, active in zip(factor_code, config, active_factors) if active])
        else:
            config_name = ''.join([''.join([key, str(value)]) for key, value in zip(factor_code, config)])

        with tf.name_scope(config_name):
            for copy in range(copies):
                if short_key:
                    configc = tuple([factor for factor, active in zip(config, active_factors) if active] + ([copy] if copies>1 else []))
                else:
                    configc = config + (copy,)
                #set coretype
                if coretype is 'canon':
                    ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                    cores[configc] = tn.Canonical(N, K, ranks, orthogonalstyle=tn.OrthogonalMatrix)
                elif coretype is 'perm':
                    ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                    repranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
                    cores[configc] = tn.PermutationCore_augmented(N, K, repranks, ranks)
                else:
                    ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                    cores[configc] = tn.Core(N, K, ranks)
                
                #build q model
                if do_anneal:
                    anneal_invtemp = 1.+tf.train.exponential_decay(tf.cast(anneal_rate, dtype), global_step, anneal_decay_steps, anneal_decay_rate, staircase=True)
                else:
                    anneal_invtemp = 1.
                q[configc] = tn.MPS(N, K, ranks, cores=cores[configc])
                cvweight = 1.
                if objective is '':
                    objective, elbo, loss, entropy, marginalentropy, marginalcv = (q[configc].elbo(lambda sample: p.batch_logp(sample, Xt, observed=mask), nsamples=nsamples, fold=False, report=True, cvweight=cvweight, invtemp=anneal_invtemp))
                elif objective is 'modes':
                    objective, elbo, loss, entropy, marginalentropy, marginalcv = (q[configc].elbowithmodes(lambda sample: p.batch_logp(sample, Xt, observed=mask), anchors, nsamples=nsamples, fold=False, report=True, cvweight=cvweight, invtemp=anneal_invtemp))
                #objective, elbo, loss, entropy, marginalentropy, marginalcv = (q[configc].elbowithmodes(lambda sample: p.batch_logp(sample, Xt, observed=mask), anchors, nsamples=nsamples, fold=False, report=True, cvweight=cvweight, invtemp=anneal_invtemp))
                pred = q[configc].pred(lambda sample: p.batch_logp(sample, Xt, observed=predictionmask), nsamples=nsamples, fold=False)
                softpred[configc] = tf.reduce_mean(pred)
                softelbo[configc] = tf.reduce_mean(elbo)
                softobj[configc] = tf.reduce_mean(objective)
                if calculate_true:
                    qtensor[configc] = q[configc].populatetensor()
                

                #set initialization mode
                if init is 'random':
                    mode_loss = tf.convert_to_tensor(np.array(0.).astype('float64'))
                elif init is 'rank1':
                    mode_loss = tf.reduce_sum(tn.norm_rank1(q[configc], tf.nn.softmax(Z)))
                elif init is 'entropy':
                    mode_loss = -(q[configc].marginalentropy())
                elif init is 'expectation':
                    mode_loss = -tf.reduce_sum(tf.log(q[configc].batch_contraction(tf.nn.softmax(Z))))
                    
                init_opt[configc] = tf.contrib.opt.ScipyOptimizerInterface(mode_loss, var_list=cores[configc].params(),method='CG')
                
                if do_project:
                    project_loss = -(q[configc].marginalentropy())
                    proj_opt[configc] = tf.contrib.opt.ScipyOptimizerInterface(project_loss, var_list=cores[configc].params(),method='CG')
                    
                #build optimizers
                decayed_rate = tf.train.exponential_decay(tf.cast(rate, dtype), global_step, decay_steps, decay_rate, staircase=True)
                if opt_name is 'Adam':
                    opt[configc] = tf.train.AdamOptimizer(learning_rate=decayed_rate)
                    step[configc] = opt[configc].minimize(-softobj[configc], var_list=cores[configc].params())
                elif opt_name is 'Adadelta':
                    opt[configc] = tf.train.AdamOptimizer(learning_rate=decayed_rate)
                    step[configc] = opt[configc].minimize(-softobj[configc], var_list=cores[configc].params())
                elif opt_name is 'SGD':
                    opt[configc] = tf.train.GradientDescentOptimizer(learning_rate=decayed_rate)
                    step[configc] = opt[configc].minimize(-softobj[configc], var_list=cores[configc].params())
                #build summaries
                copy_summary[copy] += [tf.summary.scalar('objective', tf.reduce_mean(objective)),
                                tf.summary.scalar('ELBO', tf.reduce_mean(elbo)),
                                tf.summary.scalar('pred', tf.reduce_mean(pred)),
                                tf.summary.scalar('logp', tf.reduce_mean(loss)),
                                tf.summary.scalar('entropy', tf.reduce_mean(entropy)),
                                tf.summary.scalar('marginalentropy', tf.reduce_mean(marginalentropy))]
    if timeit:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    train = tf.group(*step.values())

    summaries = [tf.summary.merge(copy_summary_n) for copy_summary_n in copy_summary]
    sess = tf.Session()
    with tf.name_scope("truemodel"):
        if calculate_true:
            with sess.as_default():
                logptensor = p.populatetensor(X, observed=mask)
                logZ = np.logaddexp.reduce(logptensor.ravel())
                ptensor = np.exp(logptensor - logZ)
                pcore, pmps = tn.full2TT(np.sqrt(ptensor))
                true_loss_op = tf.reduce_mean(pmps.elbo(lambda sample: p.batch_logp(sample, Xt, observed=mask), nsamples=nsamples, fold=False))

    init = tf.global_variables_initializer()
    sess.run(init)

    if calculate_true:
        true_loss = sess.run(true_loss_op)
    with tf.name_scope("optimization"):    
        print("Starting optimization.")
        copy_writer = []
        for copy in range(copies):
            copy_writer.append(tf.summary.FileWriter('./train/' + folder + '_copy{}'.format(copy), sess.graph))

        mode_opt.minimize(sess)
        print('Initializing...')
        for init in init_opt.values():
            init.minimize(sess)
        
        print('Starting gradient ascent...')
        if calculate_true:
            print("ELBO at optimum: {}".format(true_loss))

        for it in tqdm.tqdm(range(steps), desc='Optimization step', total=steps, leave=False):
            if timeit:
                _, _, it_summary = sess.run([train, increment_global_step_op, summaries],
                                            options=run_options,
                                            run_metadata=run_metadata)
            else:
                _, _, it_summary = sess.run([train, increment_global_step_op, summaries])
            for writer, it_summary_copy in zip(copy_writer, it_summary):
                writer.add_summary(it_summary_copy, it)
            if timeit:
                writer.add_run_metadata(run_metadata, "trace{}".format(it))
            if it % projection_steps == 0:
                for proj in proj_opt.values():
                    proj.minimize(sess)
Zq = q[()].populatetensor().eval(session=sess)
