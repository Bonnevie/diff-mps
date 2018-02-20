import numpy as np
import tensorflow as tf
from relaxflow.reparam import CategoricalReparam
#from relaxflow.relax import RELAX, CategoricalReparam#, categorical_forward, categorical_backward
import time
dtype = 'float64'

import tqdm

from collapsedclustering import CollapsedStochasticBlock, KLcorrectedBound
import tensornets as tn
from edward.models import MultivariateNormalTriL, Dirichlet, WishartCholesky, ParamMixture

name = 'sym'
datatype = 'karate'
coretype = 'Tperm' #'Tperm', 'Tcanon', ''
N = 10
K = 3
maxranks = [3, 9, 27]#[1,6,9,12,15,18,21,24]
nsamples=100
steps = 5000
copies = 5
nmodes = 0
runs = 10
version = 2
rate = 0.001

folder = name + '{}D{}V{}N{}K{}S{}R{}'.format(coretype,datatype,version, N, K, nsamples, '-'.join(str(rank) for rank in maxranks))
#cap core rank at R

timeit = False
calculate_true = False

np.random.seed(1)

#5-22
if datatype is 'random':
    np.random.seed(1)

    #5-22
    X = np.random.randn(*(N, N))
    X =(1.-np.eye(N))*(np.tril(X) + np.tril(X).T)

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

init_types = ['random', 'expectation'] #rank1
cores = {t: {} for t in init_types}
q = {t: {} for t in init_types}
softelbo = {t: {} for t in init_types}
opt = {t: {} for t in init_types}
step = {t: {} for t in init_types}
equalize_ops = []
step_list = []
init_opt = {t: {} for t in init_types}

with tf.name_scope("model"):
    p = CollapsedStochasticBlock(N, K, alpha=100, a=10, b=10)
    Z = tf.Variable(tf.random_normal((len(maxranks),nmodes, N, K), dtype=dtype))

    if nmodes>0:
        bounds = -tf.reduce_mean([KLcorrectedBound(p, X, [z]).bound 
                                for zstack in tf.unstack(Z, axis=0) for z in tf.unstack(zstack, axis=0)])
        mode_opt = tf.contrib.opt.ScipyOptimizerInterface(bounds, var_list=[Z])
        
    Xt = tf.constant(X)
    print("building models...")
    for init in tqdm.tqdm(init_types, desc='init', total=len(init_types)):
        with tf.name_scope('init_' + init):
            for R, Zr in tqdm.tqdm(zip(maxranks, tf.unstack(Z, axis=0)), desc='rank', total=len(maxranks)):
                with tf.name_scope("rank"+str(R)):
                    if coretype is 'Tcanon':
                        ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))#(1,) + tuple() + (1,)
                        cores[init][R] = [tn.Canonical(N, K, ranks, orthogonalstyle=tn.CayleyOrthogonal) for copy in range(copies)]
                    elif coretype is 'Tperm':
                        ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                        repranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
                        cores[init][R] = [tn.PermutationCore_augmented(N, K, repranks, ranks) for copy in range(copies)]
                    else:
                        ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                        cores[init][R] = [tn.Core(N, K, ranks)  for copy in range(copies)]
                    q[init][R] = [tn.MPS(N, K, ranks, cores=core) for core in cores[init][R]]
                    softelbo[init][R] = [tf.reduce_mean(qi.elbo(lambda sample: p.batch_logp(sample, Xt), nsamples=nsamples, fold=False)) for qi in q[init][R]]
                    #softelbo[init][R] = tf.reduce_mean(q[init][R].elbo(lambda sample: 0., nsamples=nsamples, fold=False))
                    if init is 'random':
                        mode_loss = tf.convert_to_tensor(np.array(0.).astype('float64'))
                    elif init is 'rank1':
                        mode_loss = [tf.reduce_sum(tn.norm_rank1(qi, tf.nn.softmax(Zr))) for qi in q[init][R]]
                    elif init is 'entropy':
                        mode_loss = [-(qi.marginalentropy()) for qi in q[init][R]]
                    elif init is 'expectation':
                        mode_loss = [-tf.reduce_sum(qi.batch_contraction(tf.nn.softmax(Zr))) for qi in q[init][R]]
                    #mode_loss = tf.convert_to_tensor(np.array(0.).astype('float64'))
                    init_opt[init][R] = [tf.contrib.opt.ScipyOptimizerInterface(mode_loss, var_list=core.params()) for core in cores[init][R]]
                    opt[init][R] = [tf.train.AdamOptimizer(learning_rate=rate) for copy in range(copies)]
                    step[init][R] = [opti.minimize(-softelboi, var_list=core.params()) for opti, softelboi, core in zip(opt[init][R], softelbo[init][R], cores[init][R])]
                    for copy in range(copies):
                        tf.summary.scalar('ELBO{}'.format(copy), softelbo[init][R][copy])
                    if copies>1:
                        tf.summary.scalar('ELBO_mean', tf.reduce_mean(softelbo[init][R]))
                    step_list += step[init][R]




    if timeit:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    train = tf.group(*step_list)
    summaries = tf.summary.merge_all()
    
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)

    with tf.name_scope("truemodel"):
        if calculate_true:
            with sess.as_default():
                ptensor = p.populatetensor(X)
                pcore, pmps = tn.full2TT(np.sqrt(ptensor))
                true_loss_op = tf.reduce_mean(pmps.elbo(lambda sample: p.batch_logp(sample, Xt), nsamples=nsamples, fold=False))

    initialize = tf.global_variables_initializer()
    sess.run(initialize)

    if calculate_true:
        true_loss = sess.run(true_loss_op)
    with tf.name_scope("optimization"):    
        Q = {}
        print("Starting optimization.")
        for run in tqdm.tqdm(range(runs), desc="Run", total=runs):
            writer = tf.summary.FileWriter('./train/' + folder + 'run' + str(run), sess.graph)
            sess.run(initialize)
            if nmodes>0:
                mode_opt.minimize(sess)
            print('Initializing...')
            for init in init_types:
                for R in tqdm.tqdm(maxranks):
                    for copy in range(copies):
                        init_opt[init][R][copy].minimize(sess)
            
            print('Starting gradient ascent...')
            for it in tqdm.tqdm(range(steps), desc='Optimization step', total=steps, leave=False):
                if timeit:
                    _, it_summary = sess.run([train, summaries],
                                                options=run_options,
                                                run_metadata=run_metadata)
                else:
                    _, it_summary = sess.run([train, summaries])
                writer.add_summary(it_summary, it)
                if timeit:
                    writer.add_run_metadata(run_metadata, "trace{}".format(it))
            Q[run] = q
