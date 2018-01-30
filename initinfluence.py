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

name = 'initinfluence'
coretype = 'Tcanon' #'Tperm', 'Tcanon', ''
N = 10
K = 3
maxranks = [1, 3, 9, 27]#[1,6,9,12,15,18,21,24]
nsamples=100
steps = 1000
nmodes = 10
runs = 10
version = 1
rate = 0.001

folder = name + '{}V{}N{}K{}S{}R{}'.format(coretype,version, N, K, nsamples, '-'.join(str(rank) for rank in maxranks))
#cap core rank at R

timeit = False
calculate_true = True

np.random.seed(1)

#5-22
X = np.random.randn(*(N, N))
X =(1.-np.eye(N))*(np.tril(X) + np.tril(X).T)
tf.reset_default_graph()

init_types = ['random', 'entropy'] #rank1
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
                        cores[init][R] = tn.Canonical(N, K, ranks, orthogonalstyle=tn.CayleyOrthogonal)
                    elif coretype is 'Tperm':
                        ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                        repranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
                        cores[init][R] = tn.PermutationCore_augmented(N, K, repranks, ranks)
                    else:
                        ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                        cores[init][R] = tn.Core(N, K, ranks)
                    q[init][R] = tn.MPS(N, K, ranks, cores=cores[init][R])
                    softelbo[init][R] = tf.reduce_mean(q[init][R].elbo(lambda sample: p.batch_logp(sample, Xt), nsamples=nsamples, fold=False))
                    #softelbo[init][R] = tf.reduce_mean(q[init][R].elbo(lambda sample: 0., nsamples=nsamples, fold=False))
                    if init is 'random':
                        mode_loss = tf.convert_to_tensor(np.array(0.).astype('float64'))
                    elif init is 'rank1':
                        mode_loss = tf.reduce_sum(tn.norm_rank1(q[init][R], tf.nn.softmax(Zr))) 
                    elif init is 'entropy':
                        mode_loss = -(q[init][R].marginalentropy())
                    elif init is 'expectation':
                        mode_loss = -tf.reduce_sum(q[init][R].batch_contraction(tf.nn.softmax(Zr)))
                    #mode_loss = tf.convert_to_tensor(np.array(0.).astype('float64'))
                    init_opt[init][R] = tf.contrib.opt.ScipyOptimizerInterface(mode_loss, var_list=cores[init][R].params())
                    #init_opt[init][R] = tf.contrib.opt.ScipyOptimizerInterface()
                    opt[init][R] = tf.train.AdamOptimizer(learning_rate=rate)
                    step[init][R] = opt[init][R].minimize(-softelbo[init][R], var_list=cores[init][R].params())
                    tf.summary.scalar('ELBO', softelbo[init][R])
                    step_list += [step[init][R]]




    if timeit:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    train = tf.group(*step_list)
    summaries = tf.summary.merge_all()

    sess = tf.Session()
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
            mode_opt.minimize(sess)
            print('Initializing...')
            for init in init_types:
                for R in tqdm.tqdm(maxranks):
                    init_opt[init][R].minimize(sess)
            
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
