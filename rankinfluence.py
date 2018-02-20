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

name = 'report'
datatype = 'random'
coretype = 'Tcanon' #'Tperm', 'Tcanon', ''
N = 10
K = 3
maxranks = [1, 3, 9, 27]#[1,6,9,12,15,18,21,24]
nsamples=100
steps = 10000
runs = 10
version = 1
rate = 0.001

folder = name + '{}D{}V{}N{}K{}S{}R{}'.format(coretype,datatype,version, N, K, nsamples, '-'.join(str(rank) for rank in maxranks))
#cap core rank at R

timeit = False
calculate_true = True
histograms = False

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
tf.reset_default_graph()

cores = {}
q = {}
softelbo = {}
opt = {}
step = {}
equalize_ops = []
init_opt = {}
with tf.name_scope("model"):
    p = CollapsedStochasticBlock(N, K, alpha=100, a=10, b=10)
    Z = tf.Variable(tf.random_normal((10, N, K), dtype=dtype))

    bounds = -tf.reduce_mean([KLcorrectedBound(p, X, [z]).bound for z in tf.unstack(Z, axis=0)])
    mode_opt = tf.contrib.opt.ScipyOptimizerInterface(bounds, var_list=[Z])
    
    Xt = tf.constant(X)
    print("building models...")
    for R in tqdm.tqdm(maxranks):
        with tf.name_scope("rank"+str(R)):

            if coretype is 'Tcanon':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))#(1,) + tuple() + (1,)
                cores[R] = tn.Canonical(N, K, ranks, orthogonalstyle=tn.OrthogonalMatrix)
            elif coretype is 'Tperm':
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                repranks = (1,)+tuple(min((2)**min(r, N-r-2)*K, R) for r in range(N-1))+(1,)
                cores[R] = tn.PermutationCore_augmented(N, K, repranks, ranks)
            else:
                ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))
                cores[R] = tn.Core(N, K, ranks)
            q[R] = tn.MPS(N, K, ranks, cores=cores[R])
            elbo, loss, entropy, marginalentropy, marginalcv = (q[R].elbo(lambda sample: p.batch_logp(sample, Xt), nsamples=nsamples, fold=False, report=True))
            softelbo[R] = tf.reduce_mean(elbo)
            
            mode_loss = -(q[R].marginalentropy())
            init_opt[R] = tf.contrib.opt.ScipyOptimizerInterface(mode_loss, var_list=cores[R].params())
                                
            #softelbo[R] = tf.reduce_mean(q[R].elbo(lambda sample: p.batch_logp(sample, Xt), nsamples=nsamples, fold=False))
            opt[R] = tf.train.AdamOptimizer(learning_rate=rate)
            step[R] = opt[R].minimize(-softelbo[R], var_list=cores[R].params())
            tf.summary.scalar('ELBO', tf.reduce_mean(elbo))
            tf.summary.scalar('logp', tf.reduce_mean(loss))
            tf.summary.scalar('entropy', tf.reduce_mean(entropy))
            tf.summary.scalar('marginalentropy', tf.reduce_mean(marginalentropy))
            tf.summary.scalar('cv', tf.reduce_mean(marginalcv))
            if histograms:
                tf.summary.histogram('ELBO', elbo)
                tf.summary.histogram('logp', loss)
                tf.summary.histogram('entropy', entropy)
                tf.summary.histogram('cventropy', entropy+marginalcv)
                tf.summary.histogram('marginalentropy', marginalentropy)
                tf.summary.histogram('cv', marginalcv)

    if timeit:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    train = tf.group(*step.values())
    summaries = tf.summary.merge_all()

    sess = tf.Session()
    with tf.name_scope("truemodel"):
        if calculate_true:
            with sess.as_default():
                ptensor = p.populatetensor(X)
                pcore, pmps = tn.full2TT(np.sqrt(ptensor))
                true_loss_op = tf.reduce_mean(pmps.elbo(lambda sample: p.batch_logp(sample, Xt), nsamples=nsamples, fold=False))

    init = tf.global_variables_initializer()
    sess.run(init)

    if calculate_true:
        true_loss = sess.run(true_loss_op)
    #tf.get_default_graph().finalize()
    with tf.name_scope("optimization"):    
        Q = {}
        print("Starting optimization.")
        for run in tqdm.tqdm(range(runs), desc="Run", total=runs):
            writer = tf.summary.FileWriter('./train/' + folder + 'run' + str(run), sess.graph)
            sess.run(init)
            #mode_opt.minimize(sess)
            #print('Initializing...')
            for R in tqdm.tqdm(maxranks):
                init_opt[R].minimize(sess)
            
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
