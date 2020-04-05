import numpy as np
import tensorflow as tf
sess = tf.Session()
dtype = 'float64'

import tqdm
from relaxflow.relax import RELAX
from collapsedclustering import CollapsedStochasticBlock
import tensornets as tn

from AMSGrad.optimizers import AMSGrad as amsgrad

from networkx import karate_club_graph, adjacency_matrix
import matplotlib.pyplot as plt

nseeds = 10
nsteps = 2000
ncollocsamples = 10000
nsample = 100

decay = 0.5
decay_steps = 100
marginal = False

base_learningrate = 1e-2  # [1e-1,1e-2,1e-3]



N = 34
Ntest = 161  # number of edges to use for testing
K = 2  # number of communities to look for
R = 2

karate = karate_club_graph()
X = adjacency_matrix(karate).toarray().astype('float64')
X = X[:N, :N]

collocs = {}
for seed in range(10):
    np.random.seed(seed)
    tf.reset_default_graph()

    #generate mask of observed edges
    mask = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
    mask = mask < np.sort(mask.ravel())[Ntest]
    mask = np.logical_not(np.logical_or(mask,mask.T))
    mask = np.triu(mask, 1)
    predictionmask = np.triu(~mask, 1)

    mask = mask.astype(dtype)
    predictionmask = predictionmask.astype(dtype)

    ranks = tuple(min(K ** min(r, N - r), R) for r in range(N + 1))
    cores = tn.Canonical(N, K, ranks, orthogonalstyle=tn.OrthogonalMatrix)
    q = tn.MPS(N, K, ranks, cores=cores)

    tfrate = tf.convert_to_tensor(base_learningrate, dtype=dtype)
    decay_stage = tf.Variable(1, name='decay_stage', trainable=False, dtype=tf.int32)
    learningrate = tf.train.exponential_decay(tfrate, decay_stage, decay_steps, decay)

    # AMS optimizer
    beta1 = tf.Variable(0.9, dtype='float64', trainable=False)
    beta2 = tf.Variable(0.999, dtype='float64', trainable=False)
    epsilon = tf.Variable(0.999, dtype='float64', trainable=False)

    stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    var_stepper = amsgrad(learning_rate=learningrate, beta1=beta1, beta2=beta2, epsilon=epsilon)

    # Objective
    p = CollapsedStochasticBlock(N, K, alpha=1, a=1, b=1)
    logp = lambda sample: p.batch_logp(sample, X, observed=mask)

    control_samples = q.shadowrelax(nsample)
    elbo = lambda sample: -q.elbo(sample, logp, marginal=marginal)
    loss = tf.reduce_mean(elbo(control_samples[0]))

    relax_params = tn.buildcontrol(control_samples, q.batch_logp, elbo)
    grad, _ = RELAX(*relax_params, hard_params=q.params(), var_params=[], weight=q.nu)
    step = stepper.apply_gradients(grad)

    sess = tf.InteractiveSession()
    losses = []

    sess.run(tf.global_variables_initializer())

    for _ in tqdm.trange(nsteps):
        sess.run(step)
        losses.append(sess.run(loss))

    collocs[seed] = sess.run(q.collocation(ncollocsamples))

plt.figure()
for key, colloc in collocs.items():
    ax = plt.subplot(np.ceil(np.sqrt(nseeds)), np.ceil(np.sqrt(nseeds)), key + 1)
    ax.matshow(colloc, vmin=0, vmax=1)