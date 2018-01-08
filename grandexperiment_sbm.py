import numpy as np
import tensorflow as tf
from relaxflow.relax import RELAX, CategoricalReparam#, categorical_forward, categorical_backward
import time
dtype = 'float64'

import tqdm
import matplotlib.pyplot as plt

from collapsedclustering import CollapsedStochasticBlock
from tensornets import MPS, Canonical, symmetrynorm
from edward.models import MultivariateNormalTriL, Dirichlet, WishartCholesky, ParamMixture

N = 5
K = 2
R = 3
nsamples=5
steps = 10000

datademo = False
timeit = False
dotrain = True
dynamic = True
conditional = False
grad_summaries= True
folder = 'new_dynamic2'
#cap core rank at R
ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))

np.random.seed(1)

#5-22
#10-54


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

X = Akarate[:N,:N]

tf.reset_default_graph()



with tf.name_scope("model_setup"):
    print("Defining model...")
    cores = Canonical(N, K, ranks)
    rcores = Canonical(N, K, ranks)
    ccores = Canonical(N, K, ranks)
    equalize = tf.group(rcores.copycore_op(cores), ccores.copycore_op(cores))
    cnu = tf.Variable(1., name="weight", dtype=dtype)
    nu = tf.Variable(1., name="weight", dtype=dtype)
    q = MPS(N, K, ranks, cores=cores)
    qr = MPS(N, K, ranks, cores=rcores)
    qc = MPS(N, K, ranks, cores=ccores)

    p = CollapsedStochasticBlock(N, K)

    elbof = lambda sample: -(p.logp(sample, tf.constant(X)) -
                             tf.log(q.contraction(sample)))

    relbof = lambda sample: -(p.logp(sample, tf.constant(X)) -
                             tf.log(qr.contraction(sample)))
    if conditional:
        celbof = lambda sample: -(p.logp(sample, tf.constant(X)) -
                                 tf.log(qc.contraction(sample)))


        def condloss(sample):
            P = p.logp_conditionals(sample, X)
            Q = qc.gibbsconditionals(sample)
            return tf.reduce_sum(tf.exp(Q)*(Q-P))/N

#sample = tf.stop_gradient(q.sample())
#elbo_s = p.logp(sample, tf.constant(X)) - tf.log(q.contraction(sample))

with tf.name_scope("objective_setup"):
    print("Building objectives and estimators...")
#rsample = tf.stop_gradient(qr.sample())
#relbo = elbof(rsample)
    with tf.name_scope("soft"):
        print("\tBuilding soft estimator...")
        t = time.time()
        softelbo = 0.
        for n in range(nsamples):
            softelbo += elbof(q.softsample())
        softelbo = softelbo/nsamples
        tf.summary.scalar('ELBO', -softelbo)
        soft_grad = tf.gradients(softelbo, cores.params())
        print(" (completed in {}s)".format(time.time()-t))

    #RELAX
    with tf.name_scope("RELAX"):
        print("\tBuilding RELAX parameters...")
        t = time.time()
        relax_params = qr.buildcontrol(relbof, nsamples=nsamples)
        #f, c, cb, p = relax_params
        #relax_params = (0., c, cb, p)
        print(" (completed in {}s)".format(time.time()-t))
        print("\tBuilding RELAX estimator...")
        t = time.time()
        if dynamic:
            relax_grad, relax_var_grad = RELAX(*relax_params, rcores.params(),
                                           var_params=[qr._tempvar, nu], weight=nu, summaries=grad_summaries)
            tf.summary.scalar('second_moment', tf.reduce_mean(tf.concat([tf.reshape(tf.square(g), (-1,)) for g, p in relax_grad], axis=0)))
            tf.summary.scalar('nu', nu)
            tf.summary.scalar('temp', tf.nn.softplus(qr._tempvar))
        else:
            relax_grad, relax_var_grad = RELAX(*relax_params, rcores.params(), weight=nu, summaries=grad_summaries)
        #relax_elbo = relax_params[0]
        rsoftelbo = 0.
        for n in range(nsamples):
            rsoftelbo += relbof(qr.softsample())
        rsoftelbo = rsoftelbo/nsamples
        print(" (completed in {}s)".format(time.time()-t))
        tf.summary.scalar('ELBO', -rsoftelbo)
    if conditional:
        # CONDITIONAL
        with tf.name_scope("conditional"):
            print("\tBuilding conditional RELAX parameters...")
            t = time.time()
            cond_relax_params = qc.buildcontrol(condloss, nsamples=nsamples)
            #f, c, cb, p = relax_params
            #relax_params = (0., c, cb, p)
            print(" (completed in {}s)".format(time.time()-t))

            print("\tBuilding conditional loss estimator")
            t = time.time()
            if dynamic:
                cond_grad, cond_var_grad = RELAX(*cond_relax_params, ccores.params(),
                                               var_params=[qc._tempvar, cnu], weight=cnu, summaries=grad_summaries)
                tf.summary.scalar('second_moment', tf.reduce_mean(tf.concat([tf.reshape(tf.square(g), (-1,)) for g, p in cond_grad], axis=0)))
                tf.summary.scalar('nu', cnu)
                tf.summary.scalar('temp', tf.nn.softplus(qc._tempvar))
            else:
                cond_grad, cond_var_grad = RELAX(*cond_relax_params, ccores.params(), weight=cnu, summaries=grad_summaries)
            csoftelbo = 0.
            closs = 0.
            for n in range(nsamples):
                csoftelbo += celbof(qc.softsample())
                closs += condloss(qc.softsample())
            csoftelbo = csoftelbo/nsamples
            tf.summary.scalar('ELBO', -csoftelbo)
            tf.summary.scalar('condloss', -closs)

            print(" (completed in {}s)".format(time.time()-t))

#synorm_op = symmetrynorm(cores.cores)
#entropy_op = q.marginalentropy()
with tf.name_scope("opt_step"):
    #step = train.minimize(relbo)
    print("Constructing optimizers...")
    soft_train = tf.train.AdamOptimizer()
    soft_step = soft_train.minimize(softelbo, var_list=[cores.params()])
    relax_train = tf.train.AdamOptimizer()
    relax_step = relax_train.apply_gradients(relax_grad)
    if conditional:
        cond_train = tf.train.AdamOptimizer()
        cond_step = cond_train.apply_gradients(cond_grad)
    else:
        cond_step = tf.no_op()
    if dynamic:
        var_train = tf.train.AdamOptimizer()
        var_step = var_train.apply_gradients(relax_var_grad)
        if conditional:
            cond_var_train = tf.train.AdamOptimizer()
            cond_var_step = cond_var_train.apply_gradients(cond_var_grad)
        else:
            cond_var_step = tf.no_op()
    else:
        var_step = tf.no_op()
        cond_var_step = tf.no_op()
    all_step = tf.group(soft_step, relax_step, var_step, cond_step, cond_var_step)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('./train/' + folder, sess.graph)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
#tf.get_default_graph().finalize()
sess.run(init)
sess.run(equalize)
print("Commencing optimization.")
if dotrain:

    run_options = tf.RunMetadata()
    for it in tqdm.tqdm(range(steps), total=steps):
        if timeit:
            summary, _ = sess.run([merged, all_step],
                               options=run_options,
                               run_metadata=run_metadata)
        else:
            summary, _ = sess.run([merged, all_step])
        with tf.name_scope("trace"):
            train_writer.add_summary(summary, it)
    if timeit:
        train_writer.add_run_metadata(run_metadata, "runtrace")
