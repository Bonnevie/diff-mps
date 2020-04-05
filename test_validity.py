import numpy as np
import tensorflow as tf
sess = tf.Session()
dtype = 'float64'

import tqdm
from relaxflow.relax import RELAX
from collapsedclustering import CollapsedStochasticBlock
import tensornets as tn

from itertools import product

from AMSGrad.optimizers import AMSGrad as amsgrad

decay = 0.5
decay_steps = 100
marginal = False

base_learningrate = 1e-2  # [1e-1,1e-2,1e-3]
nsample = 1000

N = 3
K = 2
R = 2

X = tf.constant(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype='float64'))
mask = tf.constant(np.triu(np.ones((3, 3), dtype='float64'), 1))

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

for _ in tqdm.trange(5000):
    sess.run(step)
    losses.append(sess.run(loss))
##
estimated_prob = {}
joint_log_prob = {}
for labeling in product([0, 1], repeat=3):
    Z = tf.expand_dims(tf.constant(np.eye(2, dtype='float64')[list(labeling)]), 0)
    joint_log_prob[labeling] = p.batch_logp(Z=Z[None], X=X, observed=mask).eval()
    estimated_prob[labeling] = q.batch_logp(Z=Z)[0].eval()


joint_log_partition = np.logaddexp.reduce(list(joint_log_prob.values()))
estimated_log_partition = np.logaddexp.reduce(list(estimated_prob.values()))

posterior_prob = {key: np.exp(val - joint_log_partition) for key, val in joint_log_prob.items()}
estimated_prob = {key: np.exp(val - estimated_log_partition) for key, val in estimated_prob.items()}


##
ptensor = np.zeros((2,2,2))
true_nsamples = 100000
for key, val in posterior_prob.items():
    ptensor[key] = val
true_mps = tn.full2TT(np.sqrt(ptensor), normalized=True)[1]
true_mps_elbo_tf = lambda sample: true_mps.elbo(sample, logp, marginal=False, report=True)
from collections import defaultdict
true_mps_samples = sess.run(true_mps.sample(true_nsamples))
true_mps_freq = defaultdict(int)
for sample in true_mps_samples:
    true_mps_freq[tuple(sample[:,1].tolist())] += 1

true_mps_log_prob = {}
true_mps_log_prob_util = {}
estimated_prob = {}
joint_log_prob = {}
for labeling in product([0, 1], repeat=3):
    Z = tf.expand_dims(tf.constant(np.eye(2, dtype='float64')[list(labeling)]), 0)
    true_mps_log_prob[labeling] = tf.log(true_mps.batch_contraction(Z=Z)[0]).eval()
    true_mps_log_prob_util[labeling] = (true_mps.batch_logp(Z=Z)[0]).eval()
    joint_log_prob[labeling] = p.batch_logp(Z=Z[None], X=X, observed=mask).eval()
    estimated_prob[labeling] = q.batch_logp(Z=Z)[0].eval()

true_mps_freq = {key: val/true_nsamples for key, val in true_mps_freq.items()}

true_mps_log_partition = np.logaddexp.reduce(list(true_mps_log_prob.values()))
true_mps_prob = {key: np.exp(val - true_mps_log_partition) for key, val in true_mps_log_prob.items()}

true_mps_log_partition = np.logaddexp.reduce(list(true_mps_log_prob_util.values()))
true_mps_prob_util = {key: np.exp(val - true_mps_log_partition) for key, val in true_mps_log_prob_util.items()}



true_mps_entropy = -np.sum([prob * np.log(prob) for prob in true_mps_prob.values()])
true_mps_cross_divergence = np.sum([true_mps_prob[key] * joint_log_prob[key] for key in estimated_prob.keys()])

true_mps_true_elbo = true_mps_cross_divergence + true_mps_entropy
true_mps_estimated_elbo, true_mps_estimated_llk, true_mps_estimated_entropy = sess.run([tf.reduce_mean(x) for x in true_mps_elbo_tf(true_mps_samples)])

##
elbo = sess.run(true_mps.elbo(true_mps.sample(1e5), logp, marginal=False, report=False))

##
model_entropy = -np.sum([prob * np.log(prob) for prob in estimated_prob.values()])
cross_divergence = np.sum([estimated_prob[key] * joint_log_prob[key] for key in estimated_prob.keys()])
true_elbo = cross_divergence + model_entropy

