import numpy as np
import tensorflow as tf
import pickle
from relaxflow.reparam import CategoricalReparam
import time
dtype = 'float64'

import tqdm
import pandas as pd

import matplotlib.pyplot as plt

from collapsedclustering import CollapsedStochasticBlock, KLcorrectedBound
import tensornets as tn

from itertools import product

#FLAGS
timeit = False #log compute time of every op for Tensorboard visualization (costly)
calculate_true = False #calculate true tensor and MPS (WARNING: can cause OOM for N>>14)
do_anneal = False #do entropy annealing
name = 'lowrank' 
version = 3
N = 7 #number of vertices in graph
Ntest = 5 #number of edges to use for testing
K = 3 #number of communities to look for
nsamples=100 #number of gradient samples per iteration

folder = name + 'V{}N{}K{}S{}'.format(version, N, K, nsamples)

#factors variations to run experiments over
copies = 50
coretypes = [''] #types of cores to try 
#Options are: '' for ordinary cores, canon' for canonical, and 'perm' for permutation-free
maxranks = [3,6,9,12,15,18,21,24,27]
#Options are: 'SGD', 'Adadelta', 'Adam'
objective = ['']

factor_code = ['T','R']
factor_names = ['coretype','rank']
factors = [coretypes, maxranks]
short_key = False
active_factors = [len(factor)>1 for factor in factors]
all_config = list(product(*factors))
config_count = np.prod([len(factor) for factor in factors])
config_full_name = ''.join([code + '-'.join([str(fact) for fact in factor]) for code, factor in zip(factor_code, factors)])
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
predictionmask = np.triu(~mask, 1)
mask = tf.convert_to_tensor(mask.astype(dtype))
predictionmask = tf.convert_to_tensor(predictionmask.astype(dtype))

sym = lambda X: np.triu(X, 1) + np.triu(X, 1).T
X = np.stack([sym(np.random.randn(N, N)>0.5).astype(dtype) for _ in range(copies)])


cores = {}
q = {}
softelbo = {}
true_elbo = {}
softpred = {}
softobj = {}
trueelbo = {}
pred = {}
opt = {}
step = {}
equalize_ops = []
qtensor = {}
Xt = {}

copy_summary = [[] for copy in range(copies)]
all_anchors = tf.stack([np.eye(K)[list(index)] for index in np.ndindex((K,)*N)])

global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
increment_global_step_op = tf.assign(global_step, global_step+1)
with tf.name_scope("model"):
    p = CollapsedStochasticBlock(N, K, alpha=1, a=1, b=1)
    #p = CollapsedStochasticBlock(N, K, alpha=100, a=10, b=10)
    
    print("building models...")
    for config in tqdm.tqdm(all_config, total=config_count):
        coretype, R = config
        if short_key:
            config_name = ''.join([''.join([key, str(value)]) for key, value, active in zip(factor_code, config, active_factors) if active])
        else:
            config_name = ''.join([''.join([key, str(value)]) for key, value in zip(factor_code, config)])
        pred[config] = []
        trueelbo[config] = []
                
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
            
            #build q model
            q[config] = tn.MPS(N, K, ranks, cores=cores[config])
            pred[config] = tf.reduce_sum(q[config].batch_contraction(all_anchors)*(p.batch_logp(all_anchors,Xt[config],observed=predictionmask)))
            trueelbo[config] = tf.reduce_sum(q[config].batch_contraction(all_anchors)*(p.batch_logp(all_anchors,Xt[config],observed=mask)-tf.log(1e-16+q[config].batch_contraction(all_anchors))))
            qtensor[config] = q[config].populatetensor()
            #build optimizers
            opt[config] = tf.contrib.opt.ScipyOptimizerInterface(-trueelbo[config], var_list=cores[config].params(), options={'gtol':1e-6})

    column_names = ['ELBO','pred_llk','KL']
    index_p = pd.MultiIndex.from_product([maxranks, range(copies)], names=['rank', 'copy'])
    df_p = pd.DataFrame(np.zeros((len(maxranks)*copies,3)), index=index_p, columns=column_names)
    logptensor = {}
    logZ = {}
    ptensor = {}


    sess = tf.Session()
    with tf.name_scope("truemodel"):
        for copy in tqdm.tqdm(range(copies), total=copies):
            with sess.as_default():
                logptensor[copy] = p.populatetensor(X[copy], observed=mask)
                logZ[copy] = np.logaddexp.reduce(logptensor[copy].ravel())
                ptensor[copy] = np.exp(logptensor[copy] - logZ[copy])
                for rank in tqdm.tqdm(maxranks, total=len(maxranks)):
                    pcore, pmps = tn.full2TT(np.sqrt(ptensor[copy]), rank)
                    pred_p = sess.run(tf.reduce_sum(pmps.batch_contraction(all_anchors)*(p.batch_logp(all_anchors,X[copy],observed=predictionmask))))
                    elbo_p = sess.run(tf.reduce_sum(pmps.batch_contraction(all_anchors)*(p.batch_logp(all_anchors,X[copy],observed=mask)-tf.log(1e-16+pmps.batch_contraction(all_anchors)))))
                    ptensor_p = sess.run(pmps.populatetensor())
                    kl_p = np.sum(ptensor_p*(np.log(ptensor[copy])-np.log(ptensor_p)))
                    df_p['ELBO'][(rank, copy)] = elbo_p
                    df_p['pred_llk'][(rank, copy)] = pred_p
                    df_p['KL'][(rank, copy)] = kl_p
    init = tf.global_variables_initializer()
    
    index_c = pd.MultiIndex.from_product(factors + [range(copies)], names=factor_names + ['copy'])
    df_c = pd.DataFrame(np.zeros((config_count*copies,3)), index=index_c, columns=column_names)

    with tf.name_scope("optimization"):    
        print("Starting optimization.")
        for copy in tqdm.tqdm(range(copies), total=copies):
            sess.run(init)
            for config in tqdm.tqdm(all_config, total=config_count):
                configc = config + (copy,)
                opt[config].minimize(sess, feed_dict = {Xt[config]: X[copy]})
                elbo_c, pred_c, qtensor_c = sess.run([trueelbo[config], pred[config], qtensor[config]], feed_dict = {Xt[config]: X[copy]})
                kl_c = np.sum(qtensor_c*(np.log(ptensor[copy])-np.log(qtensor_c)))
                df_c['ELBO'][configc] = elbo_c
                df_c['pred_llk'][configc] = pred_c
                df_c['KL'][configc] = kl_c

supdict = {'df_c':df_c, 'df_p':df_p, 'logZ': logZ, }
with open(folder + config_full_name + '_optrank.pkl','wb') as handle:
    pickle.dump(supdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
