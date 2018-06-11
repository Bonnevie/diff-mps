import numpy as np
import tensorflow as tf
import pickle
from relaxflow.reparam import CategoricalReparam
import time
dtype = 'float64'

import tqdm
import pandas as pd

from collapsedclustering import CollapsedStochasticBlock, KLcorrectedBound
import tensornets as tn

from itertools import product

#FLAGS
name = 'full-wr1-wbase-corrected' 
version = 1
Ntest = 0 #number of edges to use for testing
K = 2 #number of communities to look for
folder = name + 'V{}K{}'.format(version, K)

#factors variations to run experiments over
copies = 10
random_restarts = 10
coretypes = ['canon']#,'perm'] #types of cores to try 
#Options are: '' for ordinary cores, canon' for canonical, and 'perm' for permutation-free
maxranks = [1,2,4,8,12,16]#,12,15,18]
Ns = [4,6,8]#,6,7] #number of vertices in graph

factor_code = ['N','T','R','S']
factor_names = ['size','coretype','rank','restarts']
factors = [Ns, coretypes, maxranks, range(random_restarts)]
short_key = False
active_factors = [len(factor)>1 for factor in factors]
all_config = list(product(*factors))
config_count = np.prod([len(factor) for factor in factors])
config_full_name = ''.join([code + '-'.join([str(fact) for fact in factor]) for code, factor in zip(factor_code, factors)])
copy_writer = []
        
np.random.seed(1)
tf.reset_default_graph()



sym = lambda X: np.triu(X, 1) + np.triu(X, 1).T
X = {}
mask = {}
predictionmask = {}
p = {}
for N in Ns:
    p[N] = CollapsedStochasticBlock(N, K, alpha=1, a=1, b=1)
    X[N] = np.stack([sym(np.random.randn(N, N)>0.5).astype(dtype) for _ in range(copies)])
    #generate mask of observed edges
    if Ntest > 0:
        mask[N] = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
        mask[N] = mask[N] < np.sort(mask[N].ravel())[Ntest]
        mask[N] = np.logical_not(np.logical_or(mask[N],mask[N].T))
        mask[N] = np.triu(mask[N], 1)
        predictionmask[N] = np.triu(~mask[N], 1)
        mask[N] = tf.convert_to_tensor(mask[N].astype(dtype))
        predictionmask[N] = tf.convert_to_tensor(predictionmask[N].astype(dtype))
    else:
        mask[N] = np.ones((N, N), dtype=bool)
        mask[N] = np.triu(mask[N], 1)
        mask[N] = tf.convert_to_tensor(mask[N].astype(dtype))

trueelbo = {}
pred = {}
q = {}
Xt = {}
qtensor = {}
opt = {}
cores = {}

all_anchors = {}
for N in Ns:
    all_anchors[N] = tf.stack([np.eye(K)[list(index)] for index in np.ndindex((K,)*N)])

with tf.name_scope("model"):
    
    print("building models...")
    for config in tqdm.tqdm(all_config, total=config_count):
        N, coretype, R, restart_ind = config
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
            if Ntest>0:
                pred[config] = tf.reduce_sum(q[config].batch_contraction(all_anchors[N])*(p[N].batch_logp(all_anchors[N],Xt[config],observed=predictionmask[N])))
            trueelbo[config] = tf.reduce_sum(q[config].batch_contraction(all_anchors[N])*(p[N].batch_logp(all_anchors[N],Xt[config],observed=mask[N])-tf.log(q[config].batch_contraction(all_anchors[N]))))
            qtensor[config] = q[config].populatetensor()
            #build optimizers
            opt[config] = tf.contrib.opt.ScipyOptimizerInterface(-trueelbo[config], var_list=cores[config].params())

    column_names = ['ELBO','KL']
    if Ntest>0:
        column_names += ['pred_llk']
    index_p = pd.MultiIndex.from_product([maxranks, Ns, range(copies)], names=['rank', 'size', 'copy'])
    df_p = pd.DataFrame(np.zeros((len(maxranks)*len(Ns)*copies,len(column_names))), index=index_p, columns=column_names)
    logptensor = {N:{} for N in Ns}
    logZ = {N:{} for N in Ns}
    ptensor = {N:{} for N in Ns}
    KLmf = {N:{} for N in Ns}

    #baseline evaluation
    sess = tf.Session()
    print("Calculating baselines.")
    with tf.name_scope("truemodel"):
        for N, copy in product(Ns, range(copies)):
            with sess.as_default():
                logptensor[N][copy] = p[N].populatetensor(X[N][copy], observed=mask[N])
                logZ[N][copy] = np.logaddexp.reduce(logptensor[N][copy].ravel())
                ptensor[N][copy] = np.exp(logptensor[N][copy] - logZ[N][copy])
                Z = tf.Variable(tf.random_normal((500,N, K), dtype='float64'), dtype='float64')
                bound = KLcorrectedBound(p[N], X[N][copy], [Z], batch=True, observed=mask[N])
                sess.run(tf.variables_initializer([Z]))
                bound.minimize()
                Zmf = sess.run(tf.nn.softmax(Z))
                ptensor_mf = [reduce(np.multiply.outer, vs.T) for vs in Zmf]
                KLmf[N][copy] = [np.sum(q*(np.log(q)-np.log(ptensor[N][copy]))) for q in ptensor_mf] 

                for rank in tqdm.tqdm(maxranks, total=len(maxranks)):
                    pcore, pmps = tn.full2TT(np.sqrt(ptensor[N][copy]), rank, normalized=True)
                    if Ntest>0:
                        pred_p = sess.run(tf.reduce_sum(pmps.batch_contraction(all_anchors[N])*(p[N].batch_logp(all_anchors[N],X[N][copy],observed=predictionmask[N]))))
                    elbo_p = sess.run(tf.reduce_sum(pmps.batch_contraction(all_anchors[N])*(p[N].batch_logp(all_anchors[N],X[N][copy],observed=mask[N])-tf.log(pmps.batch_contraction(all_anchors[N])))))
                    ptensor_p = sess.run(pmps.populatetensor())
                    kl_p = np.sum(ptensor_p*(np.log(ptensor_p)-np.log(ptensor[N][copy])))
                    df_p['ELBO'][(rank, N, copy)] = elbo_p
                    if Ntest>0:
                        df_p['pred_llk'][(rank, N, copy)] = pred_p
                    df_p['KL'][(rank, N, copy)] = kl_p
    init = tf.global_variables_initializer()
    
    #run all configurations
    index_c = pd.MultiIndex.from_product(factors + [range(copies)], names=factor_names + ['copy'])
    df_c = pd.DataFrame(np.zeros((config_count*copies,3)), index=index_c, columns=column_names + ['time'])
    with tf.name_scope("optimization"):    
        print("Starting optimization.")
        for copy in tqdm.tqdm(range(copies), total=copies):
            sess.run(init)
            for config in tqdm.tqdm(all_config, total=config_count):
                N = config[0]
                configc = config + (copy,)
                kl_c = np.inf
                sess.run(init)
                t0 = time.time()
                opt[config].minimize(sess, feed_dict = {Xt[config]: X[N][copy]})
                df_c['time'][configc] = time.time() - t0
                if Ntest>0:
                    elbo_c, pred_c, qtensor_c = sess.run([trueelbo[config], pred[config], qtensor[config]], feed_dict = {Xt[config]: X[N][copy]})
                    df_c['pred_llk'][configc] = pred_c
                else:
                    elbo_c, qtensor_c = sess.run([trueelbo[config], qtensor[config]], feed_dict = {Xt[config]: X[N][copy]})
                df_c['ELBO'][configc] = elbo_c
                kl_c = np.sum(qtensor_c*(np.log(qtensor_c)-np.log(ptensor[N][copy])))
                df_c['KL'][configc] = kl_c

save_name = folder + config_full_name + '_optrank.pkl'
supdict = {'name': save_name, 'df_c':df_c, 'df_p':df_p, 'logZ': logZ, 'X':X, 'KLmf': KLmf}
with open(folder + config_full_name + '_optrank.pkl','wb') as handle:
    pickle.dump(supdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

