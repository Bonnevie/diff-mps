import pickle
import tensornets as tn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

filename = "tmpV7K2R16S0LrelaxMFalseUFalseCcanonIexpectationA0.01B100_grandseq.pkl"

    #"ratetestV6K2R16S0LrelaxMFalseUFalseCcanonIexpectationA0.01-0.1-1.0-10B100_grandseq.pkl"
#filename = "tmpV5K2R16S0LrelaxMFalseUFalseCcanonIexpectationA1000.0B100_grandseq.pkl"
#filename = "/home/rabo/Cloud/PhD/Projects/MPS/code/fig15_multi_restartV1K2R4S0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19LrelaxMFalseUFalseCcanonIexpectationA0.1B100_grandseq.pkl"
#"tmpV1K2R16S0LrelaxMFalseUFalseCcanonIexpectationA1.0B100_grandseq.pkl"
#"tmpV1K2R16S0LrelaxMFalseUFalseCcanonIexpectationA1.0B100_grandseq.pkl"
    #"fig15_multi_restartV1K2R4S0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19LrelaxMFalseUFalseCcanonIexpectationA0.1B100_grandseq.pkl"
#"./tmpV1K2R16S0LrelaxMFalseUFalseCcanonIexpectationA0.1B100_grandseq.pkl"
#"./fig15_ambigraphV5K2R1-4-8S0-1LrelaxMFalseUFalseCcanonIexpectationA0.1B100_grandseq.pkl"

#'./fig15_ambigraphV3K2R1-4S0-1-2LrelaxMFalseUFalseCcanonIexpectationA0.1B100_grandseq.pkl'
#filename = './fig15_strong_priorV2K2R1-4-16S0-1-2LrelaxMFalseUFalseCcanonIexpectationA0.01B100_grandseq.pkl'

with open(filename, 'rb') as file:
    data = pickle.load(file)

q = {}
colloc = {}
loss = {}

losses = data['df_c']['loss']
gradnorms = data['df_c']['gradnorm']
ranks = sorted(losses.index.to_frame()['rank'].unique())
ranks_observed = []

plt.figure(1)
nq = len(ranks)
for qkey, qmeta in data['q'].items():
    with tf.Session() as sess:
        rank = qkey[0]
        restart = qkey[1]
        if rank not in ranks_observed:
            plt.subplot(2,nq,np.where([rank == a_rank for a_rank in ranks])[0][0] + 1)
            plt.plot(losses[qkey])
            plt.title('r={},i={}'.format(rank,restart))
            plt.subplot(2, nq, nq + np.where([rank == a_rank for a_rank in ranks])[0][0] + 1)
            plt.plot(gradnorms[qkey])
            plt.title('r={},i={}'.format(rank,restart))

        q = tn.unpackmps(qmeta, sess)[0]
        colloc[(rank, restart)] = sess.run(q.collocation(10000))
        loss[(rank, restart)] = losses[qkey].iloc[-1]
##
plt.figure(2)
def mingrid(n):
     grid = [np.floor(np.sqrt(n)), np.floor(np.sqrt(n))]
     next_idx = 0
     while grid[0] * grid[1] < n:
         grid[next_idx] += 1
         next_idx = (next_idx + 1) % 2
     return (int(grid[0]), int(grid[1]))

for index, ((rank, restart), collmat) in enumerate(colloc.items()):
    ax = plt.subplot(*mingrid(len(colloc)), index + 1)
    ax.matshow(collmat, vmin=0, vmax=1, cmap='RdBu')
    ax.axis('off')
    plt.title('r={},i={},l={:.2}'.format(rank,restart, loss[(rank, restart)]))