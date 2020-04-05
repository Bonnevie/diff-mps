import numpy as np
import tqdm
from scipy.special import betaln, gammaln
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from collapsedclustering import CollapsedStochasticBlock
import tensorflow as tf

def multibetaln(a):
    return gammaln(a).sum(axis=-1) - gammaln(a.sum(axis=-1))

class SBM:
    def __init__(self, X, mask, a=1, b=1, alpha=1, initial_Z=None):
        self.X = X
        self.mask = mask

        self.N = self.X.shape[0]
        self.K = 2

        self.alpha = alpha
        self.a = a
        self.b = b

        self.w = np.random.dirichlet(self.alpha * np.ones(2)) #
        self.beta = np.triu(np.random.beta(self.a, self.b, size=(2,2)))
        self.Z = np.random.multinomial(1, self.w, self.N) if initial_Z is None else initial_Z # N x 2

        self.tfsbm = CollapsedStochasticBlock(self.N, self.K, alpha, a, b)
        convert = lambda x: tf.constant(x.astype('float64'))
        self.tfZ = tf.placeholder('float64', (self.N, self.K), 'Z')
        self.tflogp_op = self.tfsbm.logp(self.tfZ, convert(self.X), convert(self.mask))

    def logp_mikkel(self):
        N = self.X.shape[0]
        masked_edges = self.X * self.mask
        Y = masked_edges + masked_edges.T
        W = 1. - (self.mask + self.mask.T) - np.eye(N)

        zz = self.Z
        m = np.sum(zz, axis=0, keepdims=True).T
        M1 = np.matmul(np.matmul(zz.T, Y), zz) - np.diag(np.sum(np.matmul(Y, zz) * zz, axis=0) / 2)
        W1 = np.matmul(np.matmul(zz.T, W), zz) - np.diag(np.sum(np.matmul(W, zz) * zz, axis=0) / 2)
        M0 = np.matmul(m, m.T) - np.diag(np.squeeze(m * (m + 1) / 2)) - M1 - W1
        logp =  np.sum(np.triu(betaln(M1 + self.a, M0 + self.b) - betaln(self.a, self.b))) + np.sum(gammaln(m + self.alpha) - gammaln(self.alpha))
        return logp #, (np.sum(gammaln(m + self.alpha) - gammaln(self.alpha)), np.sum(np.triu(betaln(M1 + self.a, M0 + self.b) - betaln(self.a, self.b)))), (m, M1, M0)

    def logp_original(self):
        Z = self.Z
        X = self.X
        observed = self.mask

        membership = np.sum(Z, axis=0, keepdims=True)
        edgecounts = np.einsum('mk,mn,nl', Z, observed*X, Z) # Z^T*X*Z
        notedgecounts = np.einsum('mk,mn,nl', Z, observed, Z) - edgecounts
        edgecounts = np.triu(edgecounts) + np.tril(edgecounts, -1).T
        notedgecounts = np.triu(notedgecounts) + np.tril(notedgecounts, -1).T

        lnprior = np.sum(multibetaln((self.alpha + membership)) -
                                multibetaln((self.alpha + np.zeros_like(membership))))
        lnlink = np.sum(np.triu(betaln(self.a + edgecounts, self.b + notedgecounts) -
                               betaln(self.a + np.zeros_like(edgecounts), self.b + np.zeros_like(notedgecounts))))
        return lnprior + lnlink #, (lnprior, lnlink), (membership, edgecounts, notedgecounts)

    def compare_tf(self):
        Z = self.Z
        X = self.X
        observed = self.mask

        membership = np.sum(Z, axis=0, keepdims=True)
        edgecounts = np.einsum('mk,mn,nl', Z, observed*X, Z) # Z^T*X*Z
        notedgecounts = np.einsum('mk,mn,nl', Z, observed, Z) - edgecounts
        edgecounts = np.triu(edgecounts) + np.tril(edgecounts, -1).T
        notedgecounts = np.triu(notedgecounts) + np.tril(notedgecounts, -1).T

        lnprior = np.sum(multibetaln((self.alpha + membership)) -
                                multibetaln((self.alpha + np.zeros_like(membership))))
        lnlink = np.sum(np.triu(betaln(self.a + edgecounts, self.b + notedgecounts) -
                               betaln(self.a + np.zeros_like(edgecounts), self.b + np.zeros_like(notedgecounts))))

        convert = lambda x: tf.constant(x.astype('float64'))
        Z = self.tfZ#convert(self.Z)
        X = convert(self.X)
        observed = convert(self.mask)
        tril = np.tril(np.ones((self.K, self.K)),-1)
        triu = np.triu(np.ones((self.K, self.K)))
        tf_membership = tf.reduce_sum(Z, axis=0, keepdims=True)
        tf_edgecounts = tf.einsum('mk,mn,nl', Z, observed*X, Z) #Z^T*X*Z
        tf_notedgecounts = tf.einsum('mk,mn,nl', Z, observed, Z) - tf_edgecounts
        tf_edgecounts = (triu * tf_edgecounts + tf.transpose(tril * tf_edgecounts))
        tf_notedgecounts = (triu * tf_notedgecounts + tf.transpose(tril * tf_notedgecounts))
        tf_lnprior = tf.reduce_sum(tf.lbeta((self.alpha + tf_membership)) -
                                tf.lbeta((self.alpha + tf.zeros_like(tf_membership))))
        tf_lnlink = tf.reduce_sum(triu * tf.lbeta(tf.stack([self.a + tf_edgecounts,
                                                  self.b + tf_notedgecounts],
                                                  axis=2)) -
                               triu * tf.lbeta(tf.stack([self.a + tf.zeros_like(tf_edgecounts),
                                                  self.b + tf.zeros_like(tf_notedgecounts)],
                                                  axis=2)))
        with tf.Session() as sess:
            print(sess.run(tf_lnprior, feed_dict={self.tfZ: self.Z}))
            print(sess.run(tf_lnlink, feed_dict={self.tfZ: self.Z}))
            print((sess.run(tf_lnprior+tf_lnlink, feed_dict={self.tfZ: self.Z}), sess.run(self.tflogp_op, feed_dict={self.tfZ: self.Z})))
    def symmetric_beta(self):
        beta = self.beta + self.beta.T
        beta.ravel()[::beta.shape[0] + 1] = np.diag(self.beta)
        return beta

    def membership(self):
        return self.Z.sum(axis=0)

    def connections(self):
        edge_counts = sbm.Z.T.dot(self.mask * self.X).dot(self.Z)
        return np.triu(edge_counts) + np.tril(edge_counts, -1).T

    def connections_for_unknown_n(self, n):
        edge_counts = sbm.Z.T.dot(self.mask * self.X).dot(self.Z)
        edge_difference = np.einsum('tk,l->tkl', np.eye(self.K) - self.Z[n, None], self.neighbours_in_group(n))
        edge_counts = edge_counts[None] + edge_difference
        return np.triu(edge_counts) + np.transpose(np.tril(edge_counts, -1),[0,2,1])

    def non_connections(self):
        no_edge_counts = sbm.Z.T.dot(self.mask * (1. - self.X)).dot(self.Z)
        return np.triu(no_edge_counts) + np.tril(no_edge_counts, -1).T

    def non_connections_for_unknown_n(self, n):
        non_edge_counts = sbm.Z.T.dot(self.mask * (1. - self.X)).dot(self.Z)
        edge_difference = np.einsum('tk,l->tkl', np.eye(self.K) - self.Z[n, None], self.non_neighbours_in_group(n))
        non_edge_counts = non_edge_counts[None] + edge_difference
        return np.triu(non_edge_counts) + np.transpose(np.tril(non_edge_counts, -1),[0,2,1])

    def neighbours_in_group(self, n=None):
        n = slice(None) if n is None else n
        graph = self.mask * self.X
        return (graph + graph.T)[n, :].dot(self.Z)

    def non_neighbours_in_group(self, n=None):
        n = slice(None) if n is None else n
        no_graph = self.mask * (1. - self.X)
        return (no_graph + no_graph.T)[n, :].dot(self.Z)

    def sample_w(self):
        self.w = np.random.dirichlet(self.alpha + self.membership())
        return self.w

    def sample_beta(self):
        self.beta = np.triu(np.random.beta(self.a + self.connections(), self.b + self.non_connections()))
        return self.beta

    def cond_log_Z(self, normalized=True):
        positive = np.einsum('il, kl ->ik', self.neighbours_in_group(), np.log(self.symmetric_beta()))
        negative = np.einsum('il, kl ->ik', self.non_neighbours_in_group(), np.log(1. - self.symmetric_beta()))
        w_star = np.log(self.w)[None, :] + positive + negative
        if normalized:
            return w_star - np.logaddexp.reduce(w_star, axis=1, keepdims=True)

    def cond_log_z(self, n, normalized=True):
        positive = np.einsum('l, kl ->k', self.neighbours_in_group(n), np.log(self.symmetric_beta()))
        negative = np.einsum('l, kl ->k', self.non_neighbours_in_group(n), np.log(1. - self.symmetric_beta()))
        w_star = np.log(self.w) + positive + negative
        if normalized:
            return w_star - np.logaddexp.reduce(w_star)
        else:
            return w_star

    def collapsed_cond_log_z(self, n, normalized=True):
        edge_logp = np.triu(betaln(sbm.a + sbm.connections_for_unknown_n(n),
                            sbm.b + sbm.non_connections_for_unknown_n(n)) -
                     betaln(sbm.a, sbm.b))
        membership_with_unknown_n = (sbm.membership() - self.Z[n])[None] + np.eye(self.K)
        marg_prior = multibetaln(sbm.alpha + membership_with_unknown_n) - multibetaln(sbm.alpha * np.ones(self.K))
        logp = marg_prior + edge_logp.sum(axis=(1,2))
        if normalized:
            return logp - np.logaddexp.reduce(logp)
        else:
            return logp

    def sample_z(self, collapsed=False):
        for n in range(self.X.shape[0]):
            if collapsed:
                w_star = np.exp(self.collapsed_cond_log_z(n))
            else:
                w_star = np.exp(self.cond_log_z(n))
            self.Z[n] = np.random.multinomial(1, w_star)
        return self.Z.copy()

    def collapsed_gibbs_sweep(self):
        return self.sample_z(collapsed=True)

    def gibbs_sweep(self):
        w = self.sample_w()
        beta = self.sample_beta()
        Z = self.sample_z(False)
        return (w.copy(), beta.copy(), Z.copy())

    def calculate_test_posterior(self, samples=True, normalize=True, nsamples=1000):
        assert(self.N <= 10 and self.K == 2)
        mikkel_prob = {}
        original_prob = {}
        tf_prob = {}
        if samples:
            Zs = self.samples(nsamples, warmup=10000, collapsed=True)
            sample_freq = defaultdict(int)
            for zs in Zs:
                sample_freq[tuple(zs[:, 1].tolist())] +=1
            sample_prob = {key: value / nsamples for key, value in sample_freq.items()}

        sess = tf.Session()
        for labeling in product([0, 1], repeat=self.N):
            self.Z = np.eye(self.K, dtype='float64')[list(labeling)]
            mikkel_prob[labeling] = self.logp_mikkel()
            original_prob[labeling] = self.logp_original()
            tf_prob[labeling] = sess.run(self.tflogp_op, feed_dict={self.tfZ: self.Z})
        if normalize:
            mikkel_log_partition = np.logaddexp.reduce(list(mikkel_prob.values()))
            original_log_partition = np.logaddexp.reduce(list(original_prob.values()))
            tf_log_partition = np.logaddexp.reduce(list(tf_prob.values()))
            mikkel_prob = {key: np.exp(val - mikkel_log_partition) for key, val in mikkel_prob.items()}
            original_prob = {key: np.exp(val - original_log_partition) for key, val in original_prob.items()}
            tf_prob = {key: np.exp(val - tf_log_partition) for key, val in tf_prob.items()}
        return (mikkel_prob, original_prob, tf_prob) + ((sample_prob,) if samples else ())

    def samples(self, n, warmup=1000, collapsed=False):
        if collapsed:
            for _ in tqdm.trange(warmup):
                self.collapsed_gibbs_sweep()
            return np.stack(self.collapsed_gibbs_sweep() for _ in tqdm.trange(n))
        else:
            for _ in tqdm.trange(warmup):
                self.gibbs_sweep()

            (ws, betas, Zs) = zip(*[self.gibbs_sweep() for _ in tqdm.trange(n)])
            return np.stack(ws), np.stack(betas), np.stack(Zs)

network = 'ambigraph'
colloctest = True
collapsed = True
if network == 'karate':
    from networkx import karate_club_graph, adjacency_matrix

    karate = karate_club_graph()
    X = adjacency_matrix(karate).toarray().astype('float64')
    N = 34
    Ntest = 161
    X = X[:N,:N]

    np.random.seed(8)

    #generate mask of observed edges
    mask = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
    mask = mask < np.sort(mask.ravel())[Ntest]
    mask = np.logical_not(np.logical_or(mask, mask.T))
    mask = np.triu(mask, 1)
if network == 'shavedkarate':
    from networkx import karate_club_graph, adjacency_matrix

    karate = karate_club_graph()
    X = adjacency_matrix(karate).toarray().astype('float64')
    N = 34
    Ntest = 161
    #X = X[2:-2,2:-2]

    np.random.seed(1)

    #generate mask of observed edges
    mask = np.random.randn(N,N) + np.triu(np.inf*np.ones(N))
    mask = mask < np.sort(mask.ravel())[Ntest]
    mask = np.logical_not(np.logical_or(mask, mask.T))
    #mask = np.ones_like(X)
    mask = np.triu(mask, 1)
if network == '2clusters':
    N = 10
    X = np.zeros((2*N,2*N))
    X[:N,:N] =  np.triu(np.ones((N,N)),1)
    X[N:, N:] = np.triu(np.ones((N, N)), 1)
    mask = np.triu(np.ones((2*N,2*N)),1)
if network == 'basic':
    X = np.array([[0,1,0],[0,0,1],[0,0,0]])
    mask = np.triu(np.ones((3,3)),1)
if network == 'ambigraph':
    np.random.seed(704)
    Ncom = 10
    N = 4 * Ncom
    grades = [0.2, 0.5, 0.8]
    group1 = [0,0,1,1]
    group2 = [0,1,0,1]
    Z1 = np.concatenate([np.tile(np.eye(2)[g], (Ncom, 1)) for g in group1])
    Z2 = np.concatenate([np.tile(np.eye(2)[g], (Ncom, 1)) for g in group2])
    strengths = np.array([[grades[(group1[i] == group1[j]) + (group2[i] == group2[j])] for j in range(4)] for i in range(4)])
    X = np.triu(np.block([[np.random.rand(Ncom, Ncom) <= strengths[i,j] for j in range(4)] for i in range(4)]), 1)
    mask = np.triu(np.ones((N, N)), 1)


#sbm = SBM(X, mask, initial_Z=None)
if colloctest and (network == 'karate' or network == 'shavedkarate'):
    colloc = {}
    iterations = 1
    nwarmup = 1000
    nsamples = 10000

    values = [1,5,10,25,50]

    for a in values:
        for b in values:
            np.random.seed(values)
            Z = np.random.multinomial(1, np.ones(2)/2., X.shape[0])
            sbm = SBM(X, mask, a=a, b=b, initial_Z=Z)
            if collapsed:
                Zs = sbm.samples(nsamples, warmup=nwarmup, collapsed=True)
            else:
                ws, betas, Zs = sbm.samples(nsamples, warmup=nwarmup, collapsed=False)
            colloc[(a,b)] = (np.einsum('bik,bjk->ij',Zs,Zs)/nsamples)

    for index, (key, collocation) in enumerate(colloc.items()):
        ax = plt.subplot(len(values),len(values), index + 1)
        ax.matshow(collocation, vmin=0, vmax=1, cmap='RdBu')
        plt.axis('off')
        plt.title(key)
elif  network == "ambigraph":
    print("running ambigraph")
    nwarmup = 1000
    nsamples = 10000
    np.random.seed(0)
    colloc = {}
    for index, Zinit in enumerate([Z1,Z2] + [np.random.multinomial(1, np.ones(2)/2., X.shape[0]) for _ in range(6)]):
        sbm = SBM(X, mask, initial_Z=Zinit)
        Zs = sbm.samples(nsamples, warmup=nwarmup, collapsed=True)
        colloc[index] = (np.einsum('bik,bjk->ij',Zs,Zs)/nsamples)

    for key in range(8):
        ax = plt.subplot(2, 4, key + 1)
        collocation = colloc[key]
        ax.matshow(collocation, vmin=0, vmax=1, cmap='RdBu_r')
        plt.axis('off')
        plt.title(key)