import numpy as np
import tensorflow as tf
from itertools import product, chain
from tfutils import tffunc, tfmethod

import tqdm
EPS = 1e-16

dtype='float64'

@tffunc(1)
def replaceZi(Z, i, shape=None):
    '''
    take Z (NxK) and make K copies where the i'th element is replaced
    consecutively by all K-length one-hot vectors. Returns KxNxK
    '''
    if shape is None:
        N, K = [s.value for s in tf.get_shape(Z)]
    else:
        N, K = shape
    one_hot = tf.one_hot(i, N, dtype=dtype)
    replace = tf.eye(K, dtype=dtype)-tf.expand_dims(Z[i], 0)
    return tf.expand_dims(Z, 0) + tf.einsum('i,kl->kil',
                                           one_hot, replace)

class KLcorrectedBound:
    '''Compute KL-corrected variational bound for a q-model of independent
    single-sample mulitnomial/categorical distributions.
    The true log-joint distribution is assumed to be *linear* in q parameters.
    '''
    def __init__(self, model, data, params, batch=True, **kwargs):
        self.model = model
        self.data = data
        self.params = params
        self.Zs = [tf.nn.softmax(param) for param in self.params]
        if batch:
            self.entropy = [-tf.reduce_sum(Z*tf.log(Z), axis=[1,2]) for Z in self.Zs]
            self.bound = self.model.batch_logp(self.Zs, self.data, **kwargs) + tf.reduce_sum(self.entropy, axis=0)
        else:
            self.entropy = [-tf.reduce_sum(Z*tf.log(Z)) for Z in self.Zs]
            self.bound = self.model.logp(self.Zs, self.data, **kwargs) + tf.reduce_sum(self.entropy)
        self.objective = -tf.reduce_mean(self.bound)
        self.gradients = tf.gradients(self.objective, self.Zs)

    def update_op(self):
        '''update using coordinate ascent'''
        return tf.group(*[param.assign(-grad) for grad, param in zip(self.gradients, self.params)])

    def minimize(self):
        '''minimize completely using Scipy BFGS'''
        opt = tf.contrib.opt.ScipyOptimizerInterface(self.objective, self.params)
        opt.minimize()

class CollapsedMixture:
    def __init__(self, N, K):
        self.N = N #observations
        self.K = K #components

    def logp(self, Z, X):
        raise(NotImplementedError)

    #COSTLY
    @tfmethod(2)
    def logp_conditionals(self, Z, X):
        '''
        compute all complete (Gibbs) conditionals by making K replacements with replaceZi for each index,
        then applying logp to all conditionals.
        '''
        logpx = lambda Z: self.logp(Z, X)

        Zi = tf.map_fn(lambda x: (replaceZi(Z, x, (self.N, self.K))), tf.range(tf.shape(Z)[0]),dtype=dtype)
        return tf.map_fn(lambda x: tf.map_fn(logpx, x, dtype=dtype), Zi, dtype=dtype) #<--

    @tfmethod(1)
    def populatetensor(self, X, observed=None, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        if observed is None:
            observed = self._defaultobserved
        shape = self.N*(self.K,)
        size = self.K**self.N
        Zstar_vec = np.zeros(size, dtype=dtype)
        Z = tf.placeholder(dtype=dtype, shape=(self.N, self.K))
        logpx = self.logp(Z, X, observed=observed)
        for ind in tqdm.tqdm(np.ndindex(shape), total=size):
             Zstar_vec[np.ravel_multi_index(ind, shape)] = sess.run(logpx, feed_dict={Z: np.eye(self.K, dtype=np.float32)[ind, :]})
        return np.reshape(Zstar_vec, shape)

class CollapsedMultipartite(CollapsedMixture):
    def __init__(self, Ns, Ks):
        self.Ns = Ns #observations
        self.Ks = Ks #components
        self.nsets = len(self.Ns)

    def logp_conditionals(self, Zs, X):
        '''
        compute all complete (Gibbs) conditionals by making K replacements with replaceZi for each index,
        then applying logp to all conditionals.
        '''
        logp_cond = []
        for index, Z, N, K in enumerate(self.Zs, self.Ns, self.Ks):
            logpx = lambda Z: self.logp(Zs[:index] + [Z] + Zs[(index+1):], X)

            Zi = tf.map_fn(lambda x: (self.replaceZi(Z, x, (N, K))), tf.range(tf.shape(Z)[0]),dtype=dtype)
            logp_cond += [tf.map_fn(lambda x: tf.map_fn(logpx, x, dtype=dtype), Zi, dtype=dtype)]
        return logp_cond

    @tfmethod(1)
    def populatetensor(self, X, observed=None):
        if observed is None:
            observed = self._defaultobserved
        sess = tf.get_default_session()
        shapes = [N*(K, ) for N, K in zip(self.Ns, self.Ks)]
        shape = tuple(chain(*shapes))
        size = np.prod(shape)

        Zstar_vec = np.zeros(size, dtype=dtype)
        Zs = [tf.placeholder(dtype=dtype, shape=(N, K)) for N, K in zip(self.Ns, self.Ks)]
        logpx = self.logp(Zs, X, observed=observed)

        for ind in tqdm.tqdm(product(*[np.ndindex(shape) for shape in shapes]), total=size):
             Zstar_vec[np.ravel_multi_index(tuple(chain(*ind)), shape)] = sess.run(logpx, feed_dict=dict(zip(Zs, [np.eye(K, dtype=dtype)[indi, :] for indi, K in zip(ind, self.Ks)])))
        return np.reshape(Zstar_vec, shape)

class CollapsedStochasticBlock(CollapsedMixture):
    def __init__(self, N, K, alpha=1., a=1., b=1.):
        super().__init__(N, K)
        self.alpha = tf.convert_to_tensor(alpha, dtype=dtype)
        self.a = tf.convert_to_tensor(a, dtype=dtype)
        self.b = tf.convert_to_tensor(b, dtype=dtype)
        self._defaultobserved = tf.convert_to_tensor(np.triu(np.ones((self.N, self.N), dtype=dtype), 1))

    @tfmethod(2)
    def logp(self, Z, X, observed=None):
        #unpack from singleton list
        if len(Z.shape)==3:
            Z = Z[0]

        if observed is None:
            observed = tf.convert_to_tensor(np.triu(np.ones((self.N, self.N), dtype=dtype), 1))

        membership = tf.reduce_sum(Z, axis=0, keepdims=True)
        edgecounts = tf.einsum('mk,mn,nl', Z, observed*X, Z) #Z^T*X*Z
        notedgecounts = tf.einsum('mk,mn,nl', Z, observed, Z) - edgecounts

        lnprior = tf.reduce_sum(tf.lbeta((self.alpha +
                                                      membership)) -
                                tf.lbeta((self.alpha +
                                                      tf.zeros_like(membership))))
        lnlink = tf.reduce_sum(tf.lbeta(tf.stack([self.a + edgecounts,
                                                  self.b + notedgecounts],
                                                  axis=2)) -
                               tf.lbeta(tf.stack([self.a + tf.zeros_like(edgecounts),
                                                  self.b + tf.zeros_like(notedgecounts)],
                                                  axis=2)))
        return lnprior + lnlink

    def sample(self):
        w = tf.distributions.Dirichlet(alpha*tf.ones(self.K, dtype=dtype)).sample()
        Z = tf.distributions.Multinomial(1., w).sample(self.N)
        beta = tf.distributions.Beta(tf.cast(self.a, dtype), tf.cast(self.b, dtype)).sample((self.K, self.K))
        linkprob = tf.matmul(Z, tf.matmul(beta, tf.transpose(Z)))
        X = tf.distributions.Bernoulli(linkprob).sample()
        return (w, Z, beta, X)

    def conditional_sample(self, Z):
        beta = tf.distributions.Beta(tf.cast(self.a, dtype), tf.cast(self.b, dtype)).sample((self.K, self.K))
        linkprob = tf.matmul(Z, tf.matmul(beta, tf.transpose(Z)))
        X = tf.distributions.Bernoulli(linkprob).sample()
        return (beta, X)


    @tfmethod(2)
    def batch_logp(self, Z, X, observed=None):
        #unpack from singleton list
        if len(Z.shape)==4:
            Z = Z[0]
        if observed is None:
            observed = self._defaultobserved
        else:
            observed = tf.convert_to_tensor(observed, dtype)
        membership = tf.reduce_sum(Z, axis=1, keepdims=True)
        edgecounts = tf.einsum('bmk,mn,bnl->bkl', Z, observed*X, Z) #Z^T*X*Z
        notedgecounts = tf.einsum('bmk,mn,bnl->bkl', Z, observed, Z) - edgecounts


        lnprior = tf.squeeze(tf.lbeta(self.alpha + membership) -
                             tf.lbeta(self.alpha + tf.zeros_like(membership)))
        lnlink = tf.reduce_sum(tf.lbeta(tf.stack([self.a + edgecounts,
                                                  self.b + notedgecounts],
                                                  axis=3)) -
                               tf.lbeta(tf.stack([self.a + tf.zeros_like(edgecounts),
                                                  self.b + tf.zeros_like(notedgecounts)],
                                                  axis=3)), axis=[1,2])
        return lnprior + lnlink


    @tfmethod(2)
    def batch_logpred(self, Z, X, observed=None):
        if observed is None:
            observed = tf.ones((self.N, self.N), dtype=dtype)
        else:
            observed = tf.convert_to_tensor(observed, dtype)
        membership = tf.reduce_sum(Z, axis=1, keepdims=True)
        edgecounts = tf.einsum('bmk,mn,bnl->bkl', Z, observed*X, Z) #Z^T*X*Z
        notedgecounts = tf.einsum('bmk,mn,bnl->bkl', Z, observed, Z) - edgecounts


        lnlink = tf.reduce_sum(tf.lbeta(tf.stack([self.a + edgecounts,
                                                  self.b + notedgecounts],
                                                  axis=2)) -
                               tf.lbeta(tf.stack([self.a + tf.zeros_like(edgecounts),
                                                  self.b + tf.zeros_like(notedgecounts)],
                                                  axis=2)), axis=[1,2])
        return lnlink


class CollapsedBipartiteStochasticBlock(CollapsedMultipartite):
    def __init__(self, Ns, Ks, alpha0=1., a=1., b=1.):
        super().__init__(Ns, Ks)
        self.alphas = [alpha0*tf.ones((1, K), dtype=dtype) for K in self.Ks]#2xK
        self.a = tf.convert_to_tensor(a, dtype=dtype)
        self.b = tf.convert_to_tensor(b, dtype=dtype)

    @tfmethod(2)
    def suffstats(self, Zs, X):
        memberships = [tf.reduce_sum(Z, axis=0, keepdims=True) for Z in Zs]
        edgecounts = tf.matmul(tf.matmul(Zs[0], X, transpose_a=True), Zs[1])
        notedgecounts = tf.matmul(memberships[0], memberships[1], transpose_a=True) - edgecounts
        return memberships, edgecounts, notedgecounts

    @tfmethod(2)
    def logp(self, Zs, X):
        memberships, edgecounts, notedgecounts = self.suffstats(Zs, X)
        lnprior = tf.reduce_sum([tf.lbeta(a+m) - tf.lbeta(a)
                                 for a, m in zip(self.alphas, memberships)])


        lnlink = tf.reduce_sum(tf.lbeta(tf.stack([self.a + edgecounts,
                                                  self.b + notedgecounts],
                                                  axis=2)) -
                               tf.lbeta(tf.stack([[[self.a]], [[self.b]]], axis=2)))
        return lnprior + lnlink


@tffunc(1)
def lgammap(a, p):
    #multivariate log-gamma
    return p*(p-1.)/4.*np.log(np.pi)+tf.reduce_sum(tf.lgamma(a+(1.-tf.range(1,p+1, dtype=dtype)/2.)))


class CollapsedGaussianMixture(CollapsedMixture):
    def __init__(self, N, K, D, kappa=None, prec=None, nu=None, alpha=None):
        super().__init__(N, K)
        self.D = D #dimensions

        #Normal mixture with normal-Wishart prior for the mean-precision parameters.
        with tf.variable_scope("logp_{}".format(id(self))) as scope:
            self.varscope = scope

            if kappa is None:
                lnkappa0 = tf.get_variable("lnkappa0", shape=(), initializer=tf.zeros_initializer, dtype=dtype)
            else:
                lnkappa0 = tf.get_variable("lnkappa0", initializer=tf.convert_to_tensor(tf.log(tf.exp(kappa)-1.), dtype=dtype), dtype=dtype)
            self.kappa0 = tf.nn.softplus(lnkappa0)

            self.nu0 = tf.convert_to_tensor(self.D + 10. if nu is None else nu, dtype=dtype) #degrees of freedom
            if prec is None:
                lnprec0 = tf.get_variable("lnprec0", shape=(), initializer=tf.zeros_initializer, dtype=dtype)
            else:
                lnprec0 = tf.get_variable("lnprec0", initializer=tf.convert_to_tensor(tf.log(tf.exp(prec)-1.), dtype=dtype), dtype=dtype)
            self.Psiinv0 = tf.nn.softplus(lnprec0)*tf.eye(self.D,dtype=dtype)

            self.alpha0 = tf.convert_to_tensor(1*np.ones(self.K, dtype=dtype) if alpha is None else alpha, dtype=dtype) #hyperparams of Dirichlet prior on cluster assignment

            self.var_list0 = [lnkappa0, lnprec0]

    @tfmethod(1)
    def logdet(self, M):
        #log-determinant for symmetric positive-definite
        return 2.*tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(M+EPS*np.eye(self.D)))))

    @tfmethod(3)
    def A(self, Psiinv, kappa, nu):
        #log-normalizer for normal-Wishart
        with tf.variable_scope(self.varscope, reuse=True):
            return -nu/2.*self.logdet(Psiinv)+(self.D/2.)*(nu*np.log(2.)-tf.log(kappa)+lgammap(nu/2.,self.D)+np.log(2.*np.pi))

    #COSTLY
    @tfmethod(1)
    def logp(self, Z, X):
        with tf.variable_scope(self.varscope, reuse=True):
            Z = tf.convert_to_tensor(Z)

            #sufficient statistics
            m = tf.reduce_sum(Z, 0)
            muest = tf.matmul(tf.transpose(Z), X)/(EPS+tf.expand_dims(m, 1))
            S = tf.einsum('nk,nij->kij', Z, tf.einsum('ni,nj->nij', X, X))
            mmt = tf.einsum('k,kij->kij', m**2./(self.kappa0+m),
                            tf.einsum('ki,kj->kij', muest, muest))

            #posterior parameters
            kappa = self.kappa0 + m
            nu = self.nu0 + m
            Psiinv = self.Psiinv0 + S - mmt
            alpha = self.alpha0 + m

            logprobw = tf.reduce_sum(tf.lgamma(alpha)-tf.lgamma(self.alpha0))+tf.lgamma(tf.reduce_sum(self.alpha0))-tf.lgamma(tf.reduce_sum(alpha))
            logcompprob = lambda S: self.A(S[0], S[1], S[2])
            logprob = (tf.reduce_sum(tf.map_fn(logcompprob, (Psiinv, kappa, nu),
                                               dtype=dtype)) -
                       self.K*self.A(self.Psiinv0, self.kappa0, self.nu0))
            #note: reduce+map is comparable or marginally better than fold/scan.
            return logprob+logprobw

if __name__ is "__main__":
    import matplotlib.pyplot as plt

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

    karate = True
    if karate:
        Ns = (34, 34)
        Ks = (3, 3)
        A = Akarate
    else:
        Ns = (10, 10)
        Ks = (2, 2)
        blocks = np.random.randn(*Ns) > np.kron(np.array([[0.05, 1.],[1., 0.05]]), np.ones((Ns[0]//2,Ns[1]//2)))
        A = np.logical_or(blocks, (np.random.rand(*Ns) > 0.999)).astype('float64')

    sess = tf.InteractiveSession()
    bipart = CollapsedBipartiteStochasticBlock(Ns, Ks, a=1, alpha0=100.)
    #P = bipart.populatetensor(A)
    Zs = [tf.Variable(3*tf.random_normal((N, K), dtype=dtype)) for N, K in zip(Ns, Ks)]
    bound = KLcorrectedBound(bipart, A, Zs)
    update = bound.update_op()
    sess.run(tf.global_variables_initializer())
    bound.minimize()

    Z0, Z1 = sess.run(bound.Zs)

    ord0 = np.argsort(Z0.dot(Ks[0]**np.arange(3)))
    ord1 = np.argsort(Z1.dot(Ks[1]**np.arange(3)))
    plt.matshow(A[np.ix_(ord0, ord1)])

    hardord0 = np.argmax(Z0[ord0], axis=1)
    hardord1 = np.argmax(Z1[ord1], axis=1)

    plt.hlines([ind+0.5 for ind in np.where(np.diff(hardord0))], -0.5, 33.5, color='r')
    plt.vlines([ind+0.5 for ind in np.where(np.diff(hardord1))], -0.5, 33.5, color='r')


    #   #for _ in range(10):
    #    sess.run(tf.global_variables_initializer())
    #    L = []
        #for _ in range(10000):
        #    l, _ = sess.run([bound.bound, update])
        #    L.append(l)
        #plt.plot(L)
