import numpy as np
import tensorflow as tf
from rebar import categorical_forward, categorical_backward, REBAR, select_max, sigma
from tensorflow.python.ops.distributions.util import fill_lower_triangular
dtype = 'complex64'
import matplotlib.pyplot as plt
bitify = lambda x: np.sum(K**np.arange(N//2)*x)
def mosaic(Z):
    N = Z.ndim
    K = Z.shape[0]
    A = np.zeros((K**(N//2), K**(N//2)))
    for ind in np.ndindex(Z.shape):
        coord1 = ind[::2]
        coord2 = ind[1::2]
        A[bitify(coord1), bitify(coord2)] = np.log(np.maximum(Z[ind],0.))
    return A

def orthogonalize(A):
    if len(A.get_shape())<3:
        A = tf.expand_dims(A,0)
    batchT = lambda X: tf.transpose(X, [0,2,1])
    L = tf.cholesky(tf.matmul(batchT(A), A)+1e-10*np.eye(A.shape[2],
                                                         dtype=dtype))
    QT = tf.matrix_triangular_solve(L, batchT(A))
    return batchT(QT)

def randomorthogonal(shape):
    A = np.random.randn(np.maximum(*shape), np.maximum(*shape))
    return np.linalg.qr(A)[0][:shape[0], :shape[1]].astype(dtype)

def tfkron(A,B):
    return tf.reshape(A[:, None, :, None] * B[None, :, None, :],
                      (A.shape[0].value*B.shape[0].value,
                       A.shape[1].value*B.shape[1].value))

def tfkron2(A):
    return tfkron(A, A)

def multikron(A, B):
    return tf.stack([tfkron(a, b) for a, b in zip(tf.unstack(A), tf.unstack(B))])

class Core:
    def __init__(self, N, K, ranks, cores=None):
        self.N  = N
        self.K = K
        self.ranks = ranks
        assert(len(ranks) == N+1)
        assert(ranks[0] == 1 & ranks[-1] == 1)

        if cores is not None:
            self.cores = cores
        else:
            restack = lambda A, rank0, rank1: \
                      np.transpose(np.reshape(A, (rank0, self.K, rank1)),
                                   [1,0,2])
            self.cores = [tf.Variable(restack(randomorthogonal((self.K*rank0,
                                                               rank1)),
                                              rank0, rank1)
                                     )
                              for rank0, rank1 in zip(self.ranks[:-1],
                                                      self.ranks[1:])]

    def scaledcores(self, factor):
        nroot = tf.exp((1./self.N)*tf.log(factor))
        return [nroot*core for core in self.cores]

class OrthogonalCore(Core):
    def __init__(self, N, K, rank):
        self.N  = N
        self.K = K
        self.ranks = ranks
        assert(len(ranks) == N+1)
        assert(ranks[0] == 1 & ranks[-1] == 1)

        self._var = [tf.Variable(tf.random_normal((self.K, rank0,
                                                            rank1),dtype=dtype))
                              for rank0, rank1 in zip(self.ranks[:-1],
                                                      self.ranks[1:])]
        self.cores = [orthogonalize(x) for x in self._var]

class MPS:
    def __init__(self, N, K, ranks, cores=None, normalized=True):
        self.N  = N
        self.K = K
        self.ranks = ranks
        if cores:
            self.raw = cores
        else:
            self.raw = Core(N, K, ranks, cores)
        self.cores = self.raw.scaledcores((1./tf.sqrt(self.scale()))
                                          if normalized else 1.)
        self.transfer = [tf.map_fn(tfkron2, core) for core in self.cores]
        with tf.name_scope("marginalization"):
            with tf.name_scope("auxiliary"):
                flip_cores = [tf.transpose(core, [0,2,1])
                              for core in self.cores[-1::-1]]
                initial = tf.ones((1., 1.), dtype=dtype)
                inner_marg = [tf.einsum('kri,krj->ij',
                                        self.cores[0], self.cores[0])]
                for core in self.cores[1:-1]:
                    inner_marg += [self.inner_contraction(inner_marg[-1], core)]

                outer_marg = [tf.einsum('kir,kjr->ij',
                                        self.cores[-1], self.cores[-1])]
                for core in flip_cores[1:-1]:
                    outer_marg += [self.inner_contraction(outer_marg[-1], core)]

            #add boundary elements (1-vectors) and remove full products
            self.inner_marginal = [initial, *inner_marg]
            self.outer_marginal = [*outer_marg[-1::-1], initial]

    def contraction(self, Z, normalized=True):
        if normalized:
            cores = self.cores
        else:
            cores = self.raw.cores
        S = tf.ones((1,1), dtype=dtype)
        for core, z in zip(cores, tf.unstack(Z)):
            S = self.inner_contraction(S, core, z)
        return tf.reduce_sum(S)

    def scale(self):
        return self.contraction(tf.ones((self.N, self.K), dtype=dtype),
                                normalized=False)

    def zipper_forward(self):
        #uniform = tf.random_uniform(shape=(self.N, self.K))
        #gumbels = - tf.log( - tf.log(uniform + eps) + eps, name="gumbel")

        condition = tf.ones((1., 1.), dtype=dtype)
        distribution = tf.einsum('kij,ji', self.inner_broadcast(condition, self.cores[0]), self.outer_marginal[0])
        z = categorical_forward(distribution)
        Z = [z]
        sample = tf.argmax(z)

        for index in range(1, self.N):
            update = self.cores[index-1][sample]
            condition = tf.einsum('ik,kl,lj', tf.transpose(update), condition, update)
            distribution = tf.einsum('kij,ji', self.inner_broadcast(condition, self.cores[index]), self.outer_marginal[index])
            z = categorical_forward(tf.log(distribution))
            sample = tf.argmax(z)
            Z += [z]
        return tf.stack(Z)

    def zipper_backward(self, b):
        condition = tf.ones((self.rank, self.rank), dtype=dtype)
        P = []
        for index, bi in enumerate(tf.unstack(b)):
            #update = self.cores[index][tf.argmax(b[index])]
            update = tf.reduce_sum(self.cores[index]*tf.reshape(bi, (-1,1,1)), axis=0)
            condition = tf.einsum('ik,kl,lj', update, condition, tf.transpose(update))
            distribution = tf.einsum('kij,ji', self.inner_broadcast(condition, self.cores[index]), self.outer_marginal[index])
            P += [distribution]
        return categorical_backward(tf.log(tf.stack(P)), b)

    def marginals(self):
        return tf.stack([tf.einsum('kiu,ur,krs,si->k',
                                   tf.transpose(core,[0, 2, 1]),
                                   inner_marg, core, outer_marg)
                         for core, inner_marg, outer_marg in
                         zip(self.cores,
                             self.inner_marginal,
                             self.outer_marginal)])

    def inner_contraction(self, density, core, weights = None):
        if weights is not None:
            return tf.reduce_sum(tf.reshape(weights, (-1, 1, 1)) * self.inner_broadcast(density, core), axis=0)
        else:
            return tf.einsum('krs,su,kut', tf.transpose(core, [0,2,1]), density, core)

    def inner_broadcast(self, density, core):
        return tf.einsum('krs,su,kut->krt', tf.transpose(core, [0,2,1]), density, core)

    def populatetensor(self, bruteforce=False):
        if bruteforce:
            Z = np.zeros([self.K,]*self.N).astype(dtype)
            codes = np.eye(self.K, dtype=dtype)
            insert = tf.placeholder(dtype, shape=(self.N, self.K))
            fill_op = self.contraction(insert)
            for index in np.ndindex(*Z.shape):
                Z[index] = tf.get_default_session().run(fill_op, feed_dict={insert:codes[list(index)]})
        else:
            def standardcore(C):
                return tf.transpose(C, [1,0,2])
            def core2orthU(C, rank):
                return tf.reshape(C, (-1, rank))
            def core2orthV(C, rank):
                return tf.reshape(C, (rank, -1))
            Z = standardcore(self.cores[0])
            for core, rank in zip(self.cores[1:], self.ranks[1:]):
                stdcore = standardcore(core)
                Z = tf.matmul(core2orthU(Z, rank), core2orthV(stdcore, rank))
            Z = tf.square(tf.reshape(Z, [self.K,]*self.N))
        return Z

    def symmetrynorm(self):
        swap = lambda X: tf.concat([X[None, 1],
                                    X[None, 0],
                                    X[2:]], axis=0)
        swap_cores = [swap(core) for core in self.cores]

        cycle = lambda X: tf.concat([X[None, -1],
                                     X[:-1]], axis=0)
        cycle_cores = [cycle(core) for core in self.cores]
        transfer = [multikron(core, core) for core in self.cores]
        transfer_swap = [multikron(core, swap_core) for core, swap_core in
                         zip(self.cores, swap_cores)]
        transfer_cycle = [multikron(core, cycle_core) for core, cycle_core in
                          zip(self.cores, cycle_cores)]

        S = tf.ones((1,1), dtype=dtype)
        Sswap = tf.ones((1,1), dtype=dtype)
        Scycle = tf.ones((1,1), dtype=dtype)
        for core, score, ccore in zip(transfer, transfer_swap, transfer_cycle):
            S = self.inner_contraction(S, core)
            Sswap = self.inner_contraction(Sswap, score)
            Scycle = self.inner_contraction(Scycle, ccore)
        return tf.real((S - Sswap)/(S)), tf.real((S-Scycle)/(S))


    def marginalentropy(self):
        marginals = self.marginals()
        return -tf.reduce_sum(marginals*tf.log(marginals))

if __name__ is "__main__":
    sess = tf.InteractiveSession()
    N = 6
    K = 3
    R = 9
    ranks = (1,3,6) + (R,)*(N-5) + (6,3,1)
    model = MPS(N, K, ranks)
    opt = tf.contrib.opt.ScipyOptimizerInterface(tf.reduce_sum(model.symmetrynorm()), model.raw.cores)

    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.reduce_sum(z) - tf.reduce_sum(z*true_index)

    C = np.array([[0,1,0],[0,0,1],[1,0,0]])
    #ind =
    #model.contraction()
    #orth=OrthogonalCore(N, K, R)
    #rth_model = MPS(N, K, ranks, cores=orth)
    sess.run(tf.global_variables_initializer())
    opt.minimize()

    if False:
        REBAR(f, model.cores, )

        Z = tf.Variable(true_index, dtype=tf.float64)
        temp_var = tf.Variable(1.)
        temp = tf.nn.softplus(temp_var)
        nu_var = tf.Variable(1.)
        nu_switch = tf.Variable(1.)
        nu = nu_switch*tf.nn.softplus(nu_var)

        grad, loss, var_grad = categoricalREBAR(f, Z, nu, temp, K, var_params = [temp_var, nu_var])

        grad_estimator = grad[0]

        opt = tf.train.AdamOptimizer()
        train_step = opt.apply_gradients(var_grad)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

