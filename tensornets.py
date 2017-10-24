import numpy as np
import tensorflow as tf
from rebar import categorical_forward, categorical_backward, REBAR, select_max, sigma
from tensorflow.python.ops.distributions.util import fill_lower_triangular

def orthogonalize(A):
    if len(A.get_shape())<3:
        A = tf.expand_dims(A,0)
    batchT = lambda X: tf.transpose(X, [0,2,1])
    L = tf.cholesky(tf.matmul(batchT(A), A))
    QT = tf.matrix_triangular_solve(L, batchT(A))
    return batchT(QT)

class OrthogonalComplex:
    def __init__(self, N, K, rank):
        self._var = tf.Variable(0.1*np.random.randn(*(N,K*rank,rank)), dtype='float32')
        self.U = orthogonalize(self._var)
        self.cores = tf.transpose(tf.reshape(self.U, (N, rank, K, rank)), [0,2,1,3])

    def __value__(self):
        return self.cores

    def variable(self):
        return self._var

class MPS:
    def __init__(self, N, K, rank, cores=None, normalized=True):
        self.N  = N
        self.K = K
        self.rank = rank

        if cores is not None:
            self.raw_cores = cores
        else:
            self.raw_cores = tf.Variable(tf.random_normal((self.N, self.K,
                                                   self.rank, self.rank)), name="cores")

        if normalized:
            self.cores = self.raw_cores/tf.exp(tf.log(self.scale())/(2*self.N))
        else:
            self.cores = tf.identity(self.raw_cores)

        with tf.name_scope("marginalization"):
            with tf.name_scope("auxiliary"):
                flip_cores = tf.reverse(tf.transpose(self.cores,
                                                     [0, 1, 3, 2]),
                                        [0])
                initial = tf.ones((self.rank, self.rank))
                inner_marg = tf.scan(self.inner_contraction,
                                     self.cores,
                                     initializer=initial)
                outer_marg = tf.scan(self.inner_contraction,
                                     flip_cores,
                                     initializer=initial)

            #add boundary elements (1-vectors) and remove full products
            self.inner_marginal = tf.concat([tf.expand_dims(tf.ones((self.rank, self.rank)), 0),
                                                  inner_marg[:-1]],
                                                  axis=0, name="inner_marginal_cores")
            self.outer_marginal = tf.concat([tf.reverse(outer_marg[:-1], [0]),
                                                   tf.expand_dims(tf.ones((self.rank, self.rank)), 0)],
                                                  axis=0, name="outer_marginal_cores")
    def contraction(self, Z, normalized=True):
        if normalized:
            cores = self.cores
        else:
            cores = self.raw_cores
        S = tf.ones((self.rank, self.rank))
        for core, z in zip(tf.unstack(cores), tf.unstack(Z)):
            S = self.inner_contraction(S, core, z)
        return tf.reduce_sum(S)

    def scale(self):
        return self.contraction(tf.ones((self.N, self.K)), normalized=False)

    def zipper_forward(self):
        #uniform = tf.random_uniform(shape=(self.N, self.K))
        #gumbels = - tf.log( - tf.log(uniform + eps) + eps, name="gumbel")

        condition = tf.ones((self.rank, self.rank))
        distribution = tf.einsum('kij,ji', self.inner_broadcast(condition, self.cores[0]), self.outer_marginal[0])
        z = categorical_forward(distribution)
        Z = [z]
        sample = tf.argmax(z)

        for index in range(1, self.N):
            update = self.cores[index-1][sample]
            condition = tf.einsum('ik,kl,lj', update, condition, tf.transpose(update))
            distribution = tf.einsum('kij,ji', self.inner_broadcast(condition, self.cores[index]), self.outer_marginal[index])
            z = categorical_forward(tf.log(distribution))
            sample = tf.argmax(z)
            Z += [z]
        return tf.stack(Z)

    def zipper_backward(self, b):
        condition = tf.ones((self.rank, self.rank))
        P = []
        for index, bi in enumerate(tf.unstack(b)):
            #update = self.cores[index][tf.argmax(b[index])]
            update = tf.reduce_sum(self.cores[index]*tf.reshape(bi, (-1,1,1)), axis=0)
            condition = tf.einsum('ik,kl,lj', update, condition, tf.transpose(update))
            distribution = tf.einsum('kij,ji', self.inner_broadcast(condition, self.cores[index]), self.outer_marginal[index])
            P += [distribution]
        return categorical_backward(tf.log(tf.stack(P)), b)

    def marginals(self):
        return tf.einsum('nkiu,nur,nkrs,nsi->nk',tf.transpose(self.cores,[0, 1, 3, 2]),
                         self.inner_marginal, self.cores, self.outer_marginal)

    def inner_contraction(self, density, core, weights = None):
        if weights is not None:
            return tf.reduce_sum(tf.reshape(weights, (-1, 1, 1)) * self.inner_broadcast(density, core), axis=0)
        else:
            return tf.einsum('krs,su,kut', tf.transpose(core, [0,2,1]), density, core)

    def inner_broadcast(self, density, core):
        return tf.einsum('krs,su,kut->krt', tf.transpose(core, [0,2,1]), density, core)

    def populatetensor(self, bruteforce=False):
        if bruteforce:
            Z = np.zeros([self.K,]*self.N)
            codes = np.eye(self.K)
            insert = tf.placeholder(tf.float32, shape=(self.N, self.K))
            fill_op = self.contraction(insert)
            for index in np.ndindex(*Z.shape):
                Z[index] = tf.get_default_session().run(fill_op, feed_dict={insert:codes[list(index)]})
        else:
            def standardcore(C):
                return tf.transpose(C, [1,0,2])
            def core2orthU(C):
                return tf.reshape(C, (-1, self.rank))
            def core2orthV(C):
                return tf.reshape(C, (self.rank, -1))
            Z = standardcore(tf.reduce_sum(self.cores[0], axis=1, keep_dims=True))
            for core in tf.unstack(self.cores[1:-1]):
                stdcore = standardcore(core)
                Z = tf.matmul(core2orthU(Z), core2orthV(stdcore))
            stdcore = standardcore(tf.reduce_sum(self.cores[-1], axis=2, keep_dims=True))
            Z = tf.matmul(core2orthU(Z), core2orthV(stdcore))
            Z = tf.square(tf.reshape(Z, [self.K,]*self.N))
        return Z

    def symmetrynorm(self):
        swap_cores = tf.concat([tf.expand_dims(self.cores[:, 1, :, :], 1),
                                tf.expand_dims(self.cores[:, 0, :, :], 1),
                                self.cores[:, 2:, :, :]], axis=1)
        cycle_cores = tf.concat([tf.expand_dims(self.cores[:, -1, :, :], 1),
                                self.cores[:, :-1, :, :]], axis=1)


        initial = tf.ones((self.rank, self.rank))
        swapnorm = tf.reduce_sum(tf.foldl(self.inner_contraction,
                                          self.cores - swap_cores,
                                          initializer=initial))
        cyclenorm = tf.reduce_sum(tf.foldl(self.inner_contraction,
                                          self.cores - cycle_cores,
                                          initializer=initial))
        return cyclenorm

    def marginalentropy(self):
        marginals = self.marginals()
        return -tf.reduce_sum(marginals*tf.log(marginals))

if __name__ is "__main__":
    sess = tf.InteractiveSession()
    N = 6
    K = 3
    R = 4
    model = MPS(N, K, R)
    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.reduce_sum(z) - tf.reduce_sum(z*true_index)

    C = np.array([[0,1,0],[0,0,1],[1,0,0]])
    #ind =
    #model.contraction()
    orth = OrthogonalComplex(N, K, R)
    sess.run(tf.global_variables_initializer())


    if False:
        REBAR(f, model.cores, )

        Z = tf.Variable(true_index, dtype=tf.float32)
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

