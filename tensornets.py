import numpy as np
import tensorflow as tf
from relaxflow.reparam import CategoricalReparam, categorical_forward, categorical_backward
dtype = 'float64'

import matplotlib.pyplot as plt
from tfutils import tffunc, tfmethod
select_max = lambda z, K: tf.one_hot(tf.argmax(z, axis=-1), K, dtype=dtype)
bitify = lambda x: np.sum(K**np.arange(x.size)*x)

def mosaic(Z, split=None):
    '''matrix unfolding of tensor with even order with interpretable
    row/column ordering.
    Specifically, if it has order N, with each order having dimension K, we
    can associate every element with an index of length N taking values in
    {0,...,K-1}. Mosaic splits the index set into two vectors of N/2, one with
    the odd indices and the other with the even. Both vectors are then encoded
    as a base-K number. Each tensor element is then mapped to a row-column
    index in the output matrix based on the base-K codes.

    Z - tensor of even order N with a regular dimension of K.
    M - matrix unfolding of tensor Z.
    '''
    N = Z.ndim
    K = Z.shape[0]
    if split is None:
        split = N//2
    #bitify = lambda x, K: np.sum(K**np.arange(x.size)*x)
    A = np.zeros((K**split, K**(N-split)))
    for ind in np.ndindex(Z.shape):
        coord1 = ind[:split]
        coord2 = ind[split:]
        A[bitify(coord1), bitify(coord2)] = np.log(np.maximum(Z[ind], 0.))
    return A

def randomorthogonal(shape):
    '''Returns a random orthogonal matrix of shape Shape'''
    A = np.random.randn(np.maximum(*shape), np.maximum(*shape))
    return np.linalg.qr(A)[0][:shape[0], :shape[1]].astype(dtype)

@tffunc(2)
def tfkron(A, B):
    '''Takes the kronecker product of A and B'''
    return tf.reshape(A[:, None, :, None] * B[None, :, None, :],
                      (A.shape[0].value*B.shape[0].value,
                       A.shape[1].value*B.shape[1].value))

@tffunc(1)
def tfkron2(A):
    '''Takes the kronecker product of A with itself'''
    return tfkron(A, A)

@tffunc(2)
def multikron(A, B):
    '''Takes the kronecker product of A and B, iterating over the first index.'''
    return tf.stack([tfkron(a, b) for a, b in zip(tf.unstack(A), tf.unstack(B))])

class OrthogonalMatrix:
    '''Class constructs orthogonal matrix in Tensorflow as product of
    Householder reflections.'''
    def __init__(self, N):
        self._var = tf.Variable(np.random.randn(N, N).astype(dtype))
        self.U = self._var/tf.sqrt(tf.reduce_sum(tf.square(self._var),
                                                 axis=0, keep_dims=True))
        self.T = tf.expand_dims(tf.eye(N, dtype=dtype), 0) -\
                 2.*tf.einsum('ik,jk->kij', self.U, self.U)
        self.matrix = self.T[0]
        for t in tf.unstack(self.T[1:]):
            self.matrix = tf.matmul(self.matrix, t)
        # avoids While, equiv. to:
        # self.matrix = tf.foldl(tf.matmul, self.T)

@tffunc(1)
def entropy(P):
    return -tf.reduce_sum(P*tf.log(P))

@tffunc(2)
def inner_broadcast(density, core):
    '''compute M_k=A_k^T*L*A_k'''
    return tf.einsum('krs,su,kut->krt', tf.transpose(core, [0,2,1]), density, core)

@tffunc(2)
def inner_contraction(density, core, weights = None):
    '''compute Sum_k w_k*A_k^T*L*A_k'''
    if weights is not None:
        return tf.reduce_sum(tf.reshape(weights, (-1, 1, 1)) * inner_broadcast(density, core), axis=0)
    else:
        return tf.einsum('krs,su,kut', tf.transpose(core, [0,2,1]), density, core)


class MPS:
    """
        Instances represent discrete distributions over N discrete random variables
        taking values in a finite set of K elements. The distribution is implicitly defined
        in terms of a K x ... x K (N times) tensor, which is represented as a squared
        matrix product state (equiv. tensor train).

        The representation guarantees positivity and allows for efficient computation of
        the normalization constant, conditionals, and marginals. Based on this it's also
        easy to sample from the model using an ancestral sampler.

        N - Integer. Number of tensor axes.
        K - Integer. Dimensionality along each axis.
        ranks - Iterable of integers. Rank of matrices. The matrices in the i'th core have
            rank ranks[i] x ranks[i+1].
        cores (None) - Object of type Core. Its .cores field should be an
            iterable of order-3 Tensorflow tensors of length N.
            .cores[i] should have shape K x ranks[i] x ranks[i+1].
        normalized (True) - Boolean. If true, the tensor is normalized to 1.
    """
    def __init__(self, N, K, ranks, cores=None, normalized=True, multi_temp=False):
        self.N  = N
        self.K = K
        self.ranks = ranks
        with tf.name_scope("auxiliary"):
            if cores:
                self.raw = cores
            else:
                self.raw = Core(N, K, ranks, cores)


        self.cores = self.raw.scaledcores((1./tf.sqrt(self._scale()))
                                              if normalized else tf.convert_to_tensor(1., dtype=dtype))
        self.right_canonical = self.raw.right_canonical
        self.left_canonical = self.raw.left_canonical

        self._nuvar = tf.Variable(np.log(np.exp(1.)-1.), dtype=dtype)
        self.nu = tf.nn.softplus(self._nuvar)
        if multi_temp:
            self._tempvar = tf.Variable(np.log(np.exp(0.1)-1.)*np.ones(N), dtype=dtype)
            self.temperatures = tf.nn.softplus(self._tempvar)
        else:
            self._tempvar = tf.Variable(np.log(np.exp(0.1)-1.), dtype=dtype)
            self.temperatures = tf.nn.softplus(self._tempvar)*tf.ones(N, dtype=dtype)
        self.softgate = lambda z: tf.nn.softmax(z/self.temperature, dim=-1) #scaled softmax

        with tf.name_scope("marginalization"):
            with tf.name_scope("auxiliary"):
                flip_cores = [tf.transpose(core, [0,2,1])
                              for core in self.cores[-1::-1]]
                initial = tf.ones((1., 1.), dtype=dtype)
                inner_marg = [tf.einsum('kri,krj->ij',
                                        self.cores[0], self.cores[0])]
                for core in self.cores[1:-1]:
                    inner_marg += [inner_contraction(inner_marg[-1], core)]

                outer_marg = [tf.einsum('kir,kjr->ij',
                                        self.cores[-1], self.cores[-1])]
                for core in flip_cores[1:-1]:
                    outer_marg += [inner_contraction(outer_marg[-1], core)]

            #add boundary elements (1-vectors) and remove full products
            self.inner_marginal = [initial, *inner_marg]
            self.outer_marginal = [*outer_marg[-1::-1], initial]

    @tfmethod(1)
    def contraction(self, Z, normalized=True):
        """
        Compute Sum_I (Prod_n z_{n, I(n)}) T_I where I ranges over
        all indices of the tensor.
        """
        if normalized:
            cores = self.cores
        else:
            cores = self.raw.cores
        S = tf.ones((1, 1), dtype=dtype)
        for core, z in zip(cores, tf.unstack(Z)):
            S = inner_contraction(S, core, z)
        return tf.reduce_sum(S)

    @tfmethod(0)
    def buildcontrol(self, f, nsamples=1):
        '''Runs ancestral sampling routine on MPS and auxiliary shadow model
        and calculates necessary reparameterized quantities for a
        RELAX estimator.
        '''
        loss = 0. # f(b)
        control = 0. # f(shadowb)
        conditional_control = 0. # f(conditionalshadowb)
        logp = 0. # tf.log(self.contraction(b))
        for it in range(nsamples):
            b, shadowb, conditionalshadowb = self.sample(doshadowsample=True)
            loss += f(b)
            control += f(shadowb)
            conditional_control += f(conditionalshadowb)
            logp += tf.log(self.contraction(b))

        return (loss/nsamples, self.nu*control/nsamples,
                self.nu*conditional_control/nsamples, logp/nsamples)

    @tfmethod(0)
    def softsample(self):
        """Produce a single NxK sample from the shadow MPS, defined as the
        implicit generative model where sample 1 is drawn from the concrete
        relaxation of the marginal, and sample 2 (and so on) is drawn from
        the concrete relaxation of q(x2|x1), where the conditioning is soft:
        if p(x2=k|x1=i)=Tr[G_k^T*L_i*G_k*R] is the true conditional (with R
        containing the marginalization information), softsample conditions on
        Lhat=sum L_i*x1[i] so that the conditioning is correct if x1 is
        concentrated on one value.

        Returns:
            Z: NxK tensor. A sample from the shadow MPS.
        """

        shadowcondition = tf.ones((1, 1), dtype=dtype)
        shadowsamples = []
        sequence = zip(self.cores, self.outer_marginal)

        for index, (core, marginal) in enumerate(sequence):
            if self.right_canonical:
                shadowdistribution = tf.trace(
                    inner_broadcast(shadowcondition, core))
            else:
                shadowdistribution = tf.einsum(
                    'kij,ji', inner_broadcast(shadowcondition, core),
                    marginal)

            shadowreparam = CategoricalReparam(
                tf.expand_dims(tf.log(shadowdistribution), 0),
                temperature=self.temperatures[index])

            shadowsample = shadowreparam.gatedz
            shadowsamples += [shadowsample]

            shadowupdate = tf.einsum('kij,k', core,
                                     tf.squeeze(shadowsample))
            shadowcondition = tf.einsum('ik,kl,lj',
                                        tf.transpose(shadowupdate),
                                        shadowcondition, shadowupdate)

        shadowb = tf.concat(shadowsamples, axis=0)

        return shadowb

    @tfmethod(0)
    def sample(self, doshadowsample=False, coupled=False):
        '''Runs ancestral sampling routine and calculates necessary
        reparameterized quantities for a REBAR estimator.
        See shadowsample() for more info on shadow MPS.

        Args:
            shadowsample (defaults to False): Boolean. If False, return a
                sample from MPS, if True, return a tuple of samples from MPS
                and the shadow MPS.
        Returns:
            b: NxK tensor. A sample from the MPS.
            shadowb (if doshadowsample==True)): NxK tensor. A sample from the
                relaxed MPS where samples are continuous and draws from a
                concrete distribution, and conditioning is produced by
                averaging over the core relative to the sample.
                Generating noise tied to the noise producing b.
            conditionalshadowb (if doshadowsample==True): Same as shadowb, except
                conditioned on the vaulue of b being observed.
        '''
        condition = tf.ones((1, 1), dtype=dtype)
        samples = []
        sequence = zip(self.cores, self.outer_marginal)

        if doshadowsample:
            shadowcondition = tf.ones((1, 1), dtype=dtype)
            shadowsamples = []
            conditionalshadowsamples = []

        for index, (core, marginal) in enumerate(sequence):
            if self.right_canonical:
                distribution = tf.trace(inner_broadcast(condition, core))
            else:
                distribution = tf.einsum('kij,ji', inner_broadcast(condition,
                                                                   core),
                                         marginal)
            reparam = CategoricalReparam(
                tf.expand_dims(tf.log(distribution) -
                               tf.reduce_logsumexp(distribution), 0),
                coupled=coupled, temperature=self.temperatures[index])

            sample = reparam.b
            samples += [sample]

            update = tf.einsum('kij,k', core, tf.squeeze(sample))
            condition = tf.einsum('ik,kl,lj', tf.transpose(update),
                                  condition, update)

            if doshadowsample:
                if self.right_canonical:
                    shadowdistribution = tf.trace(
                        inner_broadcast(shadowcondition, core))
                else:
                    shadowdistribution = tf.einsum(
                        'kij,ji', inner_broadcast(shadowcondition, core),
                        marginal)

                shadowreparam = CategoricalReparam(
                    tf.expand_dims(tf.log(shadowdistribution) -
                                   tf.reduce_logsumexp(shadowdistribution), 0),
                    noise=reparam.u, cond_noise=reparam.v,
                    temperature=self.temperatures[index])

                shadowsample = shadowreparam.gatedz
                shadowsamples += [shadowsample]

                #CHANGE: CRITICAL MODIFICATION
                zb = reparam.zb
                sb = zb + shadowreparam.param - reparam.param
                conditionalshadowsample = shadowreparam.softgate(
                    sb, shadowreparam.temperature)
                conditionalshadowsamples += [conditionalshadowsample]

                shadowupdate = tf.einsum('kij,k', core,
                                         tf.squeeze(shadowsample))
                shadowcondition = tf.einsum('ik,kl,lj',
                                            tf.transpose(shadowupdate),
                                            shadowcondition, shadowupdate)

        b = tf.concat(samples, axis=0)
        if doshadowsample:
            shadowb = tf.concat(shadowsamples, axis=0)
            conditionalshadowb = tf.concat(conditionalshadowsamples, axis=0)

        return (b, shadowb, conditionalshadowb) if doshadowsample else b

    @tfmethod(1)
    def gibbsconditionals(self, Z, logprob=True, normalized=True):
        #cores = [tf.einsum('kij,k', core, tf.squeeze(z))
        #         for z, core in zip(tf.unstack(Z), self.cores)]
        inner_condition = tf.ones((1,1), dtype=dtype)
        inner_conditions = [inner_condition]
        for z, kcore in zip(tf.unstack(Z), self.cores):
            core = tf.einsum('kij,k', kcore, tf.squeeze(z))
            inner_condition = tf.einsum('ik,kl,lj', tf.transpose(core),
                                  inner_condition, core)
            inner_conditions.append(inner_condition)

        outer_condition = tf.ones((1,1), dtype=dtype)
        conditionals = []
        for z, kcore, inner_condition in reversed(list(zip(tf.unstack(Z), self.cores, inner_conditions[:-1]))):
            conditional = inner_broadcast(inner_condition, kcore)
            conditionals.append(tf.einsum('kij,ji',conditional, outer_condition))
            outer_condition = inner_contraction(outer_condition, tf.transpose(kcore, [0, 2, 1]), tf.squeeze(z))
        conditionals = tf.stack(conditionals[-1::-1])
        if logprob:
            conditionals = tf.log(conditionals)
            if normalized:
                conditionals = conditionals - tf.reduce_logsumexp(conditionals,
                                                                  axis=-1,
                                                                  keep_dims=True)
        else:
            if normalized:
                condititionals = conditionals/tf.reduce_sum(conditionals,
                                                            axis=-1,
                                                            keep_dims=True)
        return conditionals

    def collocation(self):
        transfers = [multikron(core, core) for core in self.cores]
        marginals = [tf.reduce_sum(transfer, axis=0) for transfer in transfers]
        A = []#tf.zeros((self.N, self.N), dtype=dtype)
        for i,j in np.ndindex((self.N, self.N)):
            if i == j:
                a = 1.
            else:
                a = 0.
                for k in range(self.K):
                    factors = [(marginals[l] if (l!=i and l!=j) else transfers[l][k]) for l in range(self.N)]
                    x = factors[0]
                    for factor in factors[1:]:
                        x = tf.matmul(x, factor)
                    a += x
            A.append(a)
            # A[i,j] = self.K*tf.foldl(tf.matmul, factors)
        return tf.reshape(tf.stack(A), (self.N, self.N))

    @tfmethod(0)
    def marginals(self):
        return tf.stack([tf.einsum('kiu,ur,krs,si->k',
                                   tf.transpose(core,[0, 2, 1]),
                                   inner_marg, core, outer_marg)
                         for core, inner_marg, outer_marg in
                         zip(self.cores,
                             self.inner_marginal,
                             self.outer_marginal)])

    @tfmethod(0)
    def marginalentropy(self):
        '''calculate entropy of marginals'''
        marginals = self.marginals()
        return entropy(marginals)

    @tfmethod(0)
    def populatetensor(self):
        '''Convert MPS tensor to a dense format.

        Returns:
            Z - tensor of order N with each axis having length K.
        '''
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
        return tf.real(Z)

    def _scale(self):
            return tf.identity(self.contraction(tf.ones((self.N, self.K),
                                                        dtype=dtype),
                                                normalized=False),
                               name="scale")

    def var_params(self):
        return [self._tempvar, self._nuvar]

def symmetrynorm(cores):
    '''
    Computes a proxy error measure for tensor relabelling symmetry.
    Computes two relative errors between a squared MPS and
    the same squared MPS with
        1) the first and second dimension swapped along each axis.
        2) all dimensions along each axis permuted cyclically so that
            dimension permutation(i)=i+1 (mod K).
    The errors are calculated under the n-dimensional Frobenius norm.

    Returns:
        E_swap - Scalar. Relative error in norm to the swapped version.
        E_cycle - Scalar. Relative error in norm to the cycled version.

    '''
    swap = lambda X: tf.concat([X[None, 1],
                                X[None, 0],
                                X[2:]], axis=0)
    swap_cores = [swap(core) for core in cores]

    cycle = lambda X: tf.concat([X[None, -1],
                                 X[:-1]], axis=0)
    cycle_cores = [cycle(core) for core in cores]
    transfer = [multikron(core, core) for core in cores]
    transfer_swap = [multikron(core, swap_core) for core, swap_core in
                     zip(cores, swap_cores)]
    transfer_cycle = [multikron(core, cycle_core) for core, cycle_core in
                      zip(cores, cycle_cores)]

    S = tf.ones((1,1), dtype=dtype)
    Sswap = tf.ones((1,1), dtype=dtype)
    Scycle = tf.ones((1,1), dtype=dtype)
    for core, score, ccore in zip(transfer, transfer_swap, transfer_cycle):
        S = inner_contraction(S, core)
        Sswap = inner_contraction(Sswap, score)
        Scycle = inner_contraction(Scycle, ccore)
    return tf.real((S - Sswap)/(S)), tf.real((S-Scycle)/(S))

def bruteforce_populate(tensor_eval, N, K):
    '''Populate high-order tensor by looping over indices'''
    Z = np.zeros([K,]*N).astype(dtype)
    codes = np.eye(K, dtype=dtype)
    for index in np.ndindex(*Z.shape):
        Z[index] = tensor_eval(codes[list(index)])

class Core:
    '''Wrapper around list of order-3 tensors with compatible shapes
    such that they can be used in an MPS.'''
    def __init__(self, N, K, ranks, cores=None):
        self.N  = N
        self.K = K
        self.ranks = ranks
        self.right_canonical = False
        self.left_canonical = False
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

class Canonical(Core):
    '''
    A set of cores with the constraint that for each core either
    Sum_k U_k^T*U_k = I (left-canonical) or Sum_k U_k*U_k^T = I (right-canonical).

    If used in an MPS, this means that either the inner_marginal or outer_marginal
    matrices will be trivially equal to the identity,
    '''
    def __init__(self, N, K, ranks, left=False):
        '''
            Constructs canonical core set.

            Input:
                N - Integer. Number of cores.
                K - Integer. Number of matrices in each core.
                ranks - Iterable of integers. Rank of matrices.
                    The matrices in the i'th core have
                    rank ranks[i] x ranks[i+1].
                left (True) - Boolean. If True, then the Core is
                    left-canonical. If False, it's right-canonical.

        '''
        self.N  = N
        self.K = K
        self.ranks = ranks
        self.right_canonical = ~left
        self.left_canonical = left

        assert(len(ranks) == N+1)
        assert(ranks[0] == 1 & ranks[-1] == 1)
        restack = lambda A, rank0, rank1: \
                      tf.transpose(tf.reshape(A, (rank0, self.K, rank1)),
                                   [1,0,2])

        if left:
            orthogonal_rank = 0
            shape_factor = (self.K, 1)
        else:
            orthogonal_rank = 1
            shape_factor = (1, self.K)

        with tf.name_scope("orthogonal"):
            #orthogonal matrices
            self._U = [OrthogonalMatrix(self.K*ranks[orthogonal_rank]) for ranks
                      in zip(self.ranks[:-1], self.ranks[1:])]
            self.U = self._U
        with tf.name_scope("cores"):
            #set of canonical cores
            self.cores = [restack(u.matrix[0:shape_factor[0]*rank0,
                                           0:shape_factor[1]*rank1], rank0, rank1)
                          for u, rank0, rank1 in zip(self.U, self.ranks[:-1],
                                                     self.ranks[1:])]

    def params(self):
        return [u._var for u in self._U]

    def copycore_op(self, core):
        ops = [u1._var.assign(u2._var) for u1, u2 in zip(self._U, core._U)]
        return tf.group(*ops)

def rootofunity(K):
    angle = 2.*np.pi/K
    return np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle),np.cos(angle)]])



class SwapInvariant(Core):
    def __init__(self, N, K, ranks):
        self.N  = N
        self.K = K
        self.ranks = ranks
        assert(len(ranks) == N+1)
        assert(ranks[0] == 1 & ranks[-1] == 1)
        restack = lambda A, rank0, rank1: \
                      tf.transpose(tf.reshape(A, (rank0, self.K, rank1)),
                                   [1,0,2])

        if left:
            orthogonal_rank = 0
            shape_factor = (self.K, 1)
        else:
            orthogonal_rank = 1
            shape_factor = (1, self.K)

        #orthogonal matrices
        self._U = [OrthogonalMatrix(self.K*ranks[orthogonal_rank]) for ranks
                  in zip(self.ranks[:-1], self.ranks[1:])]
        self.U = self._U
        #set of canonical cores
        self.cores = [restack(u.matrix[0:shape_factor[0]*rank0,
                                       0:shape_factor[1]*rank1], rank0, rank1)
                      for u, rank0, rank1 in zip(self.U, self.ranks[:-1],
                                                 self.ranks[1:])]


class LeftUniformCanonical(Canonical):
    def __init__(self, N, K, ranks):
        self.N  = N
        self.K = K
        self.ranks = ranks

        self._U = [[OrthogonalMatrix(max(rank0, rank1)) for _ in range(self.K)] for rank0, rank1 in zip(self.ranks[:-1], self.ranks[1:])]
        self.G = [[ui.matrix[:rank0, :rank1]/np.sqrt(self.K) for ui in u] for u, rank0, rank1 in zip(self._U, self.ranks[:-1], self.ranks[1:])]
        self.U = [tf.concat(g, axis=0) for g in self.G]
        self.cores = [tf.concat([tf.expand_dims(gi, 0) for gi in g], axis=0) for g in self.G]

    def params(self):
        return [ui._var for u in self._U for ui in u]


if __name__ is "__main__":
    sess = tf.InteractiveSession()
    N = 6
    K = 2
    R = K**N
    ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))

    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.reduce_sum(z) - tf.reduce_sum(z*true_index)

    C = np.array([[0,1,0],[0,0,1],[1,0,0]])
    orth=Canonical(N, K, ranks)
    orth_model = MPS(N, K, ranks, cores=orth)

    #optimization routine for marginal entropy
    opt = tf.contrib.opt.ScipyOptimizerInterface(-orth_model.marginalentropy(), orth.params(), tol=1e-10,method='CG')

    #"symmetry norm"
    symf = tf.reduce_sum(symmetrynorm(orth_model.cores))

    #optimization routine for symmetry norm
    sopt = tf.contrib.opt.ScipyOptimizerInterface(symf, orth.params(), tol=1e-10,method='CG')

    sess.run(tf.global_variables_initializer())
    opt.minimize()

    def estvar(samples):
        mean = sum(samples)
        return sum([np.square(sample-mean) for sample in samples])/(len(samples)-1)

    ns = 10000
    cost = lambda x: -tf.log(orth_model.contraction(x))
    grads, _, _ = RELAX(*orth_model.buildcontrol(cost), orth.params())
    print("RELAX:")
    print(np.sqrt(estvar([grads[0][0].eval() for _ in range(ns)])))
    sess.run(orth_model._nuvar.assign(-100))
    print("UnRELAXed:")
    print(np.sqrt(estvar([grads[0][0].eval() for _ in range(ns)])))
    #quantities of potential interest
#    G = orth.cores[1].eval()
#    g0 = np.squeeze(orth.cores[0].eval())
#    L = orth_model.inner_marginal[1].eval()
#    O = orth.U[1].matrix.eval()
