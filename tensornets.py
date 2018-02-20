import numpy as np
import tensorflow as tf
from relaxflow.reparam import CategoricalReparam, categorical_forward, categorical_backward
dtype = 'float64'

from tfutils import tffunc, tfmethod, HouseholderChain
select_max = lambda z, K: tf.one_hot(tf.argmax(z, axis=-1), K, dtype=dtype)
bitify = lambda x, K: np.sum(K**np.arange(len(x))*x)

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
        A[bitify(coord1, K), bitify(coord2, K)] = np.log(np.maximum(Z[ind], 0.))
    return A

def full2TT(A, maxrank=np.inf):
    Z = A
    k = A.shape[0]
    ndim = A.ndim
    G = []
    K = []
    rprev = 1
    ranks = [1]
    for d in range(1,ndim+1):
        U, S, V = np.linalg.svd(Z.reshape([rprev*k, k**(ndim-d)]), full_matrices=0)
        #print(S.min())
        r = np.min([np.sum(~np.isclose(S,0.,1e-10)), maxrank]).astype('int16')
        Z = S[:r,None]*(V[:r, :])
        #r = S.size
        ranks += [r]
        G += [U[:, :r].reshape(rprev,k,r)]
        K += [U[:, :r]]
        rprev = r
    G[-1] *= Z
    cores = Core(ndim, k, ranks, cores=[tf.transpose(g,[1,0,2]) for g in G])
    return cores, MPS(ndim, k, ranks, cores=cores, normalized=False)

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
    def __init__(self, N, eye=False):
        self.N = N
        if eye:
            initial = np.eye(N, dtype=dtype)
        else:
            initial = np.random.randn(N, N).astype(dtype)
        self._var = tf.Variable(initial)
        self.V = self._var/tf.sqrt(tf.reduce_sum(tf.square(self._var),
                                                 axis=1, keep_dims=True))
        self.neg_matrix = HouseholderChain(self.V)

    def dot(self, A, left_product=True):
        if left_product:
            return -self.neg_matrix.dot(A)
        else:
            return -tf.transpose(self.neg_matrix(tf.transpose(A)))

    def dense(self):
        return self.dot(tf.eye(self.N, dtype=dtype))

class CayleyOrthogonal:
    def __init__(self, N):
        self.N = N
        self._var = tf.Variable(np.random.randn(N, N).astype(dtype))
        self.triu = tf.matrix_band_part(self._var, 0, -1)
        self.skew = self.triu-tf.transpose(self.triu)
        I = tf.eye(N, dtype=dtype)
        self.matrix = tf.matrix_solve(I + self.skew,
                                      I - self.skew)

    def dot(self, A, left_product=True):
        if left_product:
            return tf.matmul(self.matrix, A)
        else:
            return tf.transpose(tf.matmul(self.matrix, A, transpose_b=True))


    def dense(self):
        return self.matrix

@tffunc(1)
def entropy(P):
    return -tf.reduce_sum(P*tf.log(1e-16+P))

@tffunc(2)
def inner_broadcast(density, core, opt_einsum=False):
    '''compute M_k=A_k^T*L*A_k'''
    if opt_einsum:
        M  = tf.einsum('su, ksr->kur', density, core)
        return tf.einsum('kurb,kut->krt', M, core)
    else:
        return tf.einsum('krs,su,kut->krt', tf.transpose(core, [0,2,1]), density, core)

@tffunc(2)
def batch_inner_broadcast(density, core, opt_einsum=False):
    '''compute M_k=A_k^T*L*A_k'''
    if opt_einsum:
        M  = tf.einsum('bsu,ksr->kurb', density, core)
        return tf.einsum('kurb,kut->bkrt', M, core)
    else:
        return tf.einsum('krs,bsu,kut->bkrt', tf.transpose(core, [0,2,1]), density, core)


@tffunc(2)
def inner_contraction(density, core, weights = None, opt_einsum=False):
    '''compute Sum_k w_k*A_k^T*L*A_k'''
    if opt_einsum:
        if weights is not None:
            M = tf.einsum('su,ksr->kur', density, core)
            M = tf.einsum('kur,kut->krt', M, core)
            return tf.einsum('krt,k->rt', M, weights)
        else:
            M = tf.einsum('su,ksr->kur', density, core)
            return tf.einsum('kur,kut->rt', M, core)
    else:
        if weights is not None:
            return tf.reduce_sum(tf.reshape(weights, (-1, 1, 1)) * inner_broadcast(density, core), axis=0)
        else:
            return tf.einsum('krs,su,kut', tf.transpose(core, [0,2,1]), density, core)


@tffunc(2)
def batch_inner_contraction(density, core, weights = None, opt_einsum=False):
    '''compute Sum_k w_k*A_k^T*L*A_k'''
    if opt_einsum:
        if weights is not None:
            M = tf.einsum('bsu,ksr->kurb', density, core)
            M = tf.einsum('kurb,kut->krtb', M, core)
            return tf.einsum('krtb,bk->brt', M, weights)
        else:
            M = tf.einsum('kut,ksr->surt', core, core)
            return tf.einsum('surt,bsu->brt', M, density)
    else:
        if weights is not None:
            return tf.einsum('bkij,bk->bij', batch_inner_broadcast(density, core), weights)
        else:
            return tf.einsum('krs,bsu,kut', tf.transpose(core, [0,2,1]), density, core)

def packmps(name, mps, sess=None):
    mps_metadata = {
                'N': mps.N,
                'K': mps.K,
                'ranks': mps.ranks,
                'normalized': mps.normalized,
                'multi_temp': mps.multi_temp
                }
    core_type = mps.raw.__class__
    core_metadata = {
                     'N': mps.N,
                     'K': mps.K,
                     'ranks': mps.ranks,
                     }
    if core_type is Canonical:
        core_metadata.update({
                              'left': mps.raw.left_canonical,
                              'initials': mps.raw.initials,
                              'orthogonalstyle': mps.raw.orthogonalstyle
                              })
    elif (core_type is CanonicalPermutationCore or
          core_type is CanonicalPermutationCore2):
        core_metadata.update({'orthogonalstyle': mps.raw.orthogonalstyle})


    #saver = tf.train.Saver(mps.raw.params(), max_to_keep=None)
    basic_metadata = {'core_type': core_type, 'name': name}#, 'save_path': saver.save(sess, folder + name + '.ckpt')}
    if sess is None:
        sess = tf.get_default_session()
    hardcopy = [sess.run(param) for param in mps.raw.params()]

    metadata = {'basic': basic_metadata, 'hardcopy': hardcopy,
                'core': core_metadata, 'mps': mps_metadata}
    return metadata
    
def dictpack(name, dictionary, folder='', sess=None):
    new_d = {key: packmps(name+'_key{}'.format(key), value) for key, value in dictionary.items()} 
    with open(folder + name+'.pkl', 'wb') as handle:
        pickle.dump(new_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def unpackmps(metadata):
    cores = metadata['basic']['core_type'](**metadata['core'])
    mps = tn.MPS(**metadata['mps'], cores=cores)
    mass_assign = ([tf.assign(var, value)
                    for var, value in
                    zip(cores.params(), metadata['hardcopy'])] +
                    [tf.assign(var, value)
                    for var, value in
                    zip(cores.params(), metadata['hardcopy'])])
    if sess is None:
        sess = tf.get_default_session()
    sess.run(mass_assign)
    return (mps, cores, mass_assign, metadata)

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

        self.right_canonical = self.raw.right_canonical
        self.left_canonical = self.raw.left_canonical


        if self.right_canonical or self.left_canonical:
            self.cores = self.raw.cores
        else:
            self.cores = self.raw.scaledcores((1./tf.sqrt(self._scale()))
                                                if normalized else
                                                tf.convert_to_tensor(1., dtype=dtype))


        self._nuvar = tf.Variable(np.log(np.exp(1.)-1.), dtype=dtype)
        self.nu = tf.nn.softplus(self._nuvar)
        if multi_temp:
            self._tempvar = tf.Variable(np.log(np.exp(0.5)-1.)*np.ones(N), dtype=dtype)
            self.temperatures = tf.nn.softplus(self._tempvar)
        else:
            self._tempvar = tf.Variable(np.log(np.exp(0.5)-1.), dtype=dtype)
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
        return tf.squeeze(S)

    @tfmethod(1)
    def batch_contraction(self, Z, normalized=True):
        if normalized:
            cores = self.cores
        else:
            cores = self.raw.cores
        batches = Z.shape[0]
        S = tf.ones((batches, 1, 1), dtype=dtype)

        for core, z in zip(cores, tf.unstack(Z, axis=1)):
            S = batch_inner_contraction(S, core, z)
        return tf.squeeze(S)

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
    def softsample(self, nsamples=1):
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

        shadowcondition = tf.ones((nsamples, 1, 1), dtype=dtype)
        shadowsamples = []
        if self.left_canonical:
            sequence = zip([tf.transpose(c, [0,2,1]) for c in self.cores[-1::-1]],
                           self.inner_marginal[-1::-1])
        else:
            sequence = zip(self.cores, self.outer_marginal)

        for index, (core, marginal) in enumerate(sequence):
            if self.right_canonical or self.left_canonical:
                shadowdistribution = tf.trace(
                    batch_inner_broadcast(shadowcondition, core))
            else:
                shadowdistribution = tf.einsum(
                    'bkij,ji', batch_inner_broadcast(shadowcondition, core),
                    marginal)

            shadowreparam = CategoricalReparam(
                tf.log(1e-16+shadowdistribution),
                temperature=self.temperatures[index])

            shadowsample = shadowreparam.gatedz
            shadowsamples += [shadowsample]

            shadowupdate = tf.einsum('kij,bk', core,
                                     shadowsample)
            shadowcondition = tf.einsum('bik,bkl,blj->bij',
                                        tf.transpose(shadowupdate, [0,2,1]),
                                        shadowcondition, shadowupdate)
        if self.left_canonical:
            shadowsamples = shadowsamples[-1::-1]
        shadowb = tf.transpose(tf.stack(shadowsamples), [1,0,2])
        return tf.squeeze(shadowb)

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

    def collocation(self, nsamples=100000):
        if nsamples > 0:
            z = self.softsample(nsamples)
            return tf.einsum('bik,bjk', z, z)/nsamples
        else:
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
    def marginals(self, uniform=False):
        if uniform:
            return tf.ones((self.N, self.K))/self.K
        if self.left_canonical:
            return tf.stack([tf.einsum('kir,krs,si->k',
                                       tf.transpose(core,[0, 2, 1]),
                                       core, outer_marg)
                             for core, outer_marg in
                             zip(self.cores,
                                 self.outer_marginal)])
        elif self.right_canonical:
            return tf.stack([tf.einsum('kiu,ur,kri->k',
                                       tf.transpose(core,[0, 2, 1]),
                                       inner_marg, core)
                             for core, inner_marg in
                             zip(self.cores,
                                 self.inner_marginal)])
        else:
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
    def elbo(self, f, nsamples=1, fold=False, marginal=True, invtemp=1., cvweight=1., report=False):
        '''calculate ELBO or another entropy-weighted expectation using nsamples MC samples'''
        samples = self.softsample(nsamples)
        if fold:
            llk = tf.map_fn(f, samples)
        else:
            llk = f(samples)
        if marginal:
            marginals = self.marginals()
            marginalentropy = -tf.reduce_sum(marginals * tf.log(1e-16+marginals))
            marginalcv = (marginalentropy +
                          tf.reduce_sum(samples *
                                        tf.log(1e-16+marginals)[None, :, :],
                                        axis=[1, 2]))
        else:
            marginalcv = 0.
        entropy = -tf.log(1e-16+self.batch_contraction(samples))
        elbo = llk + entropy + cvweight*marginalcv
        objective = llk + invtemp*(entropy + cvweight*marginalcv)
        if report:
            return (objective, elbo, llk, entropy, marginalentropy, marginalcv)
        else:
            return elbo
    #def totalcorrelation(self, nsamples=5):
    #    sample =
    #    return tf.log(self.contraction())

    @tfmethod(0)
    def pred(self, f, nsamples=1, fold=False):
        '''calculate expectation of function f over nsamples samples from model'''
        samples = self.softsample(nsamples)
        if fold:
            llk = tf.map_fn(f, samples)
        else:
            llk = f(samples)
        return llk

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

def inner_product(mps1, mps2):
    cores1 = mps1.cores
    cores2 = mps2.cores
    transfer = [multikron(core1, core2)
                  for core1, core2 in zip(cores1, cores2)]
    inprod = tf.ones((1,1), dtype=dtype)
    for core in zip(transfer):
        inprod = inner_contraction(inprod, core)
    return inprod

def norm(mps1, mps2):
    cores1 = mps1.cores
    cores2 = mps2.cores
    transfer1 = [multikron(core, core) for core in cores1]
    transfer2 = [multikron(core, core) for core in cores2]
    transfer12 = [multikron(core1, core2)
                  for core1, core2 in zip(cores1, cores2)]
    normmps1 = tf.ones((1,1), dtype=dtype)
    normmps2 = tf.ones((1,1), dtype=dtype)
    normmps12 = tf.ones((1,1), dtype=dtype)
    for core1, core2, core12 in zip(transfer1, transfer2, transfer12):
        normmps1 = inner_contraction(normmps1, core1)
        normmps2 = inner_contraction(normmps2, core2)
        normmps12 = inner_contraction(normmps12, core12)
    return normmps1 + normmps2 - 2.*normmps12

def expectation(mps, rank1):
    return mps.batch_contraction(rank1)



def norm_rank1(mps, rank1):
    cores = mps.cores
    transfer = [multikron(core, core) for core in cores]
    N  = rank1.shape[0].value

    normr1 = tf.einsum('kni,lni->', rank1, rank1)
    innerproducts = mps.batch_contraction(rank1)

    normmps = tf.ones((1,1), dtype=dtype)
    for core in transfer:
        normmps = inner_contraction(normmps, core)

    return normmps + tf.reduce_sum(normr1)/N**2. - 2.*tf.reduce_sum(innerproducts)/N

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
    for index in tqdm.tqdm(np.ndindex(*Z.shape)):
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
    def __init__(self, N, K, ranks, left=False, initials=None, orthogonalstyle=OrthogonalMatrix):
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
        self.right_canonical = not left
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

        if initials is None:
            self.initials = []
            for ranks in zip(self.ranks[:-1], self.ranks[1:]):
                I = np.eye(self.K*ranks[orthogonal_rank])
                self.initials.append(I[:shape_factor[0]*ranks[0],
                                       :shape_factor[1]*ranks[1]])
        else:
            for initial, rank0, rank1 in zip(self.initials, self.ranks[:-1],
                                            self.ranks[1:]):
                assert(initial.shape == (shape_factor[0]*rank0,
                                         shape_factor[1]*rank1))
            self.initials = initials

        with tf.name_scope("orthogonal"):
            #orthogonal matrices
            self.U = [orthogonalstyle(self.K*ranks[orthogonal_rank]) for ranks
                      in zip(self.ranks[:-1], self.ranks[1:])]
        with tf.name_scope("cores"):
            #set of canonical cores
            self.cores = []
            for initial, u, rank0, rank1 in zip(self.initials, self.U,
                                                self.ranks[:-1],
                                                self.ranks[1:]):
                try:
                    C = u.dot(initial)
                except ValueError:
                    C = tf.transpose(u.dot(tf.transpose(initial)))
                self.cores.append(restack(C, rank0, rank1))

    @staticmethod
    def restack(A, rank0, rank1):
        '''transforms orthogonal matrix of shape K*r0 x r1 into stack of shape r0 x K x r1.'''
        return tf.transpose(tf.reshape(A, (rank0, self.K, rank1)),
                            [1,0,2])

    def params(self):
        return [u._var for u in self.U]

    def copycore_op(self, core):
        ops = [u1._var.assign(u2._var) for u1, u2 in zip(self.U, core.U)]
        return tf.group(*ops)

def rootofunity(K):
    angle = 2.*np.pi/K
    return np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle),np.cos(angle)]])

class PermutationCore(Core):
    def __init__(self, N, K, ranks, projection=True):
        self.N = N
        self.K = K
        self.ranks = ranks
        self.left_canonical = False
        self.right_canonical = False
        assert(np.all([np.logical_not(np.mod(rank, self.K)) or rank==1 for rank in self.ranks]))
        self.rep = [np.eye(self.K, dtype=dtype)[np.roll(np.arange(self.K),k)] for k in range(self.K)]
        self.V = np.column_stack([np.eye(self.K, dtype=dtype)[0], np.ones(self.K, dtype=dtype)/np.sqrt(self.K)])

        if projection:
            self.core0 = [tf.Variable(tf.random_normal(ranks, dtype=dtype)) for ranks in zip(self.ranks[:-1], self.ranks[1:])]
            self.proj = self.V.dot(np.linalg.solve(self.V.T.dot(self.V), self.V.T))
            self.repproj = [rep.dot(self.proj) for rep in self.rep]
            self.cores = [tf.stack([tf.matmul(tf.matmul(self.repk(repproj_i,
                                                                  rank0//self.K),
                                                        core0_i),
                                              self.repk(repproj_i, rank1//self.K),
                                              transpose_b=True)
                                    for repproj_i in self.repproj])
                          for core0_i, rank0, rank1 in zip(self.core0,
                                                           self.ranks[:-1],
                                                           self.ranks[1:])]
        else:
            self.core0 = [tf.Variable(tf.random_normal((2*(rank0//self.K) if rank0>1 else 1, 2*(rank1//self.K) if rank1>1 else 1), dtype=dtype)) for rank0, rank1 in zip(self.ranks[:-1], self.ranks[1:])]
            self.cores = [tf.stack([tf.matmul(tf.matmul(self.repk(rep_i.dot(self.V), rank0//self.K),
                                                            core0_i),
                                                  self.repk(rep_i.dot(self.V), rank1//self.K),
                                                  transpose_b=True) for rep_i in self.rep])
                                                  for core0_i, rank0, rank1 in zip(self.core0, self.ranks[:-1], self.ranks[1:])]

    def repk(self, rep, k):
        if not k:
            return tf.convert_to_tensor([[1.]], dtype=dtype)
        else:
          return np.kron(np.eye(k, dtype=dtype), rep)


    def params(self):
        return self.core0

class PermutationCore_augmented(Core):
    def __init__(self, N, K, repranks, ranks, projection=True):
        self.N = N
        self.K = K
        self.repranks = repranks
        self.ranks = ranks
        self.left_canonical = False
        self.right_canonical = False
        assert(np.all([np.logical_not(np.mod(rank, self.K)) or rank==1 for rank in self.repranks]))
        self.rep = [np.eye(self.K, dtype=dtype)[np.roll(np.arange(self.K),k)] for k in range(self.K)]
        self.V = np.column_stack([np.eye(self.K, dtype=dtype)[0], np.ones(self.K, dtype=dtype)/np.sqrt(self.K)])

        if projection:
            self.core0 = [tf.Variable(tf.random_normal(repranks, dtype=dtype)) for repranks in zip(self.repranks[:-1], self.repranks[1:])]
            self.proj = self.V.dot(np.linalg.solve(self.V.T.dot(self.V), self.V.T))
            self.repproj = [rep.dot(self.proj) for rep in self.rep]
            self.cores = [tf.stack([tf.matmul(tf.matmul(self.repk(repproj_i,
                                                                  rank0//self.K),
                                                        core0_i),
                                              self.repk(repproj_i, rank1//self.K),
                                              transpose_b=True)
                                    for repproj_i in self.repproj])
                          for core0_i, rank0, rank1 in zip(self.core0,
                                                           self.repranks[:-1],
                                                           self.repranks[1:])]
        else:
            self.core0 = [tf.Variable(tf.random_normal((2*(rank0//self.K) if rank0>1 else 1, 2*(rank1//self.K) if rank1>1 else 1), dtype=dtype)) for rank0, rank1 in zip(self.repranks[:-1], self.repranks[1:])]
            self.augment = [tf.Variable(tf.random_normal(rank0, rank1)) for rank0, rank1 in zip(self.ranks[:-1], self.ranks[1:])]
            self.repcores = [tf.stack([tf.matmul(tf.matmul(self.repk(rep_i.dot(self.V), rank0//self.K),
                                                            core0_i),
                                                  self.repk(rep_i.dot(self.V), rank1//self.K),
                                                  transpose_b=True) for rep_i in self.rep])
                                                  for core0_i, rank0, rank1 in zip(self.core0, self.repranks[:-1], self.repranks[1:])]
            self.cores = [augment[None, :, :] + tf.pad(repcore, np.column_stack([np.zeros(3), augment.shape-repcore.shape]), 'CONSTANT') for augment, repcore in zip(self.augment, self.repcores)]

    def repk(self, rep, k):
        if not k:
            return tf.convert_to_tensor([[1.]], dtype=dtype)
        else:
          return np.kron(np.eye(k, dtype=dtype), rep)

    def projk(self, rep, k):
        if not k:
            return tf.convert_to_tensor([[1.]], dtype=dtype)
        else:
          return np.kron(np.eye(k, dtype=dtype), rep)


    def params(self):
        return self.core0



class CanonicalPermutationCore(Core):
    def __init__(self, N, K, ranks, orthogonalstyle=CayleyOrthogonal):
        self.N = N
        self.K = K
        self.ranks = ranks
        self.left_canonical = True
        self.right_canonical = False
        assert(np.all([np.logical_not(np.mod(rank, self.K)) or rank==1 for rank in self.ranks]))
        self.rep = [np.eye(self.K, dtype=dtype)[np.roll(np.arange(self.K),k)] for k in range(self.K)]
        self.V = np.column_stack([np.eye(self.K, dtype=dtype)[0], np.ones(self.K, dtype=dtype)/np.sqrt(self.K)])
        self.omega_sqrtinv = (np.sqrt(np.sqrt(self.K)/(np.sqrt(self.K)+1)) *
                              np.ones((2,2)) +
                              np.sqrt(np.sqrt(self.K)/(np.sqrt(self.K)-1)) *
                              (np.eye(2) - np.eye(2)[[1,0]]))/2. #np.matrix  np.linalg.cholesky(self.V.T.dot(self.V))
        self.Uk = [orthogonalstyle(2*(ranks[0])//self.K)
                   for ranks in zip(self.ranks[1:-2], self.ranks[2:-1])]
        self.Uones = [orthogonalstyle((ranks[1])//self.K)
                      for ranks in zip(self.ranks[1:-2], self.ranks[2:-1])]
        self.cap = tf.Variable(tf.ones((2,1), dtype=dtype))
        self.Qk = [u.dense()[:, :rank//self.K] for u, rank in zip(self.Uk, self.ranks[2:-1])]
        self.Qtilde = [(tf.matmul(q, uones.dense())-q)/np.sqrt(self.K) for uk, uones, q, rank in zip(self.Uk, self.Uones, self.Qk, self.ranks[2:-1])]
        self.core0 = ([np.array([[1., -2./np.sqrt(self.K)]])] +
                      [tf.matmul(self.repk(self.omega_sqrtinv, rank//self.K),
                                 tf.concat([q, qtilde], axis=1))
                       for q, qtilde, rank in zip(self.Qk, self.Qtilde,
                                                  self.ranks[1:-2])] +
                     [tf.matmul(self.omega_sqrtinv, self.cap/tf.norm(self.cap))/np.sqrt(self.K)])
        self.cores = [tf.stack([tf.matmul(tf.matmul(self.repk(rep_i.dot(self.V), rank0//self.K),
                                                        core0_i),
                                              self.repk([rep_i.dot(self.V)[:, k] for k in range(2)], rank1//self.K, ordered=True),
                                              transpose_b=True) for rep_i in self.rep])
                                              for core0_i, rank0, rank1 in zip(self.core0, self.ranks[:-1], self.ranks[1:])]

    def repk(self, rep, k, ordered=False):
        if not k:
            return tf.convert_to_tensor([[1.]], dtype=dtype)
        else:
            if ordered:
                return tf.concat([tfkron(tf.eye(k, dtype=dtype), rep_i[:, None])
                                  for rep_i in rep], axis=1)
            else:
                return np.kron(np.eye(k, dtype=dtype), rep)


    def params(self):
        return [u._var for u in self.Uk] + [u._var for u in self.Uones] + [self.cap]

class CanonicalPermutationCore2(Core):
    def __init__(self, N, K, ranks, orthogonalstyle=CayleyOrthogonal):
        self.N = N
        self.K = K
        self.ranks = ranks
        self.left_canonical = True
        self.right_canonical = False
        assert(np.all([np.logical_not(np.mod(rank, self.K)) or rank==1 for rank in self.ranks]))
        self.rep = [np.eye(self.K, dtype=dtype)[np.roll(np.arange(self.K),k)] for k in range(self.K)]
        self.V = np.column_stack([np.eye(self.K, dtype=dtype)[0], np.ones(self.K, dtype=dtype)/np.sqrt(self.K)])
        self.omega_sqrtinv = (np.sqrt(np.sqrt(self.K)/(np.sqrt(self.K)+1)) *
                              np.ones((2,2)) +
                              np.sqrt(np.sqrt(self.K)/(np.sqrt(self.K)-1)) *
                              (np.eye(2) - np.eye(2)[[1,0]]))/2. #np.matrix  np.linalg.cholesky(self.V.T.dot(self.V))

        #parameters
        self.Ubasis = [orthogonalstyle(2*rank0//self.K) for rank0 in ranks[1:-2]]
        self.UH = [orthogonalstyle(rank1//self.K) for rank1 in ranks[2:-1]]
        self.UW = [orthogonalstyle(rank1//self.K) for rank1 in ranks[2:-1]]
        self.UL = [orthogonalstyle(2*rank0//self.K-rank1//self.K) for rank0, rank1 in zip(ranks[1:-2], ranks[2:-1])]
        self.orthdim = [max(0, min(rank1//self.K,
                                   2*rank0//self.K - rank1//self.K))
                        for rank0, rank1 in zip(ranks[1:-2], ranks[2:-1])]

        self.UC = [orthogonalstyle(rank1//self.K - orthdim) for orthdim, rank1 in zip(self.orthdim, ranks[2:-1])]
        self.C = [tf.concat([tf.zeros((orthdim, rank1//self.K - orthdim),
                                      dtype=dtype),
                             uc.dense()], axis=0) for orthdim, rank1, uc in zip(self.orthdim, ranks[2:-1], self.UC)]
        self.s_var = [tf.Variable(tf.zeros(orthdim, dtype=dtype))
                      for orthdim in self.orthdim]


        self.S = [tf.diag(tf.tanh(var)/np.sqrt(self.K)) for var in self.s_var]
        self.Sdual = [tf.diag((tf.sqrt(1.-self.K*tf.square(tf.diag_part(S))))) for S in self.S]
        S_padded = [tf.concat([S, tf.zeros((max(2*rank0//self.K -
                                               rank1//self.K -
                                               S.shape[0].value, 0),
                                           S.shape[0].value), dtype=dtype)], axis=0)
                    for S, rank0, rank1 in zip(self.S,
                                               ranks[1:-2],
                                               ranks[2:-1])]
        Sdual_padded = [tf.concat([Sdual,
                                   tf.zeros((max(rank1//self.K -
                                                 Sdual.shape[0].value, 0),
                                            Sdual.shape[0].value), dtype=dtype)], axis=0)
                        for Sdual, rank0, rank1 in zip(self.Sdual,
                                                       ranks[1:-2],
                                                       ranks[2:-1])]
        self.H = [(w.dot(uh.dot(tf.concat([sdual, c], axis=1)),
                        left_product=False) - tf.eye(w.N, dtype=dtype))/np.sqrt(self.K) for uh, w, sdual, c in
                        zip(self.UH, self.UW, Sdual_padded, self.C)]
        self.Hhat = [w.dot(ul.dot(tf.concat([s, tf.zeros((s.shape[0].value, max(0, w.N - s.shape[0].value)), dtype=dtype)], axis=1)),
                        left_product=False) for ul, w, s, c in
                        zip(self.UL, self.UW, S_padded, self.C)]

        self.cap = tf.Variable(tf.ones((2,1), dtype=dtype))
        self.B = [u.dot(tf.concat([tf.concat([tf.eye(h.shape[0].value, dtype=dtype),
                                              tf.zeros_like(hhat, dtype=dtype)],
                                             axis=0),
                                   tf.concat([h,hhat],axis=0)], axis=1))
                  for u, h, hhat in zip(self.Ubasis, self.H, self.Hhat)]
        self.core0 = ([np.array([[1., -2./np.sqrt(self.K)]])] +
                      [tf.matmul(self.repk(self.omega_sqrtinv, rank//self.K),
                                 b)
                       for b, rank in zip(self.B, self.ranks[1:-2])] +
                      [tf.matmul(self.omega_sqrtinv,
                                 self.cap/tf.norm(self.cap))/np.sqrt(self.K)])
        self.cores = [tf.stack([tf.matmul(tf.matmul(self.repk(rep_i.dot(self.V), rank0//self.K),
                                                        core0_i),
                                              self.repk([rep_i.dot(self.V)[:, k] for k in range(2)], rank1//self.K, ordered=True),
                                              transpose_b=True) for rep_i in self.rep])
                                              for core0_i, rank0, rank1 in zip(self.core0, self.ranks[:-1], self.ranks[1:])]

    def repk(self, rep, k, ordered=False):
        if not k:
            return tf.convert_to_tensor([[1.]], dtype=dtype)
        else:
            if ordered:
                return tf.concat([tfkron(tf.eye(k, dtype=dtype), rep_i[:, None])
                                  for rep_i in rep], axis=1)
            else:
                return np.kron(np.eye(k, dtype=dtype), rep)


    def params(self):
        return ([self.cap] + self.s_var +
        [uh._var for uh in self.UH] + [uw._var for uw in self.UW] +
        [ul._var for ul in self.UL] + [ub._var for ub in self.Ubasis] +
        [uc._var for uc in self.UC])


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



if __name__ is "__main__":
    sess = tf.InteractiveSession()
    N = 7
    K = 3
    R = K**N
    ranks = (1,3,6,6,6,3,1)
    #ranks = tuple(min(K**min(r, N-r), R) for r in range(N+1))

    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.reduce_sum(z) - tf.reduce_sum(z*true_index)

    C = np.array([[0,1,0],[0,0,1],[1,0,0]])
    orth=CanonicalPermutationCore2(N, K, ranks) #Canonical(N, K, ranks)
    orth_model = MPS(N, K, ranks, cores=orth, normalized=False)

    loss = orth_model.elbo(lambda x: 1., nsamples=10)
    #optimization routine for marginal entropy
    opt = tf.contrib.opt.ScipyOptimizerInterface(-orth_model.marginalentropy(), orth.params(), tol=1e-10,method='CG')

    #"symmetry norm"
    symf = tf.reduce_sum(symmetrynorm(orth_model.cores))

    #optimization routine for symmetry norm
    sopt = tf.contrib.opt.ScipyOptimizerInterface(symf, orth.params(), tol=1e-10,method='CG')

    sess.run(tf.global_variables_initializer())
    #tf.get_default_graph().finalize()
    #opt.minimize()
if False:
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
