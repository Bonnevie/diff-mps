#FLAGS
name = 'fig15reruns'
version = 1
Ntest = 161 #number of edges to use for testing
K = 2 #number of communities to look for
folder = name + 'V{}K{}'.format(version, K)

settings = {
 'rank': [16],
 'restarts': range(0, 3),
 'objective': ['relax'],
 'marginal': [False],
 'unimix': [False],
 'coretype': ['canon'],
 'init': ['expectation'],
 'learningrate': [0.01],
 'nsample': [200]
}

#factors variations to run experiments over
random_restarts = 10
nsteps = 10000

decay = 0.5
decay_steps = 100
optimizer = 'ams' #options: ams
marginal = False
timeit = False

#sizes = [5,10,20,30,40,50,75,100,150,200]
objectives = ['relax']
maxranks = [16]
marginals = [False]
unimixes = [False]
coretypes = ['canon']
inits = ['expectation']
learningrates = [1e-2]#[1e-1,1e-2,1e-3]
nsamples = [100]


