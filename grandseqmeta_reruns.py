
#FLAGS
name = 'tmp' #'fig15_multi_restart'    #'fig15_ambigraph'
seed = 1
with_leaders = True
network = 'karate'
concentration = 1.
a = 1.
b = 1.
alpha = 1.

if name == 'tmp':
    version = 8
    network = 'ambigraph'
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 1
    nsteps = 20000

    decay = 1. #10**(-1)
    decay_steps = 40000
    optimizer = 'adabound'  # options: ams
    marginal = False
    timeit = False


    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [32]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [0.01]  # [1e-1,1e-2,1e-3]
    nsamples = [100]
elif name== 'ratetest':
    version = 6
    network = 'ambigraph'
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 1
    nsteps = 1000

    decay = 1.
    decay_steps = 40000
    optimizer = 'adabound'  # options: ams
    marginal = False
    timeit = False


    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [16]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2, 1e-1, 1., 10]  # [1e-1,1e-2,1e-3]
    nsamples = [100]

elif name == 'fig15_multi_restart':
    version = 1
    network = 'ambigraph'
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 20
    nsteps = 15000

    decay = 10 ** (-1)
    decay_steps = 40000
    optimizer = 'adabound'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [4]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-1]  # [1e-1,1e-2,1e-3]
    nsamples = [100]

elif name == 'fig15_ambigraph':
    version = 5
    network = 'ambigraph'
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 2
    nsteps = 25000

    decay = 10**(-1)
    decay_steps = 40000
    optimizer = 'adabound'  # options: ams
    marginal = False
    timeit = False


    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [1, 4, 8]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-1]  # [1e-1,1e-2,1e-3]
    nsamples = [100]


if name == 'fig15_strong_prior':
    version = 3
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 3
    nsteps = 10000

    b = 50.

    decay = 0.75
    decay_steps = 20
    optimizer = 'adabound'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [1, 4, 16]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]  # [1e-1,1e-2,1e-3]
    nsamples = [100]

elif name == 'reduced_fig15':
    with_leaders = False
    version = 2
    Ntest = 90
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 1
    nsteps = 10000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [1, 4]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]  # [1e-1,1e-2,1e-3]
    nsamples = [100]

elif name == 'multifig15':
    version = 1
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 1
    nsteps = 10000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [1,8,16]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]  # [1e-1,1e-2,1e-3]
    nsamples = [100]
elif name == 'rank1karate':
    version = 4
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 1
    nsteps = 10000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [1]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]  # [1e-1,1e-2,1e-3]
    nsamples = [1000]
elif name == 'superconvergence':
    version = 1
    Ntest = 161 #number of edges to use for testing
    K = 2 #number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    #factors variations to run experiments over
    random_restarts = 1
    nsteps = 20000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams' #options: ams
    marginal = False
    timeit = False

    #sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [8]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]#[1e-1,1e-2,1e-3]
    nsamples = [200]
elif name == 'fig15reruns':
    version = 2
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

elif name == 'highrank':
    version = 1
    Ntest = 161 #number of edges to use for testing
    K = 2 #number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    #factors variations to run experiments over
    random_restarts = 1
    nsteps = 2000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams' #options: ams
    marginal = False
    timeit = False

    #sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [32]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]#[1e-1,1e-2,1e-3]
    nsamples = [200]

elif name == 'multihighrank':
    version = 1
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 1
    nsteps = 2000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [8,16,32]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]  # [1e-1,1e-2,1e-3]
    nsamples = [200]

elif name == 'multihighrank2':
    version = 1
    Ntest = 161  # number of edges to use for testing
    K = 2  # number of communities to look for
    folder = name + 'V{}K{}'.format(version, K)

    # factors variations to run experiments over
    random_restarts = 2
    nsteps = 5000

    decay = 0.5
    decay_steps = 100
    optimizer = 'ams'  # options: ams
    marginal = False
    timeit = False

    # sizes = [5,10,20,30,40,50,75,100,150,200]
    objectives = ['relax']
    maxranks = [8,16,32]
    marginals = [False]
    unimixes = [False]
    coretypes = ['canon']
    inits = ['expectation']
    learningrates = [1e-2]  # [1e-1,1e-2,1e-3]
    nsamples = [200]


