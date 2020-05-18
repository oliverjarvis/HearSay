"""
Parameter search function
"""
import pickle
from pathlib import Path
from hyperopt import fmin, tpe, hp, Trials
import numpy

def parameter_search(ntrials, objective_function, data_path, task="veracity"):
    output_path = Path("output")
    if not output_path.is_dir():
        output_path.mkdir()
    trial_path = "trials_{}.txt".format(task)
    bp_path = "bestparams_{}.txt".format(task)
    trialsfile = open(output_path / trial_path, "wb+")
    paramsfile = open(output_path / bp_path, "wb+")

    search_space = {'data_dir':hp.choice('data_dir', [data_path]),
                    'num_dense_layers': hp.choice('nlayers', [1, 2]),
                    'num_dense_units': hp.choice('num_dense', [200, 300, 400]),
                    'num_epochs': hp.choice('num_epochs',  [32, 64]),
                    'num_lstm_units': hp.choice('num_lstm_units', [100, 200,
                                                                   300]),
                    'num_lstm_layers': hp.choice('num_lstm_layers', [1, 2, 3]),
                    'learn_rate': hp.choice('learn_rate', [1e-4, 3e-4, 1e-3, 3e-2]),
                    'mb_size': hp.choice('mb_size', [128, 264]),
                    'l2reg': hp.choice('l2reg', [0.0, 1e-4, 3e-4, 1e-3]),
                    'dropout': hp.choice('dropout', [0.2, 0.3, 0.4, 0.5]),
                    'attention': hp.choice('attention', [0, 1])
                    }
    
    trials = Trials()
    best = fmin(objective_function,
                space=search_space,
                algo=tpe.suggest,
                max_evals=ntrials,
                trials=trials,
                rstate=numpy.random.RandomState(1))
    
    
    print(best)
    
    bp = trials.best_trial['result']['Params']
    
    
    pickle.dump(trials, trialsfile)
    f.close()
    
    
    pickle.dump(bp, paramsfile)
    f.close()
    
    return bp, trials
