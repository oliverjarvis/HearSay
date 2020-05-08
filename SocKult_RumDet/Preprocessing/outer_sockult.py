from numpy.random import seed
seed(364)
import tensorflow as tf
tf.random.set_seed(364)
from parameter_search import parameter_search
from objective_functions import objective_function_veracity_branchLSTM
from evaluation_functions import evaluation_function_veracity_branchLSTM
import json

from sklearn.metrics import f1_score, accuracy_score

#%%
    
def labell2strB(label):
    
    if label == 0:
        return("true")
    elif label == 1:
        return("false")
    elif label == 2:
        return("unverified")
    else:
        print(label)

#%%

def convertsave_competitionformat(idsB, predictionsB, confidenceB):
    
    subtaskbenglish = {}
    
    
    for i,id in enumerate(idsB):
        subtaskbenglish[id] = [labell2strB(predictionsB[i]),confidenceB[i]]
    
    answer = {}
    answer['subtaskbenglish'] = subtaskbenglish
    
    with open("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\output\\answer.json", 'w') as f:
        json.dump(answer, f)
        
    return answer
#%%
print ('Rumour Veracity classification') 
ntrials = 100
task = 'veracity'
paramsB, trialsB = parameter_search(ntrials, objective_function_veracity_branchLSTM, task)


#Hyperparameters
#search_space = {'num_dense_layers':1,
#                'num_dense_units':200,
#                'num_epochs': 100,
#                'num_lstm_units': 100,
#                'num_lstm_layers': 1,
#                'learn_rate':1e-4,
#                'mb_size': 128,
#                'l2reg': 1e-3,
#                'rng_seed': 364
#                }

#output = objective_function_veracity_branchLSTM(search_space)

#%%
best_trial_idB = trialsB.best_trial["tid"]
best_trial_lossB = trialsB.best_trial["result"]["loss"]
dev_result_idB = trialsB.attachments["ATTACH::%d::ID" % best_trial_idB]
dev_result_predictionsB = trialsB.attachments["ATTACH::%d::Predictions" % best_trial_idB]
dev_result_labelB = trialsB.attachments["ATTACH::%d::Labels" % best_trial_idB]
confidenceB = [1.0 for i in range((len(dev_result_predictionsB)))]

print(accuracy_score(dev_result_labelB,dev_result_predictionsB))
print(f1_score(dev_result_labelB,dev_result_predictionsB,average='macro'))

#%%
#test_result_idB, test_result_predictionsB, confidenceB  = evaluation_function_veracity_branchLSTM(paramsB)

#confidenceB = [1.0 for i in range((len(test_result_predictionsB)))]

#%%
a = convertsave_competitionformat(dev_result_idB, dev_result_predictionsB, confidenceB)

#a = convertsave_competitionformat(test_result_id, test_result_predictions, test_result_idB, test_result_predictionsB,confidenceB )
