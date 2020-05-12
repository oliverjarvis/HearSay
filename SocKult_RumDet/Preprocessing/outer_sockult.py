from parameter_search import parameter_search
from objective_functions import objective_function_veracity_branchLSTM
from evaluation_functions import evaluation_function_veracity_branchLSTM
import json
import pickle
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
#ntrials = 100
task = 'veracity'
#paramsB, trialsB = parameter_search(ntrials, objective_function_veracity_branchLSTM, task)

#%%
trialsB = pickle.load(open("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\output\\trials2_veracity.txt", "rb"))
paramsB = pickle.load(open("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\output\\bestparams2_veracity.txt", "rb"))

best_trial_idB = trialsB.best_trial["tid"]
best_trial_lossB = trialsB.best_trial["result"]["loss"]
dev_result_idB = trialsB.attachments["ATTACH::%d::ID" % best_trial_idB]
dev_result_predictionsB = trialsB.attachments["ATTACH::%d::Predictions" % best_trial_idB]
dev_result_labelB = trialsB.attachments["ATTACH::%d::Labels" % best_trial_idB]

print(accuracy_score(dev_result_labelB, dev_result_predictionsB))
print(f1_score(dev_result_labelB, dev_result_predictionsB, average='macro'))

metafeatures_combinations = [
    #["cosine_similarity", "user_information"],
    #["cosine_similarity", "social_interest"],
    #["cosine_similarity", "social_interest", "user_information"],
    ["cosine_similarity", "user_information", "stance"],
    ["cosine_similarity", "social_interest", "stance"]
    #["cosine_similarity", "social_interest", "user_information", "stance"]
]
paramsB["num_epochs"] = 40

for metac in metafeatures_combinations:
    print("Commencing training with the combination: {}".format(metac))
    test_result_idB, test_result_predictionsB, test_result_labelB, confidenceB, mactest_F  = evaluation_function_veracity_branchLSTM(paramsB, metac)
    print("With combination: ", metac, ":", sep=" ")
    print("F1: {}".format(mactest_F))
    print("Accuracy: {}".format(accuracy_score(test_result_labelB, test_result_predictionsB)))
    print("")
    
#print(accuracy_score(test_result_labelB, test_result_predictionsB))
#print(f1_score(test_result_labelB, test_result_predictionsB, average='macro'))

#confidenceB = [1.0 for i in range((len(test_result_predictionsB)))]

#print(accuracy_score(test_result_labelB,test_result_predictionsB))

#print(mactest_F)
#%%
#a = convertsave_competitionformat(dev_result_idB, dev_result_predictionsB, confidenceB)

b = convertsave_competitionformat(test_result_idB, test_result_predictionsB,confidenceB )
