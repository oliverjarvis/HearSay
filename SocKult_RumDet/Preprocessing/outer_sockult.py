from parameter_search import parameter_search
from objective_functions import objective_function_veracity_branchLSTM
from evaluation_functions import evaluation_function_veracity_branchLSTM
import json
import os
import pickle
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
import time
from datetime import datetime
import atexit
#%%
class Logger:
    def __init__(self, log_path, log_name=""):
        if not os.path.exists(log_path):
            Path(log_path).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_path)
        self.log_name = log_name + datetime.now().strftime("%a%d%B-%H_%M") + ".log"
        self.log_path = self.path / self.log_name
        self.log_file = open(self.log_path, "w+")
        self.log_file.close()
        print("<Saving logs to: [{}]>".format(self.path / log_name))
        atexit.register(self.cleanup)
    def cleanup(self):
        print("<Closing log file>")
        self.log_file.close()
    def log(self, message):
        print("<Writing log...>")
        with open(self.log_path, "a") as f:
            f.write(message)
        

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

#Logger
log_path = "C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\output\\logger"
log_name = "outputs"
logger = Logger(log_path, log_name)

best_trial_idB = trialsB.best_trial["tid"]
best_trial_lossB = trialsB.best_trial["result"]["loss"]
dev_result_idB = trialsB.attachments["ATTACH::%d::ID" % best_trial_idB]
dev_result_predictionsB = trialsB.attachments["ATTACH::%d::Predictions" % best_trial_idB]
dev_result_labelB = trialsB.attachments["ATTACH::%d::Labels" % best_trial_idB]

print(accuracy_score(dev_result_labelB, dev_result_predictionsB))
print(f1_score(dev_result_labelB, dev_result_predictionsB, average='macro'))

metafeatures_combinations = [
    ["stance"], # stance = S
    ["social_interest", "user_information"], # Metadata = MD
    ["cosine_similarity"], # Semantic content = SC
    ["stance", "social_interest", "user_information"], # S + MD
    ["cosine_similarity", "social_interest", "user_information"], # SC + MD
    ["cosine_similarity", "stance"], # SC + S
    ["cosine_similarity", "social_interest", "user_information", "stance"] # FULL MODEL
]
event_splits = pickle.load(open("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\splits.pickle", "rb"))

for Early_Stopping in [False, True]:
    for metac_idx in range(len(metafeatures_combinations)):
        if metac_idx in [0, 1, 3]:
            embeddings_present = False
        else:
            embeddings_present = True
        print("Commencing training with the combination: {}".format(metafeatures_combinations[metac_idx]))
        print("Using Embeddings: ", embeddings_present)
        if Early_Stopping:
            paramsB["num_epochs"] = 64
        test_result_idB, test_result_predictionsB, test_result_labelB, confidenceB, mactest_F  = evaluation_function_veracity_branchLSTM(paramsB, metafeatures_combinations[metac_idx], use_embeddings=embeddings_present, Early_Stopping=Early_Stopping)
        precision = precision_score(test_result_labelB, test_result_predictionsB, average = 'macro')
        recall = recall_score(test_result_labelB, test_result_predictionsB, average = 'macro')
        message = "Combination: {}\nUsing Embeddings: {}\nWith EarlyStopping: {}\nResulted in: \nPrecision: {}\nRecall: {}\nF1: {}\nAccuracy: {}\n\n".format(metafeatures_combinations[metac_idx], embeddings_present, Early_Stopping, precision, recall, mactest_F, accuracy_score(test_result_labelB, test_result_predictionsB))
        logger.log(message)
    #print("With combination: ", metac, ":", sep=" ")
    #print("F1: {}".format(mactest_F))
    #print("Accuracy: {}".format(accuracy_score(test_result_labelB, test_result_predictionsB)))
    #print("")

#print(accuracy_score(test_result_labelB, test_result_predictionsB))
#print(f1_score(test_result_labelB, test_result_predictionsB, average='macro'))

#confidenceB = [1.0 for i in range((len(test_result_predictionsB)))]

#print(accuracy_score(test_result_labelB,test_result_predictionsB))

#print(mactest_F)
#%%
#a = convertsave_competitionformat(dev_result_idB, dev_result_predictionsB, confidenceB)

#b = convertsave_competitionformat(test_result_idB, test_result_predictionsB,confidenceB )



"""metafeatures_combinations = [
    ["stance"], # stance = S
    ["social_interest", "user_information"], # Metadata = MD
    ["cosine_similarity"], # Semantic content = SC
    ["stance", "social_interest", "user_information"], # SC + MD
    #["cosine_similarity", "user_information"], # SC + ui
    #["cosine_similarity", "social_interest"], # SC + si
    ["cosine_similarity", "social_interest", "user_information"], # SC + MD
    ["cosine_similarity", "stance"], # SC + S
    #["cosine_similarity", "user_information", "stance"], # SC + ui + S
    #["cosine_similarity", "social_interest", "stance"], # SC + si + S
    ["cosine_similarity", "social_interest", "user_information", "stance"] # over 9000
]"""