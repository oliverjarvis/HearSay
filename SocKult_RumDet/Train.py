import json, os, pickle, argparse

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path

from Model.parameter_search import parameter_search
from Model.objective_functions import objective_function_veracity_branchLSTM
from Model.evaluation_functions import evaluation_function_veracity_branchLSTM

from Utils.Logger import Logger

def train(data_path, params_path, log_path, log_name, HPsearch):
    
    data_path = Path(data_path)
    params_path = Path(params_path)
    log_path = Path(log_path)
    log_name = log_name

    logger = Logger(log_path, log_name)

    if HPsearch:
        ntrials = 100
        paramsB, trialsB = parameter_search(ntrials, objective_function_veracity_branchLSTM, data_path)

    else:
        trialsB = pickle.load(open(params_path / 'trials_veracity.txt', "rb"))
        paramsB = pickle.load(open(params_path / 'bestparams_veracity.txt', "rb"))

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
    
    #Running the model with our different folds
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
            test_result_idB, test_result_predictionsB, test_result_labelB, confidenceB, mactest_F  = evaluation_function_veracity_branchLSTM(data_path, paramsB, metafeatures_combinations[metac_idx], use_embeddings=embeddings_present, Early_Stopping=Early_Stopping)
            precision = precision_score(test_result_labelB, test_result_predictionsB, average = 'macro')
            recall = recall_score(test_result_labelB, test_result_predictionsB, average = 'macro')
            message = "Combination: {}\nUsing Embeddings: {}\nWith EarlyStopping: {}\nResulted in: \nPrecision: {}\nRecall: {}\nF1: {}\nAccuracy: {}\n\n".format(metafeatures_combinations[metac_idx], embeddings_present, Early_Stopping, precision, recall, mactest_F, accuracy_score(test_result_labelB, test_result_predictionsB))
            logger.log(message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HearSay Train')
    parser.add_argument('--input', '-i', help='Folder containing both train/dev/test data', default="")
    parser.add_argument('--params', '-p', help='Folder for hyperparameter files', default="parameters")
    parser.add_argument('--HPsearch', help="Flag for whether or not hyperparameter search should be activated", action="store_true")
    parser.add_argument('--log_path', '-g', help="Path for logging data", default="logs")
    parser.add_argument('--log_name', '-n', help="Name for logging folder", default="log")
    parser = parser.parse_args()

    train(
        data_path = parser.input.strip(), 
        params_path = parser.params.strip(), 
        log_path = parser.log_path.strip(), 
        log_name = parser.log_name.strip(),
        HPsearch = parser.HPsearch
        )
    