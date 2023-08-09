import os
import pandas as pd
from sklearn.model_selection import train_test_split

from Classification.Authenticator import Authenticator
from DataManager.FeatureManager import combine_feature_files
from Param import IdParams, AuthParams
# datasets = ['BBMAS', 'ANTAL']
datasets = ['BBMAS']
# classifiers = ['KNN', 'RAF', 'SVM', 'LRG',  'MLP']
classifiers = ['RAF']

dataset_dictionary= {'BBMAS': 'BBMAS', 'ANTAL': 'ANTAL', 'FRANK': 'FRANK'}
result_table = pd.DataFrame(columns=['dataset', 'classifier', 'user', 'FAR', 'FRR', 'HTER'])
entry_id = 0
for dataset in datasets: # Run for every dataset
    for classifier in classifiers: # run for every classifier
        print(f'Running for {classifier} classifier on {dataset}')
        if dataset not in AuthParams.SeparateTrainTest:
            print('Single feature file framework!')
            data_location = os.path.join(os.path.dirname(os.getcwd()), 'SavedFeatures', dataset)
            feature_location = os.path.join(data_location)
            feature_df = combine_feature_files(feature_location)
            user_ids = feature_df.user_id.unique().tolist()
            user_ids.sort()
            for user_id in user_ids:
                gen = feature_df[feature_df['user_id'] == user_id]
                imp = feature_df[feature_df['user_id'] != user_id]
                genuine_train, genuine_test = train_test_split(gen, train_size=IdParams.TRAIN_PERCENT,
                    random_state=IdParams.SEED)
                impostor_train, impostor_test = train_test_split(imp, train_size=IdParams.TRAIN_PERCENT,
                    random_state=IdParams.SEED)

                genuine_train.reset_index(drop=True, inplace=True)  # resettling the index and dropping the old one
                genuine_test.reset_index(drop=True, inplace=True)  # resettling the index and dropping the old one
                impostor_train.reset_index(drop=True, inplace=True)  # resettling the index and dropping the old one
                impostor_test.reset_index(drop=True, inplace=True)  # resettling the index and dropping the old one

                # DROP THE USER ID COLUMNS FROM EACH OF THE MATRIX, YOU CANT  USE THEM IN FEATURE
                genuine_train.drop('user_id', axis=1, inplace=True)
                impostor_train.drop('user_id', axis=1, inplace=True)
                genuine_test.drop('user_id', axis=1, inplace=True)
                impostor_test.drop('user_id', axis=1, inplace=True)

                # creating an authenticator object | we can save these objects and evaluate later
                authenticator = Authenticator(genuine_train, impostor_train, genuine_test, impostor_test,
                    classifier=classifier)

                # if needed to scale the features
                if AuthParams.SCALE_FEATURES:
                    print('Feature scaling in progress!')
                    authenticator.scale_features(AuthParams.SCALING_METHOD)
                # if needed to select the features
                if AuthParams.SELECT_FEATURES:
                    print('Feature selection in progress!')
                    authenticator.select_features()

                # training the authenticator!
                authenticator.train()

                FAR, FRR, gen_scores, imp_scores = authenticator.evaluate()
                # uncomment the following if printing user-wise!
                print(
                    f'dataset: {dataset_dictionary[dataset]}, classifier: {classifier}, user_id: {user_id}, FAR: {FAR * 100}%, FRR: {FRR * 100}%, HTER: {(FAR + FRR) / 2 * 100}%')
                # print('gen_scores: ', gen_scores)
                # print('imp_scores: ', imp_scores)
                result_table.loc[entry_id] = [dataset_dictionary[dataset], classifier, user_id, FAR, FRR, (FAR + FRR) / 2]
                entry_id = entry_id + 1
            print(f'Average for dataset {dataset} and {classifier}')
            current_slice = result_table.loc[
                (result_table['dataset'] == dataset_dictionary[dataset]) & (result_table['classifier'] == classifier)]
            print(current_slice[['FAR', 'FRR', 'HTER']].mean())

        else: # separate training testing files framework
            print('Two feature files (training and testing) framework!')
            data_location = os.path.join(os.path.dirname(os.getcwd()), 'SavedFeatures', dataset)
            training_feature_location = os.path.join(data_location, 'Training')
            train_df = combine_feature_files(training_feature_location)
            testing_feature_location = os.path.join(data_location, 'Testing')
            test_df = combine_feature_files(testing_feature_location)
            user_ids = train_df.user_id.unique().tolist()
            user_ids.sort()
            for user_id in user_ids:
                genuine_train = train_df[train_df['user_id'] == user_id]
                impostor_train = train_df[train_df['user_id'] != user_id]
                genuine_test = test_df[test_df['user_id'] == user_id]
                impostor_test = test_df[test_df['user_id'] != user_id]

                # DROP THE USER ID COLUMNS FROM EACH OF THE MATRIX, YOU CANT  USE THEM IN FEATURE
                genuine_train.drop('user_id', axis=1, inplace=True)
                impostor_train.drop('user_id', axis=1, inplace=True)
                genuine_test.drop('user_id', axis=1, inplace=True)
                impostor_test.drop('user_id', axis=1, inplace=True)

                # creating an authenticator object | we can save these objects and evaluate later
                authenticator = Authenticator(genuine_train, impostor_train, genuine_test, impostor_test, classifier = classifier)

                # if needed to scale the features
                if AuthParams.SCALE_FEATURES:
                    print('Feature scaling in progress!')
                    authenticator.scale_features(AuthParams.SCALING_METHOD)
                # if needed to select the features
                if AuthParams.SELECT_FEATURES:
                    print('Feature selection in progress!')
                    authenticator.select_features()

                # training the authenticator!
                authenticator.train()

                FAR, FRR, gen_scores, imp_scores  = authenticator.evaluate()
                result_table.loc[entry_id] = [dataset_dictionary[dataset], classifier, user_id, FAR, FRR, (FAR + FRR) / 2]
                print('gen_scores: ', gen_scores)
                print('imp_scores: ', imp_scores)
                entry_id = entry_id + 1
                # print(f' dataset: {dataset}, classifier: {classifier}, user_id: {user_id}, FAR: {FAR}, FRR: {FRR}, HTER: {(FAR + FRR) / 2}')
            print(f'Average for dataset {dataset} and {classifier}')
            current_slice = result_table.loc[(result_table['dataset'] == dataset_dictionary[dataset]) & (result_table['classifier'] == classifier)]
            print(current_slice[['FAR', 'FRR', 'HTER']].mean())


