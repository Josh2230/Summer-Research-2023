import os
import pandas as pd
from sklearn.model_selection import train_test_split

from Classification.Authenticator import Authenticator
from Param import IdParams, AuthParams

datasets = ['BBMAS']
classifiers = ['KNN', 'RAF', 'SVM', 'LRG',  'MLP']
classifiers = ['SVM']
def combine_feature_files(folder_location):
    ''''This function takes a location, which will have user files which will be combined into one giant data frame'''
    files = os.listdir(folder_location)
    combined_frames = []
    for file in files:
        user_id = int(file[4:-4])  # user id as numbers! assuming User<>.csv/.txt
        file_location = os.path.join(folder_location, file)  # create the path for the file
        # depending on whether you have a header or not or index column or not index_col=0, header=0 changes
        cuser_fvs = pd.read_csv(file_location, index_col=0, header=0)  # read the data from the file
        cuser_fvs.columns = cuser_fvs.columns.astype(str)

        cuser_fvs = cuser_fvs[cuser_fvs['faulty'] == False]  # discard all the faulty feature vectors
        cuser_fvs = cuser_fvs.drop(['faulty'], axis=1)  # drop the faulty column because we cant use it as a feature

        cuser_fvs['user_id'] = user_id  # appending the user id to the corresponding feature matrix
        combined_frames.append(cuser_fvs)
    combined_df = pd.concat(combined_frames)  # concatinating all the frames vertically
    combined_df.reset_index(drop=True, inplace=True)  # resettting the index and dropping the old one
    return combined_df

dataset_dictionary= {'BBMAS': 'BBMAS', 'ANTAL': 'ANTAL'}
result_table = pd.DataFrame(columns=['dataset', 'classifier', 'user', 'FAR', 'FRR', 'HTER'])
entry_id = 0
for dataset in datasets:
    for classifier in classifiers:
        print(f'Running for {classifier} classifier on {dataset}')
        if dataset in AuthParams.SeparateTrainTest: # different sessions
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
                    authenticator.scale_features(AuthParams.SCALING_METHOD)
                # if needed to select the features
                if AuthParams.SELECT_FEATURES:
                    authenticator.select_features()
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

        else: # same session
            print('Running the same session framework!')
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
                # creating an authenticator object | we can save these objects and evaluate later
                authenticator = Authenticator(genuine_train, impostor_train, genuine_test, impostor_test, classifier = classifier)
                authenticator.train()
                FAR, FRR, gen_scores, imp_scores = authenticator.evaluate()
                # uncomment the following if printing user-wise!
                print(f'dataset: {dataset_dictionary[dataset]}, classifier: {classifier}, user_id: {user_id}, FAR: {FAR}, FRR: {FRR}, HTER: {(FAR + FRR)/2}')
                print('gen_scores: ', gen_scores)
                print('imp_scores: ', imp_scores)
                result_table.loc[entry_id] = [dataset_dictionary[dataset], classifier, user_id, FAR, FRR, (FAR + FRR) / 2]
                entry_id = entry_id + 1
            print(f'Average for dataset {dataset} and {classifier}')
            current_slice = result_table.loc[(result_table['dataset'] == dataset_dictionary[dataset]) & (result_table['classifier'] == classifier)]
            print(current_slice[['FAR', 'FRR', 'HTER']].mean())

