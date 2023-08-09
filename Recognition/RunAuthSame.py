import os
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from Classification.Authenticator import Authenticator
from DataManager.FFilesManager import combine_feature_files
from Param import IdParams, AuthParams
from Param.AuthParams import SCALING_METHOD, SCALE_FEATURES
datasets = ['FRANK']
# classifiers = ['KNN', 'RAF', 'SVM', 'LRG',  'MLP', 'GNB']
classifiers = ['KNN']
result_table = pd.DataFrame(columns=['dataset', 'classifier', 'user', 'FAR', 'FRR', 'HTER'])
entry_id = 0
for dataset in datasets: # Run for every dataset
    for classifier in classifiers: # run for every classifier
        print(f'Running for {classifier} classifier on {dataset} with scaling: {SCALING_METHOD} set to {SCALE_FEATURES}')
        data_location = os.path.join(os.path.dirname(os.getcwd()), 'SavedFeatures', dataset)
        feature_location = os.path.join(data_location)
        feature_df = combine_feature_files(feature_location)
        user_ids = feature_df.user_id.unique().tolist()
        user_ids.sort()
        combined_genuine_scores = []
        combined_impostor_scores = []
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
            # passing only
            authenticator = Authenticator(genuine_train, impostor_train, genuine_test, impostor_test)

            # if needed to scale the features
            if AuthParams.SCALE_FEATURES:
                authenticator.scale_features(AuthParams.SCALING_METHOD)
            # if needed to select the features
            if AuthParams.SELECT_FEATURES:
                authenticator.select_features()

            # training the authenticator!
            authenticator.train_verifier(classifier=classifier)

            FRR, gen_scores = authenticator.evaluate_genuine_fail()
            FAR, imp_scores = authenticator.evaluate_impostor_pass()

            # uncomment the following if printing user-wise!
            print(f'dataset: {dataset}, classifier: {classifier}, user_id: {user_id}, FAR: {round(FAR * 100, 2)}%, FRR: {round(FRR * 100, 2)}%, HTER: {round((FAR + FRR) / 2 * 100, 2)}%')
            combined_genuine_scores.extend(gen_scores)
            combined_impostor_scores.extend(imp_scores)
            result_table.loc[entry_id] = [dataset, classifier, user_id, round(FAR * 100, 2), round(FRR * 100, 2), round((FAR + FRR) / 2 * 100, 2)]
            entry_id = entry_id + 1
        print(f'Average for dataset {dataset} and {classifier}')

        # Score Analysis for Manual Thresholidng!! Right now the decision is based on probability,  the predict willl side the bigger prob
        # cgs_frame = pd.DataFrame(combined_genuine_scores, columns=['gscores', 'iscores'])
        # sns.ecdfplot(data=cgs_frame, x="gscores")
        # cis_frame = pd.DataFrame(combined_impostor_scores, columns=['gscores', 'iscores'])
        # sns.ecdfplot(data=cis_frame, x="iscores")
        # plt.show()
        #
        # sns.kdeplot(data=cgs_frame, x="gscores")
        # sns.kdeplot(data=cis_frame, x="iscores")
        # plt.legend(['gscores', 'iscores'])
        # plt.show()
        #

        current_slice = result_table.loc[(result_table['dataset'] == dataset) & (result_table['classifier'] == classifier)]
        print(current_slice[['FAR', 'FRR', 'HTER']].mean())


