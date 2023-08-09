def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import pandas as pd
import numpy as np
from Classification.Identifier import Identifier
import Param.AuthParams
from DataManager.FFilesManager import combine_feature_files
from Param import IdParams
from Param.AuthParams import SCALING_METHOD, SCALE_FEATURES, SMOTE_DATA
datasets = ['Watch2015Acc', 'Watch2015Gyro']
classifiers = ['KNN', 'RAF', 'SVM', 'LRG',  'MLP', 'GNB']
# classifiers = ['KNN']
result_table = pd.DataFrame(columns=['dataset', 'classifier', 'rank_accuracies', 'avg_acc'])
entry_id = 0

models = {}
for dataset in datasets:
    print(f'Working on {dataset} dataset')
    for classifier in classifiers:
        print(f'Running for {classifier} classifier')
        data_location = os.path.join(os.path.dirname(os.getcwd()), 'SavedFeatures', dataset)
        training_feature_location = os.path.join(data_location, 'Training')
        train_df = combine_feature_files(training_feature_location)

        testing_feature_location = os.path.join(data_location, 'Testing')
        test_df = combine_feature_files(testing_feature_location)

        experimental_users = [id for id in range(1, len(os.listdir(training_feature_location))+1)]
        # print(f'selecting only following users for identification experiments: ', experimental_users)
        train_iddf = train_df[train_df['user_id'].isin(experimental_users)]
        test_iddf = test_df[test_df['user_id'].isin(experimental_users)]
        # Split the features and labels
        # Drop the 'user_id' column which represent the feature matrix
        y_train = train_iddf['user_id']
        x_train = train_iddf.drop('user_id', axis=1, inplace=False)

        y_test = test_iddf['user_id']
        x_test = test_iddf.drop('user_id', axis=1, inplace=False)

        # creating an authenticator object | we can save these objects and evaluate later
        # passing only
        identifier = Identifier(x_train, y_train, x_test, y_test)

        if SMOTE_DATA:
            identifier.smote_data()

        # if needed to scale the features
        if IdParams.SCALE_FEATURES:
            identifier.scale_features(IdParams.SCALING_METHOD)
        # if needed to select the features
        if IdParams.SELECT_FEATURES:
            identifier.select_features()

        # training the authenticator!
        identifier.train_identifier(classifier=classifier)
        accuracies = identifier.evaluate()
        # uncomment the following if printing user-wise!
        print(
            f'dataset: {dataset}, classifier: {classifier}, accuracies: {accuracies}')
        # combined_genuine_scores.extend(gen_scores)
        # combined_impostor_scores.extend(imp_scores)
        result_table.loc[entry_id] = [dataset, classifier, accuracies, np.mean(accuracies)]
        entry_id = entry_id + 1
    print(result_table)