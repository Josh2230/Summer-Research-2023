# We will evaluate identification performance via this script!
import os
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from sklearn import linear_model

from Param import Params


class IdentificationExp:
    def __init__(self, dataset_name):
        self.feature_path = os.path.join(os.path.dirname(os.getcwd()), 'FeatureFiles', dataset_name)
        # print(f'using features files located at {self.feature_path} for classification')

    def prepare_data(self, specific_user_ids=None):
        ''''
        '''
        feature_files = os.listdir(self.feature_path)
        if specific_user_ids is None:
            ordered_file_names = ['User' + str(user_id) + '.csv' for user_id in range(1, len(feature_files) + 1)]
        else:
            ordered_file_names = ['User' + str(user_id) + '.csv' for user_id in specific_user_ids]
        print(f'Preparing data.... and users included: {ordered_file_names}')
        dataframes = []
        for file_name in ordered_file_names:
            user_id = int(file_name[4:-4])
            # print(f'user id: {user_id}----> user file name: {file_name}')
            file_path = os.path.join(self.feature_path, file_name)
            all_swipes = pd.read_csv(file_path, index_col=0)
            all_swipes = all_swipes[all_swipes['faulty'] == False]
            all_swipes = all_swipes.drop(['faulty'], axis=1)
            all_swipes['user_id'] = user_id  # appending the user id to the corresponding feature matrix
            # print(all_swipes.head(20).to_string())
            dataframes.append(all_swipes)  # creating a list of all the data frames
        self.combined_df = pd.concat(dataframes)  # concatinating all the frames

    def run_knn(self):
        ''''
        '''
        # creating a classifier object
        final_model = KNeighborsClassifier(n_neighbors=23, metric='cosine')
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with default hyperparams')

    def run_raf(self):
        ''''
        '''
        # creating a classifier object
        final_model = RandomForestClassifier(n_jobs=-1, random_state=Params.SEED)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with default param values')
    def run_svm(self):
        ''''
        '''
       # creating a classifier object
        final_model = svm.SVC(random_state=Params.SEED, tol= 1e-5, probability=True)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with default params')

    def run_mlp(self):
        ''''
        '''
        # creating a classifier object
        final_model = MLPClassifier(max_iter=1000, random_state=Params.SEED, early_stopping=True)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with def params')

    def run_lrg(self):
        ''''
        '''
        final_model = linear_model.LogisticRegression(random_state=Params.SEED, tol=1e-5)
        # setting up the parameter search using bayes search cv
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with def param')

    def run_classifier(self, classifier_name):
        ''''
        '''
        # splitting the data into training and testing part
        # if we have multiple session data then we dont have to do this!
        from sklearn.model_selection import train_test_split
        # Split the features and labels
        # Drop the 'user_id' column which represent the feature matrix
        X = self.combined_df.drop('user_id', axis=1, inplace=False)
        # Assign 'user_id' column to y the labels (user_ids)
        y = self.combined_df['user_id']
        # Splitting the dataset into 70 training and 30 testing.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                random_state=Params.SEED)
        # for the majority of the classifiers it is essential that we scale | MinMax has done better than Scaler,
        # but we can see!
        mm_scaler = MinMaxScaler()
        # print(f'applying minmax scaler')
        self.X_train = mm_scaler.fit_transform(self.X_train)
        self.X_test = mm_scaler.transform(self.X_test)

        if classifier_name == "KNN":
            self.classifier = classifier_name
            self.run_knn()
        elif classifier_name == "RAF":
            self.classifier = classifier_name
            print(f'Running {self.classifier} classifier')
            self.run_raf()
        elif classifier_name == "SVM":
            self.classifier = classifier_name
            print(f'Running {self.classifier} classifier')
            self.run_svm()
        elif classifier_name == "MLP":
            self.classifier = classifier_name
            print(f'Running {self.classifier} classifier')
            self.run_mlp()
        elif classifier_name == "LRG":
            self.classifier = classifier_name
            print(f'Running {self.classifier} classifier')
            self.run_lrg()
        else:
            raise ValueError('Unknown classifier!')
