# We will evaluate identification performance via this script!
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score

from Classification import Accuracy
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Classification.Accuracy import KRank
from Classification.TrainAuthenticatorGrid import TrainAuthenticator
from Classification.TrainIdentifierGrid import TrainIdentifier
from Param.AuthParams import FEAT_SELECT_BASIS, SELECT_FEATURES_PERCENT
from imblearn.over_sampling import ADASYN, SVMSMOTE

class Identifier:  # for each user
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def smote_data(self):
        oversample = SVMSMOTE(k_neighbors=4)  # doesnt create a model
        self.x_train, self.y_train = oversample.fit_resample(self.x_train, self.y_train)

    def scale_features(self, method='MINMAX'):
        if method == 'MINMAX':
            mm_scaler = MinMaxScaler(feature_range=(0, 1), clip=False) # clip sounds phishy!
            mm_scaler = mm_scaler.fit(self.x_train)
            self.x_train = mm_scaler.transform(self.x_train)
            self.x_test = mm_scaler.transform(self.x_test)

        elif method == 'STANDARD':
            standard_scaler = StandardScaler()
            standard_scaler = standard_scaler.fit(self.x_train)
            self.x_train = standard_scaler.transform(self.x_train)
            self.x_test = standard_scaler.transform(self.x_test)
        else:
            raise ValueError('Scaling method unknown!')


    def select_features(self):
        # print('Number of features BEFORE selection: ', self.x_train.shape[1])
        fselector = SelectKBest(FEAT_SELECT_BASIS,
            k=int(self.x_train.shape[1]*SELECT_FEATURES_PERCENT))
        fselector = fselector.fit(self.x_train, self.y_train)
        self.x_train = fselector.transform(self.x_train)
        self.x_test = fselector.transform(self.x_test)


    def train_identifier(self, classifier='KNN'):
        self.classifier = classifier
        CIO = TrainIdentifier(classifier)
        CIO.train(self.x_train, self.y_train)
        self.trained_model = CIO

    def evaluate(self, auto_compute = True):
        pred_scores = self.trained_model.get_scores(self.x_test)
        if auto_compute:
            y_pred = self.trained_model.get_preds(self.x_test)
            score_matrix = self.trained_model.get_scores(self.x_test)
            AccKRank = KRank(score_matrix, self.y_test, y_pred)
            return AccKRank.get_up_to_k_rank_accuracy(5)
