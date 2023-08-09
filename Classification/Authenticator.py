# We will evaluate identification performance via this script!
import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Classification.TrainAuthenticatorGrid import TrainAuthenticator
from Param.AuthParams import FEAT_SELECT_BASIS, SELECT_FEATURES_PERCENT

class Authenticator:  # for each user
    def __init__(self, genuine_train, impostor_train, genuine_test, impostor_test):
        self.genuine_train = genuine_train
        self.impostor_train = impostor_train
        self.x_train = pd.concat((genuine_train, impostor_train))
        genuine_train_labels = np.ones(genuine_train.shape[0])
        impostor_train_labels = np.zeros(self.impostor_train.shape[0])
        self.y_train = np.concatenate((genuine_train_labels, impostor_train_labels))

        self.genuine_test = genuine_test
        self.impostor_test = impostor_test

    def smote_data(self):
        oversample = SVMSMOTE()  # doesnt create a model
        self.x_train, self.y_train = oversample.fit_resample(self.x_train, self.y_train)

    def scale_features(self, method='MINMAX'):
        if method == 'MINMAX':
            mm_scaler = MinMaxScaler(feature_range=(0, 1), clip=False) # clip sounds phishy!
            mm_scaler = mm_scaler.fit(self.genuine_train) # fitting on just genuine training data!
            self.x_train = mm_scaler.transform(self.x_train)
            self.genuine_train = mm_scaler.transform(self.genuine_train)
            self.impostor_train = mm_scaler.transform(self.impostor_train)
            self.genuine_test = mm_scaler.transform(self.genuine_test)
            self.impostor_test = mm_scaler.transform(self.impostor_test)

        elif method == 'STANDARD':
            standard_scaler = StandardScaler()
            standard_scaler = standard_scaler.fit(self.genuine_train) # fitting on just genuine training data!
            self.x_train = standard_scaler.transform(self.x_train)
            self.genuine_train = standard_scaler.transform(self.genuine_train)
            self.impostor_train = standard_scaler.transform(self.impostor_train)
            self.genuine_test = standard_scaler.transform(self.genuine_test)
            self.impostor_test = standard_scaler.transform(self.impostor_test)
        else:
            raise ValueError('Scaling method unknown!')


    def select_features(self):
        # print('Number of features BEFORE selection: ', self.x_train.shape[1])
        fselector = SelectKBest(FEAT_SELECT_BASIS,
            k=int(self.x_train.shape[1]*SELECT_FEATURES_PERCENT))
        fselector = fselector.fit(self.x_train, self.y_train)
        self.x_train = fselector.transform(self.x_train)
        self.genuine_train = fselector.transform(self.genuine_train)
        self.impostor_train = fselector.transform(self.impostor_train)
        self.genuine_test = fselector.transform(self.genuine_test)
        self.impostor_test = fselector.transform(self.impostor_test)


    def train_verifier(self, classifier='KNN'):
        self.classifier = classifier
        CAM = TrainAuthenticator(classifier)  # creating one authentication mode for the current classifier
        CAM.train(self.x_train, self.y_train)
        self.trained_model = CAM

    def evaluate_genuine_fail(self, genuine_samples=None, auto_compute = True):
        if genuine_samples is None:
            genuine_samples = self.genuine_test
        gen_pred_scores = self.trained_model.get_scores(genuine_samples)
        if auto_compute:
            gen_direct_preds = self.trained_model.get_preds(genuine_samples)
            FRR = gen_direct_preds.tolist().count(0)/len(gen_direct_preds)
        else:
            pass # implemente the manual computation here
        return FRR, gen_pred_scores
    def evaluate_impostor_pass(self, impostor_samples=None, auto_compute = True):
        if impostor_samples is None:
            impostor_samples = self.impostor_test
        imp_pred_scores = self.trained_model.get_scores(impostor_samples)
        if auto_compute:
            imp_direct_preds = self.trained_model.get_preds(impostor_samples)
            FAR = imp_direct_preds.tolist().count(1)/len(imp_direct_preds)
        else:
            # TODO: implement manual computation of
            pass
        return FAR, imp_pred_scores