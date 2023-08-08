# We will evaluate identification performance via this script!
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Classification.TrainAuthenticatorGrid import TrainAuthenticator
from Param import AuthParams


# TODO: think about how to keep only one copy of auth data and use N classifiers

class Authenticator:  # for each user
    def __init__(self, genuine_train, impostor_train, genuine_test, impostor_test, classifier):
        self.genuine_train = genuine_train
        self.impostor_train = impostor_train
        self.x_train = np.vstack((genuine_train, impostor_train))
        genuine_train_labels = np.ones(genuine_train.shape[0])
        impostor_train_labels = -1 * np.ones(self.impostor_train.shape[0])
        self.y_train = np.concatenate((genuine_train_labels, impostor_train_labels))
        self.genuine_test = genuine_test
        self.impostor_test = impostor_test
        self.x_test = np.vstack((genuine_test, impostor_test))
        genuine_test_labels = np.ones(genuine_test.shape[0])
        impostor_test_labels = -1 * np.ones(impostor_test.shape[0])
        self.y_test = np.concatenate((genuine_test_labels, impostor_test_labels))
        self.classifier = classifier

    def scale_features(self, method='MinMax'):
        # print('------------before scaling------------')
        if method == 'MINMAX':
            mm_scaler = MinMaxScaler()
            mm_scaler.fit(self.x_train)
            # print(f'applying MinMaxScaler')
            self.x_train = mm_scaler.transform(self.x_train)
            self.x_test = mm_scaler.transform(self.x_test)
            self.genuine_train = mm_scaler.transform(self.genuine_train)
            self.impostor_train = mm_scaler.transform(self.impostor_train)
            self.genuine_test = mm_scaler.transform(self.genuine_test)
            self.impostor_test = mm_scaler.transform(self.impostor_test)

        elif method == 'STANDARD':
            standard_scaler = StandardScaler()
            standard_scaler.fit(self.x_train)
            # print(f'applying StandardScaler')
            self.x_train = standard_scaler.transform(self.x_train)
            self.x_test = standard_scaler.transform(self.x_test)
            self.genuine_train = standard_scaler.transform(self.genuine_train)
            self.impostor_train = standard_scaler.transform(self.impostor_train)
            self.genuine_test = standard_scaler.transform(self.genuine_test)
            self.impostor_test = standard_scaler.transform(self.impostor_test)
        else:
            raise ValueError('Scaling method unknown!')
        # print('------------after scaling------------')


    def select_features(self):
        # print('Number of features BEFORE selection: ', self.x_train.shape[1])
        fselector = SelectKBest(mutual_info_classif,
            k=int(self.x_train.shape[1]*AuthParams.SELECT_FEATURES_PERCENT))  # selecting 2/3rd of the features only
        fselector.fit(self.x_train, self.y_train)
        self.x_train = fselector.transform(self.x_train)
        self.genuine_test = fselector.transform(self.genuine_test)
        self.impostor_test = fselector.transform(self.impostor_test)
        self.x_test = fselector.transform(self.x_test)  # not necessary!
        # print('Number of features AFTER selection: ', self.x_train.shape[1])

    def train(self):
        CAM = TrainAuthenticator(self.classifier)  # creating one authentication mode for the current classifier
        if self.classifier not in AuthParams.AUTOCLS:
            CAM.train(self.x_train, self.y_train)
        else:
            CAM.train(self.x_train, self.y_train, self.x_test, self.y_test)
        self.trained_model = CAM
        return self

    def evaluate(self):
        self.trained_model.get_genuine_predictions(self.genuine_test)
        self.trained_model.get_impostor_predictions(self.impostor_test)
        FAR = self.trained_model.get_far()
        FRR = self.trained_model.get_frr()
        gen_scores = self.trained_model.get_genuine_scores(self.x_test)
        imp_scores = self.trained_model.get_imp_scores(self.x_test)
        return FAR, FRR, gen_scores, imp_scores
