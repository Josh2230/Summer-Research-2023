# We will evaluate identification performance via this script!
import numpy as np

from Classification.ModelingAuthentication import AuthModel
from DataManager import FFilesManager
from DataManager.FFilesManager import FMLoader
from Classification.ModelingIdentification import IDModel
from Param import AuthParams


# TODO: think abou how to keep only one copy of auth data and use N classifiers
class Authenticator: # for each user
    def __init__(self, genuine_train, genuine_test, impostor_train, impostor_test):
        self.genuine_train = genuine_train
        self.impostor_train = impostor_train
        self.x_train = np.vstack((genuine_train, impostor_train))
        genuine_train_labels = np.ones(genuine_train.shape[0])
        # print(f'genuine_train_labels: {genuine_train_labels.shape}')
        impostor_train_labels = np.zeros(self.impostor_train.shape[0])
        # print(f'impostor_train_labels: {impostor_train_labels.shape}')
        self.y_train = np.concatenate((genuine_train_labels, impostor_train_labels))
        self.genuine_test = genuine_test
        self.impostor_test = impostor_test
        self.x_test = np.vstack((genuine_test, impostor_test))
        genuine_test_labels = np.ones(genuine_test.shape[0])
        impostor_test_labels =np.zeros(impostor_test.shape[0])
        self.y_test = np.concatenate((genuine_test_labels, impostor_test_labels))
        self.user_id = None
        self.classifiers = []
        self.auth_models = {}  # keep the name of the classifier and the corresponding model
    def set_classifiers(self, classifiers):
        self.classifiers = classifiers
    def set_user_id(self, user_id):
        self.user_id= user_id
    def train_models(self):
        for classifier in self.classifiers:
            AM  = AuthModel()
            AM.set_classifier(classifier)
            AM.train(self.x_train, self.y_train, self.x_test, self.y_test)
            self.auth_models[classifier] = AM
    def evaluate_models(self):
        for classifier in self.classifiers:
            self.auth_models[classifier].gen_preds(self.genuine_test)
            self.auth_models[classifier].imp_preds(self.impostor_test)
            print(f'far : {self.auth_models[classifier].get_far()}, frr: {self.auth_models[classifier].get_frr()}')


class AuthenticationExp: # for each dataset
    def __init__(self, dataset = None, users = None, classifiers = None):
        self.dataset = dataset # name of the dataset
        self.users = users # user ids in integers.
        self.classifiers = classifiers
        self.user_authentication_models = {}
    def set_dataset(self, dataset):
        self.dataset = dataset
    def set_users(self, users):
        """setting users to be included in the experiments"""
        self.users = users  # user ids in integers.
    def set_classifiers(self, classifiers):
        self.classifiers = classifiers
    def train_authenticators(self): # authenticator will contain details of the user and models trained for that user
        self.userwise_authentication_models = {}
        for user_id in self.users:
            FML = FMLoader(self.dataset, self.users)
            gen_train, imp_train, gen_test, imp_test = FML.get_data_for_auth_exp(user_id)
            user_authenticator = Authenticator(gen_train, imp_train, gen_test, imp_test)
            user_authenticator.set_classifiers(classifiers=self.classifiers)
            user_authenticator.train_models()
            user_authenticator.set_user_id(user_id)
            self.userwise_authentication_models[user_id] = user_authenticator

    def evaluate_authenticators(self):
        for user_key in self.userwise_authentication_models:
            self.userwise_authentication_models[user_key].evaluate_models()


if __name__ =="__main__":
    MCC = ['RAF', 'SVM', 'MLP', 'LRG', 'KNN']
    # MCC = ['KNN']
    OCC = ['SVM1', 'LOF', 'ISF', 'ELE']
    datasets = ['BBMAS', 'ANTAL', 'BBMASORIGINAL']
    user_ids = [id for id in range(1, 5)]
    AExp = AuthenticationExp(dataset='BBMAS', users=user_ids, classifiers=MCC)
    AExp.train_authenticators()
    AExp.evaluate_authenticators()