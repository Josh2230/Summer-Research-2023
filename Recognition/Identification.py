# We will evaluate identification performance via this script!
from DataManager import FFilesManager
from DataManager.FFilesManager import FMLoader
from Classification.ModelingIdentification import IDModel
from DataManager.FMSeparate import FMSepLoader
from Param import IdParams


class IdentificationExp:
    def __init__(self, dataset = None, users = None, classifiers = None):
        print('setting up the dataset, users, and classifiers')
        if dataset is None:
            self.dataset = 'Smartwatch2015Acc' # default dataset
        else:
            self.dataset = dataset  # name of the dataset

        if users is None:
            self.users =[id for id in range(1, 41)]
        else:
            self.users = users

        if classifiers is None:
            self.classifiers = IdParams.DEFAULT_CLS
        else:
            self.classifiers = classifiers
        # This is a dataset level class and the dataset will remain the same for an object
        # Hence creating a feature loader object that will help setup the dataset and users
        self.FMSL = FMSepLoader(self.dataset, self.users) # setting up the dataset, users and combining the data...!
        self.X_train, self.X_test, self.y_train, self.y_test = self.FMSL.get_data_for_identification_exp()
        self.identification_models = {} # defining this dictionarry that keeps track of all  models


    def set_dataset(self, dataset):
        self.dataset = dataset
    def set_users(self, users):
        """setting users to be included in the experiments"""
        self.users = users  # user ids in integers.
    def set_classifiers(self, classifiers):
        self.classifiers = classifiers

    def train_identifiers(self):
        self.trained_identifiers = {}
        IDModel.set_training_data(self.X_train, self.y_train)
        IDModel.set_testing_data(self.X_test, self.y_test)
        for classifier in self.classifiers:
            print(f'training identification model using {classifier}')
            model = IDModel()
            model.set_classifier(classifier)
            model.train_classifier(classifier)
            self.trained_identifiers[classifier] = model
    def evaluate_identifiers(self):
        self.test_results = {}
        for classifier in self.trained_identifiers:
            print(f'testing identification model using {classifier}')
            predictions, prediction_scores = self.trained_identifiers[classifier].get_test_results()
            self.test_results[classifier] = (predictions, prediction_scores)
            self.trained_identifiers[classifier].print_krank_accuracy()

if __name__ =="__main__":
    datasets =IdParams.SeparateTrainTest
    MCC = ['RAF', 'KNN']
    datasets = 'Smartwatch2015Acc' # for testing purspose
    user_ids = [id for id in range(1, 41)]
    IDExp = IdentificationExp()
    IDExp.set_dataset(dataset='Smartwatch2015Acc')
    IDExp.set_users(users=user_ids)
    IDExp.set_classifiers(classifiers=IdParams.MCC)
    IDExp.train_identifiers()
    IDExp.evaluate_identifiers()
