# We will evaluate identification performance via this script!
from DataManager import FFilesManager
from DataManager.FFilesManager import FMLoader
from Classification.ModelingIdentification import IDModel
from Param import Params
class IdentificationExp:
    def __init__(self, dataset = None, users = None, classifiers = None):
        self.dataset = dataset # name of the dataset
        self.users = users # user ids in integers.
        self.classifiers = classifiers
    def set_dataset(self, dataset):
        self.dataset = dataset
    def set_users(self, users):
        """setting users to be included in the experiments"""
        self.users = users  # user ids in integers.
    def set_classifiers(self, classifiers):
        self.classifiers = classifiers
    def load_dataset(self):
        FML = FMLoader(self.dataset, self.users)
        self.X_train, self.X_test, self.y_train, self.y_test = FML.get_data_for_identification_exp()
    def train_identification_models(self):
        self.trained_models = {}
        IDModel.set_training_data(self.X_train, self.y_train)
        IDModel.set_testing_data(self.X_test, self.y_test)
        for classifier in self.classifiers:
            model = IDModel()
            model.set_classifier(classifier)
            model.train_classifier(classifier)
            self.trained_models[classifier] = model
    def evaluate_identification_models(self):
        self.test_results = {}
        for model in self.trained_models:
            predictions, prediction_scores = self.trained_models[model].get_test_results()
            self.test_results[model] = (predictions, prediction_scores)
            self.trained_models[model].print_krank_accuracy()

if __name__ =="__main__":
    MCC = ['RAF', 'SVM', 'MLP', 'LRG', 'KNN']
    OCC = ['SVM1', 'LOF', 'ISF', 'ELE']
    datasets = ['BBMAS', 'ANTAL', 'BBMASORIGINAL']
    user_ids = [id for id in range(1, 117)]
    IDE = IdentificationExp()
    IDE.set_dataset('BBMAS')
    IDE.set_users(user_ids)
    IDE.load_dataset()
    IDE.train_identification_models()
    IDE.evaluate_identification_models()
