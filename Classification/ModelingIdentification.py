from lazypredict.Supervised import LazyClassifier
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from Param import Params



class IDModel:
    # the data could be made
    def __init__(self, classifier='RAF'): # default Random Forest
        self.classifier = classifier
        self.predictions = None
        self.prediction_scores = None

    def set_training_data(X_train, y_train):
        IDModel.X_train = X_train
        IDModel.y_train = y_train

    def set_testing_data(X_test, y_test):
        IDModel.X_test = X_test
        IDModel.y_test = y_test

    def set_classifier(self, classifier):
        self.classifier = classifier
    def train_classifier(self, classifier=None):
        if classifier is not None: # updating the classifier
            self.classifier = classifier
        if self.classifier  == "KNN":
            self.train_knn()
        elif self.classifier == "RAF":
            print(f'Training using {self.classifier} classifier')
            self.train_raf()
        elif self.classifier == "SVM":
            print(f'Training using {self.classifier} classifier')
            self.train_svm()
        elif self.classifier == "MLP":
            print(f'Training using {self.classifier} classifier')
            self.train_mlp()
        elif self.classifier == "LRG":
            print(f'Training using {self.classifier} classifier')
            self.train_lrg()
        elif self.classifier == 'LZP':
            print(f'Training using {self.classifier} classifier')
            self.get_results_from_lazy()
        else:
            raise ValueError('Unknown classifier!')

    def get_test_results(self):
            return self.get_predictions(), self.get_prediction_scores()


    def get_predictions(self):
        if self.classifier != 'LZP':
            self.predictions = self.final_model.predict(IDModel.X_test)
            return self.predictions

    def get_prediction_scores(self):
        if self.classifier != 'LZP':
            self.prediction_scores = self.final_model.predict_proba(IDModel.X_test)
            return self.prediction_scores

    def print_krank_accuracy(self, k=4):
        rank_k_acc = {}
        for i in range(1, k):
            rank_k_acc['rank' + str(i)] = top_k_accuracy_score(IDModel.y_test, self.prediction_scores, k=i)
        print(rank_k_acc)

    def print_authentication_error_rates(self):
        print(classification_report(IDModel.y_test, self.prediction_scores, target_names=[-1, 1]))

    def get_results_from_lazy(self):
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        lazy_model = LazyClassifier(predictions=True)
        models, predictions = lazy_model.fit(IDModel.X_train, IDModel.X_test, IDModel.y_train, IDModel.y_test)
        return models, predictions

    def train_raf(self):
        if Params.HYPER_TUNE:
            n_estimators = [int(item) for item in range(500, 1001, 200)]
            max_depth = [int(item) for item in range(2, 4, 1)]
            min_samples_leaf = [int(item) for item in range(2, 3, 1)]
            max_features = ['sqrt', 'log2']
            # Create the param grid
            param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                          'min_samples_leaf': min_samples_leaf}
            # creating a classifier object
            model = RandomForestClassifier(n_jobs=-1, random_state=Params.SEED)
            # setting up the parameter search using bayes search cv
            optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
            # fitting the training data to the model
            optimal_model.fit(IDModel.X_train, IDModel.y_train)
            # accessing the best values of the hyperparameters
            n_estimators = optimal_model.best_params_['n_estimators']
            max_features = optimal_model.best_params_['max_features']
            max_depth = optimal_model.best_params_['max_depth']
            min_samples_leaf = optimal_model.best_params_['min_samples_leaf']
            # Retraining the model again | this is not necessary, just my paranoia
            self.final_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                max_leaf_nodes=min_samples_leaf, n_jobs=-1, random_state=Params.SEED)
            self.final_model.fit(IDModel.X_train, IDModel.y_train)
        else:
            self.final_model = RandomForestClassifier()
            self.final_model.fit(IDModel.X_train, IDModel.y_train)
    def train_knn(self):
        if Params.HYPER_TUNE:
            n_neighbors = [int(x) for x in list(range(1, 11, 2))]  # 1 NN
            dist_met = ['manhattan', 'euclidean', 'cosine']
            weights = ['uniform', 'distance']
            leaf_size = [20]
            param_grid = {'n_neighbors': n_neighbors, 'metric': dist_met, 'leaf_size': leaf_size, 'weights': weights}
            # creating a classifier object
            classifier_obj = KNeighborsClassifier()
            # setting up the parameter search using bayes search cv
            classifier_obj = BayesSearchCV(estimator=classifier_obj, search_spaces=param_grid, cv=5, scoring='accuracy')
            # fitting the training data to the model
            classifier_obj.fit(IDModel.X_train, IDModel.y_train)
            # accessing the best values of the hyperparameters
            best_nn = classifier_obj.best_params_['n_neighbors']
            best_dist = classifier_obj.best_params_['metric']
            best_leaf_size = classifier_obj.best_params_['leaf_size']
            best_weight = classifier_obj.best_params_['weights']
            print(f'neighbors:{best_nn}, best_dist: {best_dist} best_leaf_size:{best_leaf_size}, best_weight:{best_weight}')
            # Retraining the model again | this is not necessary, just my paranoia
            self.final_model = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist, weights=best_weight,
                leaf_size=best_leaf_size)
            self.final_model.fit(IDModel.X_train, IDModel.y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = KNeighborsClassifier()
            self.final_model.fit(IDModel.X_train, IDModel.y_train)

    def train_lrg(self):
        if Params.HYPER_TUNE:
            param_grid = {'solver': ['newton-cg'], 'C': [0.1, 0.2, 0.4, 0.45, 0.5]}
            # creating a classifier object
            model = linear_model.LogisticRegression(random_state=Params.SEED, tol=1e-5)
            # setting up the parameter search using bayes search cv
            optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
            # fitting the training data to the model
            optimal_model.fit(IDModel.X_train, IDModel.y_train)
            # accessing the best values of the hyperparameters
            solver = optimal_model.best_params_['solver']
            cval = optimal_model.best_params_['C']
            penalty = optimal_model.best_params_['penalty']
            # Retraining the model again | this is not necessary, just my paranoia
            self.final_model = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty, random_state=Params.SEED,
                tol=1e-5)
            self.final_model.fit(IDModel.X_train, IDModel.y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = linear_model.LogisticRegression()
            self.final_model.fit(IDModel.X_train, IDModel.y_train)

    def train_mlp(self):
        if Params.HYPER_TUNE:
            hlsize = []
            for flayer in range(100, 301, 100):
                # hlsize.append((flayer,))
                for slayer in range(50, 101, 50):
                    hlsize.append((flayer, slayer))  # print(hlsize)  # solver = ['adam', 'lbfgs']
            solver = ['adam']
            # activation = ['relu','tanh']
            activation = ['relu']
            # Read: "https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html"
            # Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by
            # constraining the size of the weights.Increasing alpha may fix high variance (a sign of overfitting) by
            # encouraging smaller weights, resulting in a decision boundary plot that appears with lesser
            # curvatures.Similarly, decreasing alpha may fix high bias (a sign of underfitting) by encouraging larger
            # weights, potentially resulting in a more complicated decision boundary.
            # please look at this while tunning alpha -- the regularization param https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
            alpha = [0.1, 0.2, 0.4, 0.8, 1.6]
            # default is 0.0001, # Experiment with this three values.. higher the alpha, simpler the boundary
            learning_rate = ['adaptive']
            # Create the random grid
            param_grid = {'hidden_layer_sizes': hlsize, 'activation': activation, 'solver': solver, 'alpha': alpha,
                          'learning_rate': learning_rate}
            # creating a classifier object
            model = MLPClassifier(max_iter=1000, random_state=Params.SEED, early_stopping=True)
            # setting up the parameter search using bayes search cv
            optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
            # fitting the training data to the model
            optimal_model.fit(IDModel.X_train, IDModel.y_train)
            # accessing the best values of the hyperparameters
            hlayers = optimal_model.best_params_['hidden_layer_sizes']
            activation = optimal_model.best_params_['activation']
            solver = optimal_model.best_params_['solver']
            alpha = optimal_model.best_params_['alpha']
            learning_rate = optimal_model.best_params_['learning_rate']
            # Retraining the model again | this is not necessary, just my paranoia
            self.final_model = MLPClassifier(max_iter=1000, hidden_layer_sizes=hlayers, activation=activation,
                random_state=Params.SEED, solver=solver, alpha=alpha, learning_rate=learning_rate)
            self.final_model.fit(IDModel.X_train, IDModel.y_train)
        else:
            self.final_model = MLPClassifier()
            self.final_model.fit(IDModel.X_train, IDModel.y_train)

    def train_svm(self):
        if Params.HYPER_TUNE:
            # creating parameter grid
            CVals = [0.02, 0.04, 0.08, 0.16, 0.32]
            Gammas = ['scale']
            Kernels = ['linear', 'rbf']
            param_grid = {'C': CVals, 'gamma': Gammas, 'kernel': Kernels}
            # creating a classifier object
            model = svm.SVC(random_state=Params.SEED, tol=1e-5, probability=True)
            # setting up the parameter search using bayes search cv
            optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
            # fitting the training data to the model
            optimal_model.fit(IDModel.X_train, IDModel.y_train)
            # accessing the best values of the hyperparameters
            cval = optimal_model.best_params_['C']
            gamma = optimal_model.best_params_['gamma']
            kernel = optimal_model.best_params_['kernel']
            # Retraining the model again | this is not necessary, just my paranoia
            self.final_model = svm.SVC(C=cval, gamma=gamma, kernel=kernel, random_state=Params.SEED, tol=1e-5, probability=True)
            self.final_model.fit(IDModel.X_train, IDModel.y_train)
        else:
            self.final_model = svm.SVC(probability=True)
            self.final_model.fit(IDModel.X_train, IDModel.y_train)