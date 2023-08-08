from lazypredict.Supervised import LazyClassifier
from sklearn import svm, linear_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from Param import AuthParams


class TrainAuthenticator:
    # the data could be made
    def __init__(self, classifier=None):  # default Random Forest
        self.gen_predictions = None
        self.imp_predictions = None
        self.classifier = classifier
        self.final_model = None
        self.gen_scores = []
        self.imp_scores = []

        from sklearn.metrics import confusion_matrix, make_scorer

        def scoring_hter(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            far = fp / (fp + tn)
            frr = fn / (fn + tp)
            hter = (far + frr) / 2
            if hter < 0:
                raise ValueError('ERROR: HTER CANT BE NEATIVE')
            return hter

        def scoring_far(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            far = fp / (fp + tn)
            frr = fn / (fn + tp)
            hter = (far + frr) / 2
            if hter < 0:
                raise ValueError('ERROR: FAR CANT BE NEATIVE')
            return far

        def scoring_frr(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            far = fp / (fp + tn)
            frr = fn / (fn + tp)
            hter = (far + frr) / 2
            if hter < 0:
                raise ValueError('ERROR: FAR CANT BE NEATIVE')
            return frr

        self.scorer_hter = make_scorer(scoring_hter, greater_is_better=False)
        self.scorer_far = make_scorer(scoring_far, greater_is_better=False)
        self.scorer_frr = make_scorer(scoring_frr, greater_is_better=False)

    def set_classifier(self, classifier):
        self.classifier = classifier

    def train(self, x_train, y_train, x_test=None, y_test=None):
        if self.classifier == "KNN":
            self.train_knn(x_train, y_train)
        elif self.classifier == "RAF":
            self.train_raf(x_train, y_train)
        elif self.classifier == "SVM":
            self.train_svm(x_train, y_train)
        elif self.classifier == "MLP":
            self.train_mlp(x_train, y_train)
        elif self.classifier == "LRG":
            self.train_lrg(x_train, y_train)
        elif self.classifier == 'LZP':
            self.print_results_from_lazy(x_train, x_test, y_train, y_test)
        else:
            raise ValueError('Unknown classifier!')

    def print_results_from_lazy(self, x_train, x_test, y_train, y_test):
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        self.final_model = LazyClassifier(predictions=True)
        models, predictions = self.final_model.fit(x_train, x_test, y_train, y_test)
        print(models)
        return models, predictions

    def get_genuine_predictions(self, x_test):
        if self.classifier not in AuthParams.AUTOCLS:
            self.gen_predictions = self.final_model.predict(x_test)
            return self.gen_predictions

    def get_impostor_predictions(self, x_test):
        if self.classifier not in AuthParams.AUTOCLS:
            self.imp_predictions = self.final_model.predict(x_test)
            return self.imp_predictions

    def get_genuine_scores(self, x_test):
        if self.classifier not in AuthParams.AUTOCLS:
            self.gen_scores = self.final_model.predict_proba(x_test)
            return self.gen_scores

    def get_imp_scores(self, x_test):
        if self.classifier not in AuthParams.AUTOCLS:
            self.imp_scores = self.final_model.predict_proba(x_test)
            return self.imp_scores

    def get_far(self):
        count_ones = 0
        for item in self.imp_predictions:
            if item == 1:
                count_ones = count_ones + 1
        self.far = count_ones / len(self.imp_predictions)
        return self.far

    def get_frr(self):
        count_ones = 0
        for item in self.gen_predictions:
            if item == 1:
                count_ones = count_ones + 1
        self.frr = 1 - (count_ones / len(self.gen_predictions))  # 1-TAR
        return self.frr

    def train_raf(self, x_train, y_train):
        # print('Training RAF')
        if AuthParams.HYPER_TUNE:
            n_estimators = [int(item) for item in range(500, 1001, 200)]
            max_depth = [int(item) for item in range(2, 4, 1)]
            min_samples_leaf = [int(item) for item in range(2, 3, 1)]
            max_features = [x / 100 for x in range(30, 51, 20)]  # need to remain float for percentages
            # max_features = ['sqrt', 'log2']
            # Create the random grid
            param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                          'min_samples_leaf': min_samples_leaf}
            model = RandomForestClassifier(n_jobs=-1, random_state=AuthParams.SEED)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=self.scorer_hter)
            optimal_model.fit(x_train, y_train)
            n_estimators = optimal_model.best_params_['n_estimators']
            max_features = optimal_model.best_params_['max_features']
            max_depth = optimal_model.best_params_['max_depth']
            min_samples_leaf = optimal_model.best_params_['min_samples_leaf']
            # print(f'n_estimators: {n_estimators}, max_features: {max_features}, max_depth:{max_depth}, min_samples_leaf:{min_samples_leaf}')
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = RandomForestClassifier(n_estimators=200)
            self.final_model.fit(x_train, y_train)

    def train_knn(self, x_train, y_train):
        # print('Training KNN')
        if AuthParams.HYPER_TUNE:
            n_neighbors = [int(x) for x in range(1, 10, 2)]
            # print('n_neighbors',n_neighbors)   # hello
            dist_met = ['manhattan', 'euclidean', 'cosine']
            # create the random grid
            param_grid = {'n_neighbors': n_neighbors, 'metric': dist_met}
            model = KNeighborsClassifier()
            # scoring_function = 'f1'
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=self.scorer_hter)
            # loss function!
            optimal_model.fit(x_train, y_train)
            best_nn = optimal_model.best_params_['n_neighbors']
            best_dist = optimal_model.best_params_['metric']
            # print(f'best_nn:{best_nn}, best_dist:{best_dist}')
            self.final_model = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = KNeighborsClassifier()
            self.final_model.fit(x_train, y_train)

    def train_lrg(self, x_train, y_train):
        # print('Training LRG')
        if AuthParams.HYPER_TUNE:
            param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                          'C': [100, 100, 10, 1.0, 0.1, 0.01], 'penalty': ['l1', 'l2', 'elasticnet']}
            model = linear_model.LogisticRegression(random_state=AuthParams.SEED, tol=1e-5)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=self.scorer_hter)
            optimal_model.fit(x_train, y_train)
            solver = optimal_model.best_params_['solver']
            cval = optimal_model.best_params_['C']
            penalty = optimal_model.best_params_['penalty']
            # print(f'solver: {solver}, cval: {cval}, penalty: {penalty}')
            self.final_model = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty,
                random_state=AuthParams.SEED, tol=1e-5)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = linear_model.LogisticRegression(C=1000)
            self.final_model.fit(x_train, y_train)

    def train_mlp(self, x_train, y_train):
        # print('Training MLP')
        if AuthParams.HYPER_TUNE:
            hlsize = []
            for flayer in range(100, 301, 100):
                # hlsize.append((flayer,))
                for slayer in range(50, 101, 50):
                    hlsize.append((flayer, slayer))
            # print(hlsize)
            solver = ['adam', 'lbfgs']
            activation = ['relu', 'tanh']
            # Read: "https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html"
            # Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by
            # constraining the size of the weights.Increasing alpha may fix high variance (a sign of overfitting) by
            # encouraging smaller weights, resulting in a decision boundary plot that appears with lesser
            # curvatures.Similarly, decreasing alpha may fix high bias (a sign of underfitting) by encouraging larger
            # weights, potentially resulting in a more complicated decision boundary.
            # please look at this while tunning alpha -- the regularization param https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
            alpha = [0.1, 1, 10, 100]
            # default is 0.0001, # Experiment with this three values.. higher the alpha, simpler the boundary
            learning_rate = ['adaptive']
            # Create the random grid
            param_grid = {'hidden_layer_sizes': hlsize, 'activation': activation, 'solver': solver, 'alpha': alpha,
                          'learning_rate': learning_rate}
            model = MLPClassifier(max_iter=500, random_state=AuthParams.SEED, early_stopping=True)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=self.scorer_hter)
            optimal_model.fit(x_train, y_train)
            hlayers = optimal_model.best_params_['hidden_layer_sizes']
            activation = optimal_model.best_params_['activation']
            solver = optimal_model.best_params_['solver']
            alpha = optimal_model.best_params_['alpha']
            learning_rate = optimal_model.best_params_['learning_rate']
            # print(f'hlayers: {hlayers}, activation: {activation}, solver: {solver}, alpha: {alpha}, learning_rate: {learning_rate}')
            self.final_model = MLPClassifier(max_iter=500, hidden_layer_sizes=hlayers, activation=activation,
                random_state=AuthParams.SEED, solver=solver, alpha=alpha, learning_rate=learning_rate)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = MLPClassifier()
            self.final_model.fit(x_train, y_train)

    def train_svm(self, x_train, y_train):
        # print('Training SVM')
        if AuthParams.HYPER_TUNE:
            CVals = [1, 10, 100, 1000, 10000]
            Gammas = ['scale']
            Kernels = ['linear', 'poly', 'rbf', 'sigmoid', ]
            param_grid = {'C': CVals, 'gamma': Gammas, 'kernel': Kernels}
            model = svm.SVC(random_state=AuthParams.SEED, tol=1e-5)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=self.scorer_hter)
            optimal_model.fit(x_train, y_train)
            cval = optimal_model.best_params_['C']
            gamma = optimal_model.best_params_['gamma']
            kernel = optimal_model.best_params_['kernel']
            # print(f'cval: {cval}, gamma: {gamma}, kernel: {kernel}')
            self.final_model = svm.SVC(C=cval, gamma=gamma, kernel=kernel, random_state=AuthParams.SEED, tol=1e-5)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = svm.SVC(probability=True, C=1000)
            self.final_model.fit(x_train, y_train)
