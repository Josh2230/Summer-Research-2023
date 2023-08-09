from sklearn import svm, linear_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from Param import IdParams
from Param.IdParams import LOSS_FUNC, CVAL


class TrainIdentifier:
    # the data could be made
    def __init__(self, classifier=None):  # default Random Forest
        self.classifier = classifier
        self.final_model = None

    def set_classifier(self, classifier):
        self.classifier = classifier

    def train(self, x_train, y_train):
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
        elif self.classifier == 'GNB':
            self.train_gnb(x_train, y_train)
        else:
            raise ValueError('Unknown classifier!')
    def get_preds(self, samples):
        predictions = self.final_model.predict(samples)
        return predictions
    def get_scores(self, samples):
        scores = self.final_model.predict_proba(samples)
        return scores

    def train_raf(self, x_train, y_train):
        # print('Training RAF')
        if IdParams.HYPER_TUNE:
            n_estimators = [int(item) for item in range(500, 501, 100)]
            max_depth = [int(item) for item in range(2, 4, 1)]
            min_samples_leaf = [int(item) for item in range(2, 3, 1)]
            max_features = [x / 100 for x in range(30, 51, 30)]  # need to remain float for percentages
            # max_features = ['sqrt', 'log2']
            # Create the random grid
            param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                          'min_samples_leaf': min_samples_leaf}
            model = RandomForestClassifier(n_jobs=-1, random_state=IdParams.SEED)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = CVAL, scoring=LOSS_FUNC)
            optimal_model.fit(x_train, y_train)
            n_estimators = optimal_model.best_params_['n_estimators']
            max_features = optimal_model.best_params_['max_features']
            max_depth = optimal_model.best_params_['max_depth']
            min_samples_leaf = optimal_model.best_params_['min_samples_leaf']
            # print(f'n_estimators: {n_estimators}, max_features: {max_features}, max_depth:{max_depth}, min_samples_leaf:{min_samples_leaf}')
            self.final_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            max_features=max_features, max_leaf_nodes=min_samples_leaf, n_jobs=-1,random_state=IdParams.SEED)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = RandomForestClassifier(n_estimators=200)
            self.final_model.fit(x_train, y_train)

    def train_knn(self, x_train, y_train):
        if IdParams.HYPER_TUNE:
            # print('Tunning KNN...')
            n_neighbors = [int(x) for x in range(7, 10, 2)]
            dist_met = ['manhattan', 'euclidean', 'cosine']
            # create the random grid
            param_grid = {'n_neighbors': n_neighbors, 'metric': dist_met}
            model = KNeighborsClassifier()
            # scoring_function = 'f1'
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = CVAL, scoring=LOSS_FUNC)
            # loss function!
            optimal_model.fit(x_train, y_train)
            best_nn = optimal_model.best_params_['n_neighbors']
            best_dist = optimal_model.best_params_['metric']
            # print(f'best_nn: {best_nn}, best_dist: {best_dist}')
            self.final_model = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
            self.final_model.fit(x_train, y_train)

    def train_lrg(self, x_train, y_train):
        # print('Training LRG')
        if IdParams.HYPER_TUNE:
            param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                          'C': [100, 100, 10, 1.0, 0.1, 0.01], 'penalty': ['l1', 'l2', 'elasticnet']}
            model = linear_model.LogisticRegression(random_state=IdParams.SEED, tol=1e-5)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = CVAL, scoring=LOSS_FUNC)
            optimal_model.fit(x_train, y_train)
            solver = optimal_model.best_params_['solver']
            cval = optimal_model.best_params_['C']
            penalty = optimal_model.best_params_['penalty']
            # print(f'solver: {solver}, cval: {cval}, penalty: {penalty}')
            self.final_model = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty,
                random_state=IdParams.SEED, tol=1e-5)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = linear_model.LogisticRegression(C=1000, solver ='newton-cg')
            self.final_model.fit(x_train, y_train)

    def train_mlp(self, x_train, y_train):
        # print('Training MLP')
        if IdParams.HYPER_TUNE:
            hlsize = []
            for flayer in range(50, 101, 50):
                # hlsize.append((flayer,))
                for slayer in range(25, 51, 25):
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
            model = MLPClassifier(max_iter=500, random_state=IdParams.SEED, early_stopping=True)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=LOSS_FUNC)
            optimal_model.fit(x_train, y_train)
            hlayers = optimal_model.best_params_['hidden_layer_sizes']
            activation = optimal_model.best_params_['activation']
            solver = optimal_model.best_params_['solver']
            alpha = optimal_model.best_params_['alpha']
            learning_rate = optimal_model.best_params_['learning_rate']
            # print(f'hlayers: {hlayers}, activation: {activation}, solver: {solver}, alpha: {alpha}, learning_rate: {learning_rate}')
            self.final_model = MLPClassifier(max_iter=500, hidden_layer_sizes=hlayers, activation=activation,
                random_state=IdParams.SEED, solver=solver, alpha=alpha, learning_rate=learning_rate)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = MLPClassifier()
            self.final_model.fit(x_train, y_train)

    def train_svm(self, x_train, y_train):
        # print('Training SVM')
        if IdParams.HYPER_TUNE:
            CVals = [1, 10, 100, 1000, 10000]
            Gammas = ['scale']
            Kernels = ['linear', 'poly', 'rbf', 'sigmoid', ]
            param_grid = {'C': CVals, 'gamma': Gammas, 'kernel': Kernels}
            model = svm.SVC(random_state=IdParams.SEED, tol=1e-5, probability=True, max_iter = 200000)
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = CVAL, scoring=LOSS_FUNC)
            optimal_model.fit(x_train, y_train)
            cval = optimal_model.best_params_['C']
            gamma = optimal_model.best_params_['gamma']
            kernel = optimal_model.best_params_['kernel']
            # print(f'cval: {cval}, gamma: {gamma}, kernel: {kernel}')
            self.final_model = svm.SVC(C=cval, gamma=gamma, kernel=kernel, random_state=IdParams.SEED, tol=1e-5, probability=True, max_iter = 200000)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = svm.SVC(probability=True, C=1000)
            self.final_model.fit(x_train, y_train)

    def train_gnb(self, x_train, y_train):
        # print('Training Gaussian Naive Bayes--best baseline!')
        if IdParams.HYPER_TUNE:
            var_smoothing = [1e-09, 1e-07, 1e-05]
            param_grid = {'var_smoothing': var_smoothing}
            model = GaussianNB()
            optimal_model = GridSearchCV(estimator=model, param_grid=param_grid, cv = CVAL, scoring=LOSS_FUNC)
            optimal_model.fit(x_train, y_train)
            var_smoothing = optimal_model.best_params_['var_smoothing']
            # print(f'var_smoothing: {var_smoothing}')
            self.final_model = GaussianNB(var_smoothing=var_smoothing)
            self.final_model.fit(x_train, y_train)  # setting the final model that will be used for prediction
        else:
            self.final_model = GaussianNB()
            self.final_model.fit(x_train, y_train)