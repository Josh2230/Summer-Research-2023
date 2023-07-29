# We will evaluate identification performance via this script!
import os
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skopt import BayesSearchCV
from sklearn import linear_model
from lazypredict.Supervised import LazyClassifier


from Param import Params


class IdentificationExp:
    def __init__(self, dataset_name):
        self.feature_path = os.path.join(os.path.dirname(os.getcwd()), 'FeatureFiles', dataset_name)
        # print(f'using features files located at {self.feature_path} for classification')
    def set_dataset_path(self, datasetname):
        self.feature_path = os.path.join(os.path.dirname(os.getcwd()), 'FeatureFiles', datasetname)

    def prepare_data(self, specific_user_ids=None):
        ''''
        '''
        feature_files = os.listdir(self.feature_path)
        if specific_user_ids is None: # take all the users if no specified users
            ordered_file_names = ['User' + str(user_id) + '.csv' for user_id in range(1, len(feature_files) + 1)]
        else: # take only specified users
            ordered_file_names = ['User' + str(user_id) + '.csv' for user_id in specific_user_ids] # 5, 6, 7,

        print(f'Preparing data.... and users included: {ordered_file_names}')
        dataframes = []
        for file_name in ordered_file_names:
            user_id = int(file_name[4:-4]) # user id as numbers!
            # print(f'user id: {user_id}----> user file name: {file_name}')
            file_path = os.path.join(self.feature_path, file_name) # create the path for the file
            current_user_fvs = pd.read_csv(file_path, index_col=0) # read the data from the file
            current_user_fvs = current_user_fvs[current_user_fvs['faulty'] == False] # discard all the faulty feature vectors
            # print('current_user_fvs.shape', current_user_fvs.shape)
            current_user_fvs = current_user_fvs.drop(['faulty'], axis=1) # drop the faulty column because we cant use it as a feature
            # print('current_user_fvs.shape', current_user_fvs.shape)
            current_user_fvs['user_id'] = user_id  # appending the user id to the corresponding feature matrix
            # print('after adding the user id current_user_fvs.shape', current_user_fvs.shape)
            # print(all_swipes.head(20).to_string())
            dataframes.append(current_user_fvs)  # creating a list of all the data frames
            # print(current_user_fvs.head().to_string())
        self.combined_df = pd.concat(dataframes)  # concatinating all the frames vertically

        # print(self.combined_df.to_string())

    def run_knn(self):
        ''''
        '''
        n_neighbors = [int(x) for x in range(1, 2)] #1 NN
        # print('n_neighbors',n_neighbors)
        # dist_met = ['manhattan', 'euclidean']
        dist_met = ['cosine']
        weights = ['uniform', 'distance']
        leaf_size = [20]
        # Define the hyperparameter grid to search

        # create the random grid of hyper-parameters
        param_grid = {'n_neighbors': n_neighbors, 'metric': dist_met, 'leaf_size':leaf_size, 'weights': weights}

        # creating a classifier object
        model = KNeighborsClassifier()
        # setting up the parameter search using bayes search cv
        optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
        # fitting the training data to the model
        optimal_model.fit(self.X_train, self.y_train)
        # accessing the best values of the hyperparameters
        best_nn = optimal_model.best_params_['n_neighbors']
        best_dist = optimal_model.best_params_['metric']
        best_leaf_size = optimal_model.best_params_['leaf_size']
        best_weight  = optimal_model.best_params_['weights']

        # Retraining the model again | this is not necessary, just my paranoia
        final_model = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist, weights=best_weight, leaf_size=best_leaf_size)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(f'rank_k_acc: {rank_k_acc} for {self.classifier} with {best_nn} neighbors and {best_dist} distance metric')

    def run_raf(self):
        ''''
        '''
        # creating parameter grid
        n_estimators = [int(item) for item in range(500, 1001, 200)]
        max_depth = [int(item) for item in range(2, 4, 1)]
        min_samples_leaf = [int(item) for item in range(2, 3, 1)]
        max_features = [x / 100 for x in range(30, 51, 20)]  # need to remain float for percentages
        # max_features = ['sqrt', 'log2']
        # Create the param grid
        param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf}
        # creating a classifier object
        model = RandomForestClassifier(n_jobs=-1, random_state=Params.SEED)
        # setting up the parameter search using bayes search cv
        optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
        # fitting the training data to the model
        optimal_model.fit(self.X_train, self.y_train)
        # accessing the best values of the hyperparameters
        n_estimators = optimal_model.best_params_['n_estimators']
        max_features = optimal_model.best_params_['max_features']
        max_depth = optimal_model.best_params_['max_depth']
        min_samples_leaf = optimal_model.best_params_['min_samples_leaf']
        # Retraining the model again | this is not necessary, just my paranoia
        final_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                             max_features=max_features, max_leaf_nodes=min_samples_leaf, n_jobs=-1,
                                             random_state=Params.SEED)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with n_estimators: {n_estimators}, max_features: {max_features}, max_depth: {max_depth}, neighbors and min_samples_leaf: {min_samples_leaf}')

    def run_svm(self):
        ''''
        '''
        # creating parameter grid
        CVals = [0.02, 0.04, 0.08, 0.16, 0.32]
        Gammas = ['scale']
        Kernels = ['linear', 'rbf']
        param_grid = {'C': CVals, 'gamma': Gammas, 'kernel': Kernels}
        # creating a classifier object
        model = svm.SVC(random_state=Params.SEED, tol= 1e-5, probability=True)
        # setting up the parameter search using bayes search cv
        optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
        # fitting the training data to the model
        optimal_model.fit(self.X_train, self.y_train)
        # accessing the best values of the hyperparameters
        cval = optimal_model.best_params_['C']
        gamma = optimal_model.best_params_['gamma']
        kernel = optimal_model.best_params_['kernel']
        # Retraining the model again | this is not necessary, just my paranoia
        final_model = svm.SVC(C=cval, gamma=gamma,kernel=kernel, random_state=Params.SEED,tol=1e-5, probability=True)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with cval: {cval}, gamma: {gamma}, kernel: {kernel}')

    def run_mlp(self):
        ''''
        '''
        # creating parameter grid
        hlsize = []
        for flayer in range(100, 301, 100):
            # hlsize.append((flayer,))
            for slayer in range(50, 101, 50):
                hlsize.append((flayer, slayer))
            # print(hlsize)
            # solver = ['adam', 'lbfgs']
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
        param_grid = {'hidden_layer_sizes': hlsize,
                      'activation': activation,
                      'solver': solver,
                      'alpha': alpha,
                      'learning_rate': learning_rate}
        # creating a classifier object
        model = MLPClassifier(max_iter=1000, random_state=Params.SEED, early_stopping=True)
        # setting up the parameter search using bayes search cv
        optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
        # fitting the training data to the model
        optimal_model.fit(self.X_train, self.y_train)
        # accessing the best values of the hyperparameters
        hlayers = optimal_model.best_params_['hidden_layer_sizes']
        activation = optimal_model.best_params_['activation']
        solver = optimal_model.best_params_['solver']
        alpha = optimal_model.best_params_['alpha']
        learning_rate = optimal_model.best_params_['learning_rate']
        # Retraining the model again | this is not necessary, just my paranoia
        final_model = MLPClassifier(max_iter=1000, hidden_layer_sizes=hlayers, activation=activation,
                                       random_state=Params.SEED, solver=solver, alpha=alpha,
                                       learning_rate=learning_rate)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(
            f'rank_k_acc: {rank_k_acc} for {self.classifier} with hlayers: {hlayers}, activation: {activation}, solver: {solver}, alpha: {alpha}, learning_rate: {learning_rate}')

    def run_lrg(self):
        ''''
        '''
        # creating parameter grid
        param_grid = {'solver': ['newton-cg'],
             'C': [0.1, 0.2, 0.4, 0.45, 0.5]}
        # creating a classifier object
        model = linear_model.LogisticRegression(random_state=Params.SEED, tol=1e-5)
        # setting up the parameter search using bayes search cv
        optimal_model = BayesSearchCV(estimator=model, search_spaces=param_grid, cv=5, scoring='accuracy')
        # fitting the training data to the model
        optimal_model.fit(self.X_train, self.y_train)
        # accessing the best values of the hyperparameters
        solver = optimal_model.best_params_['solver']
        cval = optimal_model.best_params_['C']
        penalty = optimal_model.best_params_['penalty']
        # Retraining the model again | this is not necessary, just my paranoia
        final_model = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty,random_state=Params.SEED, tol=1e-5)
        final_model.fit(self.X_train, self.y_train)
        pred_scores = final_model.predict_proba(self.X_test)
        rank_k_acc = {}
        for k in range(1, 4):
            rank_k_acc['rank' + str(k)] = top_k_accuracy_score(self.y_test, pred_scores, k=k)
        print(f'rank_k_acc: {rank_k_acc} for {self.classifier} with solver: {solver}, cval: {cval}, penalty: {penalty}')

    def run_lazypredictor(self):
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        lazy_model = LazyClassifier(predictions=True)
        models, predictions = lazy_model.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        print(models)

    def run_classifier(self, classifier_name):
        ''''
        '''
        # splitting the data into training and testing part
        # if we have multiple session data then we dont have to do this!
        from sklearn.model_selection import train_test_split
        # Split the features and labels
        # Drop the 'user_id' column which represent the feature matrix
        y = self.combined_df['user_id'] # classs labels
        X = self.combined_df.drop('user_id', axis=1, inplace=False)
        # Assign 'user_id' column to y the labels (user_ids)
        # Splitting the dataset into 70 training and 30 testing.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                random_state=Params.SEED)
        # print(self.X_train.shape)
        # print(self.X_test.shape)
        # print(self.X_train.to_string())
        # print(self.X_test.to_string())
        # # for the majority of the classifiers it is essential that we scale | MinMax has done better than Scaler,
        # # but we can see!
        mm_scaler = MinMaxScaler()
        self.X_train = mm_scaler.fit_transform(self.X_train)
        self.X_test = mm_scaler.transform(self.X_test)

        # for the majority of the classifiers it is essential that we scale | MinMax has done better than Scaler,
        # but we can see!
        # standard_scaler = StandardScaler()
        # # print(f'applying minmax scaler')
        # self.X_train = standard_scaler.fit_transform(self.X_train)
        # trainDF = pd.DataFrame(self.X_train)
        # print('self.X_train after scaler scaling', trainDF.to_string())
        # self.X_test = standard_scaler.transform(self.X_test)

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
        elif classifier_name == 'AutoSklearn':
            print(f'Running {self.classifier} classifier')
            self.run_autosklearn()
        elif classifier_name == 'LazyPredict':
            self.classifier = classifier_name
            print(f'Running {self.classifier} classifier')
            self.run_lazypredictor()
        elif classifier_name == 'AutoML':
            print(f'Running {self.classifier} classifier')
            self.run_automl()
        else:
            raise ValueError('Unknown classifier!')
