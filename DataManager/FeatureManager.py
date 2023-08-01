import os
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

from Param import Params


class FMLoader:
    def __init__(self, dataset='BBMAS', users=None):
        self.fmlocation = os.path.join(os.path.dirname(os.getcwd()), 'DataFiles', 'FeatureFiles', dataset)
        self.all_users = [id for id in range(1, len(os.listdir(self.fmlocation))+1)]
        if users is None:
            self.experimental_users = self.all_users[:] # by default all users are experimental users, they can change by calling the set_experimental_users
        else:
            self.experimental_users  = users
        self.combine_feature_files()

    def set_dataset(self, dataset):
        self.fmlocation = os.path.join(os.path.dirname(os.getcwd()), 'DataFiles', 'FeatureFiles', dataset)

    def set_experimental_users(self, users):
        self.experimental_users = users
    def combine_feature_files(self): # combine all users, you can cut if you need to in the get data methods
        ordered_file_names = ['User' + str(user_id) + '.csv' for user_id in self.all_users]
        # print(f' combining the files for all users: {ordered_file_names}')
        all_user_frames = []
        for file_name in ordered_file_names:
            user_id = int(file_name[4:-4])  # user id as numbers!
            # print(f'user id: {user_id}----> user file name: {file_name}')
            file_path = os.path.join(self.fmlocation, file_name)  # create the path for the file
            current_user_fvs = pd.read_csv(file_path, index_col=0)  # read the data from the file
            current_user_fvs = current_user_fvs[
                current_user_fvs['faulty'] == False]  # discard all the faulty feature vectors
            # print('current_user_fvs.shape', current_user_fvs.shape)
            current_user_fvs = current_user_fvs.drop(['faulty'],
                axis=1)  # drop the faulty column because we cant use it as a feature
            # print('current_user_fvs.shape', current_user_fvs.shape)
            current_user_fvs['user_id'] = user_id  # appending the user id to the corresponding feature matrix
            # print('after adding the user id current_user_fvs.shape', current_user_fvs.shape)
            # print(all_swipes.head(20).to_string())
            all_user_frames.append(current_user_fvs)  # creating a list of all the data frames  # print(current_user_fvs.head().to_string())
        self.combined_df = pd.concat(all_user_frames)  # concatinating all the frames vertically

    def select_features(self):
        # this function will change the main data frame i.e. the combined_df
        # write code for feature selection here!! everything else remain the same bro!!!
        pass
    def get_data_for_identification_exp(self):
        # getting the experimental users only
        print(f'selecting only following users for identification experiments: ', self.experimental_users)
        self.final_identification_df = self.combined_df[self.combined_df['user_id'].isin(self.experimental_users)]
        # Split the features and labels
        # Drop the 'user_id' column which represent the feature matrix
        y = self.final_identification_df['user_id']  # classs labels
        X = self.final_identification_df.drop('user_id', axis=1, inplace=False)
        # Assign 'user_id' column to y the labels (user_ids)
        # Splitting the dataset into Training and Testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Params.TRAIN_PERCENT, random_state=Params.SEED)
        return X_train, X_test, y_train, y_test

    def get_data_for_auth_exp(self, user_id):
       # Split the features and labels
       # Drop the 'user_id' column which represent the feature matrix
        genuine = self.combined_df[self.combined_df['user_id'] == user_id]
        print(genuine.shape)
        impostors = self.combined_df[self.combined_df['user_id'] != user_id]
        print(impostors.shape)

        genuine = genuine.drop(['user_id'], axis=1)
        print(genuine.shape)
        impostors = impostors.drop(['user_id'], axis=1)
        print(impostors.shape)

        gen_train, gen_test = train_test_split(genuine, train_size=Params.TRAIN_PERCENT, random_state=Params.SEED)
        imp_train, imp_test = train_test_split(impostors, train_size=Params.TRAIN_PERCENT, random_state=Params.SEED)

        return gen_train, imp_train, gen_test, imp_test


if __name__ == "__main__":
    FML = FMLoader('BBMAS')
    FML.set_experimental_users([i for i in range(1, 10)])
    X_train, X_test, y_train, y_test = FML.get_data_for_identification_exp()
    print('X_train, X_test, y_train, y_test: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    for user_id in range(1, len(FML.all_users)):
        gen_train, imp_train, gen_test, imp_test = FML.get_data_for_auth_exp(user_id)
        print('gen_train, imp_train, gen_test, imp_test: ', gen_train.shape, imp_train.shape, gen_test.shape,
            imp_test.shape)
