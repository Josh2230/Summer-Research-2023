# We will evaluate identification performance via this script!
import os
import pandas as pd
class IdentificationExp:
    def __init__(self, feature_path):
        self.feature_path = feature_path
    def prepare_data(self):
        feature_files = os.listdir(self.feature_path)
        ordered_file_names = ['User'+str(user_id)+'.csv' for user_id in range(1, len(feature_files)+1)]
        dataframes = []
        for file_name in ordered_file_names:
            user_id = int(file_name[4:-4])
            print(f'user id: {user_id}----> user file name: {file_name}')
            file_path = os.path.join(self.feature_path, file_name)
            all_swipes = pd.read_csv(file_path, index_col=0)
            all_swipes = all_swipes[all_swipes['faulty']==False]
            all_swipes = all_swipes.drop(['faulty'], axis=1)
            all_swipes['user_id'] = user_id
            # print(all_swipes.head(20).to_string())
            dataframes.append(all_swipes) # creating a list of all the data frames
            print(all_swipes.shape)
        self.combined_df = pd.concat(dataframes) # concating all the frames
        print('combined_df.shape', self.combined_df.shape)
    def run_classifier(self):
        X = self.combined_df[:-1]
        y = self.combined_df[-1]
        print(X.shape)
        print(y.shape)

# local testing
Exp1 = IdentificationExp(r'/Users/rk042/PycharmProjects/Summer-Research-2023/FeatureFiles/BBMAS')
Exp1.prepare_data()
Exp1.run_classifier()