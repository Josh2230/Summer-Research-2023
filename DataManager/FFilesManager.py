import os
import pandas as pd

def combine_feature_files(folder_location):
    ''''This function takes a location, which will have user files which will be combined into one giant data frame'''
    files = os.listdir(folder_location)
    combined_frames = []
    # file_names = ['user'+str(user_id)+'.csv' for user_id in range(1, len(files)+1)]
    for file in files:
        user_id = int(file[4:-4])  # user id as numbers! assuming User<>.csv/.txt
        file_location = os.path.join(folder_location, file)  # create the path for the file
        # depending on whether you have a header or not or index column or not index_col=0, header=0 changes
        cuser_fvs = pd.read_csv(file_location, index_col=0, header=0)  # read the data from the file
        cuser_fvs.columns = cuser_fvs.columns.astype(str)

        cuser_fvs = cuser_fvs[cuser_fvs['faulty'] == False]  # discard all the faulty feature vectors
        cuser_fvs = cuser_fvs.drop(['faulty'], axis=1)  # drop the faulty column because we cant use it as a feature

        cuser_fvs['user_id'] = user_id  # appending the user id to the corresponding feature matrix
        combined_frames.append(cuser_fvs)
    combined_df = pd.concat(combined_frames)  # concatinating all the frames vertically
    combined_df.reset_index(drop=True, inplace=True)  # resettting the index and dropping the old one
    return combined_df