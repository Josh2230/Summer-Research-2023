from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from Classification import Myscorer
SEED = 1987
CVAL = 5
TRAIN_PERCENT = .5
HYPER_TUNE = True
LOSS_FUNC = 'accuracy'
SELECT_FEATURES = True
FEAT_SELECT_BASIS = mutual_info_classif
if SELECT_FEATURES:
    SELECT_FEATURES_PERCENT = .8
SMOTE_DATA = True
SCALE_FEATURES = True
SCALING_METHOD = 'MINMAX'



