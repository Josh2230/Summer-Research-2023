SEED = 1987
TRAIN_PERCENT = .8
HYPER_TUNE = False
BAYES_LOSS_FUNC = 'f1'
GRID_LOSS_FUNC = 'self.scorer_hter'
SELECT_FEATURES = True
SELECT_FEATURES_PERCENT = 1
SCALE_FEATURES = True
SCALING_METHOD = 'MINMAX'
DEFAULT_CLS = 'KNN'
MCC = ['RAF', 'SVM', 'MLP', 'LRG', 'KNN']
OCC = ['SVM1', 'LOF', 'ISF', 'ELE']
AUTOCLS = ['LZP']
SeparateTrainTest = ['NOTHING']
CombinedTrainTest = ['ANTAL', 'BBMAS']
