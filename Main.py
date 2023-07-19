from Identification.Identification import IdentificationExp
import warnings
warnings.filterwarnings("ignore")
# local testing
Exp1 = IdentificationExp('BBMAS')
print('Preparing data...')
Exp1.prepare_data()
print('Running classifier...')
# Exp1.run_classifier("kNN")
# Exp1.run_classifier("RAF")
Exp1.run_classifier("SVM")