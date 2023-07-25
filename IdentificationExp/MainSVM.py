from ClassifiersTuning import IdentificationExp
import warnings
warnings.filterwarnings("ignore")
Exp1 = IdentificationExp('BBMAS')
user_ids = [id for id in range(1, 10)]
print(user_ids)
Exp1.prepare_data(user_ids)
Exp1.run_classifier("SVM")