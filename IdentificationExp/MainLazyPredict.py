import numpy as np

from VerificationMethods import IdentificationExp
import warnings
warnings.filterwarnings("ignore")

Exp1 = IdentificationExp('BBMAS')
user_ids = [id for id in range(1, 10)]
print(user_ids)
Exp1.prepare_data(specific_user_ids = user_ids)
Exp1.run_classifier(classifier_name="LazyPredict")