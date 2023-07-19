from Identification import IdentificationExp
import warnings
warnings.filterwarnings("ignore")
Exp1 = IdentificationExp('BBMAS')
Exp1.prepare_data()
Exp1.run_classifier("LRG")