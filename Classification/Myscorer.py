from sklearn.metrics import confusion_matrix, make_scorer

def scoring_hter(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hter = (far + frr) / 2
    if hter < 0:
        raise ValueError('Errror: hter cant be negative')
    return hter


def scoring_far(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    if far < 0:
        raise ValueError('Errror: hter cant be negative')
    return far


def scoring_frr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    frr = fn / (fn + tp)
    if frr < 0:
        raise ValueError('Errror: hter cant be negative')
    return frr


scorer_hter = make_scorer(scoring_hter, greater_is_better=False)
scorer_far = make_scorer(scoring_far, greater_is_better=False)
scorer_frr = make_scorer(scoring_frr, greater_is_better=False)