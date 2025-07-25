from confidence import Confidence

def get_bounds_from_confidence_list(confidence_list: list[Confidence]):

    lower = [conf.lower_proba.item() for conf in confidence_list]
    upper = [conf.upper_proba.item() for conf in confidence_list]

    return lower, upper