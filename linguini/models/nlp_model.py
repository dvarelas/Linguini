class NLPmodel(object):
    """
    Abstract class
    """
    def __init__(self):
        None

    def fit(self):
        raise NotImplementedError('This is an abstract method.')

    def predict(self):
        raise NotImplementedError('This is an abstract method.')
