import yaml


from linguini import BASE_PATH


class TweetBinaryClassifierConfig(object):
    """
    Configuration for the tweet binary classifier application
    """
    def __init__(self, data_path, test_data_path, target_col, text_col, language, multiprocessing, optimization):
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.target_col = target_col
        self.text_col = text_col
        self.language = language
        self.multiprocessing = multiprocessing
        self.optimization = optimization


with open(BASE_PATH + "resources/configs/tweet_binary_classifier.yml", 'r') as ymlfile:
    yaml_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

config = TweetBinaryClassifierConfig(
    data_path=BASE_PATH + yaml_config['preproc']['data_path'],
    test_data_path=BASE_PATH + yaml_config['predict']['data_path'],
    target_col=yaml_config['preproc']['target_col'],
    text_col=yaml_config['preproc']['text_col'],
    language=yaml_config['preproc']['language'],
    multiprocessing=yaml_config['preproc']['multiprocessing'],
    optimization=yaml_config['fit'])
