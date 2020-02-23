import pandas as pd
import time
import sys

from linguini import BASE_PATH
from linguini.preprocessing.sentence import PandasTweetPreprocessor
from linguini.preprocessing.encoders import BertEncoder
from linguini.models.classifiers import BertBinaryClassifier
from linguini.utils.logging import TrainingLogger
from linguini.config.tweet_binary_classifier_config import config


def main(configuration):
    # get the training data
    train = pd.read_csv(configuration.data_path)
    print(train.shape)

    # instantiate the pandas tweet preprocessor
    p = PandasTweetPreprocessor(
        language=configuration.language,
        text_col=configuration.text_col,
        cols_to_index=['keyword'])

    # preprocess the dataframe and encode the tweets
    start_time = time.time()
    elements_df = p.transform(train, multiproc=configuration.multiprocessing)
    enc = BertEncoder(configuration.optimization['model'])
    print('Preprocessing and sentence encoding took: ' + str(time.time() - start_time) + ' secs')

    # get the training data
    max_len = enc.fit(elements_df['cleaned_' + configuration.text_col].values)
    train_inputs = enc.transform(elements_df['cleaned_' + configuration.text_col].values)

    train_inputs_side_features = tuple(train_inputs)

    # instantiate the classifier
    model = BertBinaryClassifier(
        configuration.optimization['optimization'],
        enc.bert_layer,
        max_len)

    # fit the model
    model.fit(train_inputs_side_features, train['target'].values)

    # save the model
    model.save()


if __name__ == '__main__':
    exp = BASE_PATH + 'logs/bert_model.txt'

    sys.stdout = TrainingLogger(exp)
    main(config)

