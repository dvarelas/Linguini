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
    p = PandasTweetPreprocessor(
        language=configuration.language,
        text_col=configuration.text_col,
        cols_to_index=['keyword'])

    train = pd.read_csv(configuration.data_path)
    print(train.shape)

    start_time = time.time()
    elements_df = p.transform(train, multiproc=configuration.multiprocessing)
    enc = BertEncoder(configuration.optimization['model'])
    print('Preprocessing and sentence encoding took: ' + str(time.time() - start_time) + ' secs')

    max_len = enc.fit(elements_df['cleaned_' + configuration.text_col].values)
    train_inputs = enc.transform(elements_df['cleaned_' + configuration.text_col].values)

    train_inputs_side_features = tuple(train_inputs)

    model = BertBinaryClassifier(
        configuration.optimization['optimization'],
        enc.bert_layer,
        max_len)

    model.fit(train_inputs_side_features, train['target'].values)

    model.save()


if __name__ == '__main__':
    exp = BASE_PATH + 'logs/bert_model.txt'

    sys.stdout = TrainingLogger(exp)
    main(config)

