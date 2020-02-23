import pandas as pd

from linguini import BASE_PATH
from linguini.preprocessing.sentence import PandasTweetPreprocessor
from linguini.preprocessing.encoders import BertEncoder
from linguini.models.classifiers import BertBinaryClassifier
from linguini.config.tweet_binary_classifier_config import config


def main(configuration):
    p = PandasTweetPreprocessor(
        language=configuration.language,
        text_col=configuration.text_col)

    test_data = pd.read_csv(configuration.test_data_path)

    elements_df = p.transform(test_data, multiproc=configuration.multiprocessing)
    enc = BertEncoder(configuration.optimization['model'])

    max_len = enc.fit(elements_df['cleaned_' + configuration.text_col].values)
    test_inputs = enc.transform(elements_df['cleaned_' + configuration.text_col].values)

    model = BertBinaryClassifier(
        configuration.optimization['optimization'],
        enc.bert_layer,
        max_len + 1)

    predictions = model.score(
        test_inputs, BASE_PATH + 'models/bert_model.h5')

    test_data['target'] = predictions.round().astype(int)
    submission = test_data.loc[:, ['id', 'target']]
    print(submission.head())

    submission.to_csv(BASE_PATH + 'predictions/submission.csv', index=False)


if __name__ == '__main__':
    main(config)
