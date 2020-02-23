import numpy as np
from bert import tokenization
import tensorflow_hub as hub


class SentenceEncoder(object):
    """
    Abstract class
    """
    def __init__(self):
        None

    def fit(self, sentences):
        raise NotImplementedError('This is an abstract method.')

    def transform(self, sentences):
        raise NotImplementedError('This is an abstract method.')


class BertEncoder(SentenceEncoder):
    """
    Encodes sentences by using the indexer of Bert
    """
    def __init__(self, model):
        self.bert_layer = hub.KerasLayer(model['name'], trainable=model['trainable'])
        self.tokenizer = tokenization.FullTokenizer(
            self.bert_layer.resolved_object.vocab_file.asset_path.numpy(), do_lower_case=False)
        super().__init__()

    def fit(self, sentences):
        """
        Gets the maximum from the lengths of all the sentences
        :param sentences: Array of sentences
        :return: Max length
        """
        max_len = len(max(sentences.flatten(), key=len)) + 50
        return max_len

    def transform(self, sentences):
        """
        Transforms sentences by indexing them

        :param sentences: Array of sentences
        :return: Indexed sentences
        """
        max_len = self.fit(sentences)
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in sentences:
            text = text[:max_len - 2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return [np.array(all_tokens), np.array(all_masks), np.array(all_segments)]


class ColumnEncoder(object):
    """
    Indexes columns in a pandas dataframe
    """
    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self):
        return None

    def transform(self, df):
        """
        Indexes columns in a pandas dataframe

        :param df: Pandas dataframe
        :return:
        """
        result = []
        for col in self.col_names:
            result.append(df[col + '_indexed'].values)
        return result
