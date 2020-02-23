import preprocessor as preproc
import spacy
import pandas as pd
from pandarallel import pandarallel
import multiprocessing

from linguini.utils.indexers import ColumnIndexer


class SentencePreprocessor(object):
    """
    Abstract class
    """
    def __init__(self):
        None

    @staticmethod
    def _parse(text):
        raise NotImplementedError('This is an abstract method.')

    def _cleaner(self, text):
        raise NotImplementedError('This is an abstract method.')

    @staticmethod
    def tokenize(text):
        raise NotImplementedError('This is an abstract method.')

    def transform(self):
        raise NotImplementedError('This is an abstract method.')


class TweetSentencePreprocessor(SentencePreprocessor):
    """
    Class for preprocessing tweets
    """
    def __init__(self, language):
        self.language = language
        self.sp = spacy.load(language)
        super().__init__()

    def _cleaner(self, text):
        """
        Cleans the text by performing lowercasing, whitespace removal,
        stopword removal, punctuation removal and lemmatization
        :param text:
        :return:
        """
        cleaned = preproc.clean(text)
        return [
            word.lemma_.lower().strip() for word in self.sp(cleaned) if word.is_stop is False if word.is_punct is False]

    def tokenize(self, text):
        """
        Tokenizes text

        :param text:
        :return:
        """
        return [word.text for word in self.sp(text)]

    @staticmethod
    def _parse(text):
        """
        Parses elements from tweets

        :param text:
        :return:
        """
        return preproc.parse(text)

    def collect_parsed(self, text):
        """
        Collects all the parsed elements in dicts of emojis,
        urls, mentions, hashtags, smileys, numbers and reserved words

        :param text:
        :return:
        """
        elements = {}
        parsed_text = self._parse(text)
        if parsed_text.urls is not None:
            elements['urls'] = [x.match for x in parsed_text.urls]
        else:
            elements['urls'] = None
        if parsed_text.emojis is not None:
            elements['emojis'] = [x.match for x in parsed_text.emojis]
        else:
            elements['emojis'] = None
        if parsed_text.smileys is not None:
            elements['smileys'] = [x.match for x in parsed_text.smileys]
        else:
            elements['smileys'] = None
        if parsed_text.numbers is not None:
            elements['numbers'] = [x.match for x in parsed_text.numbers]
        else:
            elements['numbers'] = None
        if parsed_text.hashtags is not None:
            elements['hashtags'] = [x.match for x in parsed_text.hashtags]
        else:
            elements['hashtags'] = None
        if parsed_text.mentions is not None:
            elements['mentions'] = [x.match for x in parsed_text.mentions]
        else:
            elements['mentions'] = None
        if parsed_text.reserved_words is not None:
            elements['reserved_words'] = [x.match for x in parsed_text.reserved_words]
        else:
            elements['reserved_words'] = None

        return elements

    def transform(self, **kwargs):
        raise NotImplementedError('This is an abstract method.')


class PandasTweetPreprocessor(TweetSentencePreprocessor):
    """
    Preprocessor for tweets that live in a pandas dataframe
    """
    def __init__(self, language, text_col, cols_to_index=None):
        self.cols_to_index = cols_to_index
        self.language = language
        self.text_col = text_col
        super().__init__(language)

    def collect_parsed(self, text):
        """
        Parses tweet elements
        :param text:
        :return: Pandas series
        """
        return pd.Series(super().collect_parsed(text))

    def parse_multiple(self, df, multiproc=False):
        """
        Parses elements and puts them in a dataframe
        :param df: Pandas dataframe
        :param multiproc: Boolean. True activates multiprocessing
        :return:
        """
        if multiproc:
            pandarallel.initialize()
            elements_df = df[self.text_col].parallel_apply(self.collect_parsed)
        else:
            elements_df = df[self.text_col].apply(self.collect_parsed)
        multiple_elements_df = pd.concat([df, elements_df], axis=1)
        return multiple_elements_df

    def index_columns(self, df, col_names):
        """
        Indexes categorical columns

        :param df: Pandas dataframe
        :param col_names: Columns to index

        :return:
        """
        if col_names is not None:
            indexer = ColumnIndexer(df, col_names)
            return indexer.transform(df)
        else:
            return df

    def transform(self, df, multiproc=False):
        """
        Transforms a dataframe

        :param df: Pandas dataframe
        :param multiproc: Boolean. If true multiprocessing is activated
        :return:
        """
        elements_df = self.parse_multiple(df, multiproc=False)
        if multiproc:
            pandarallel.initialize(nb_workers=multiprocessing.cpu_count()-1)
            elements_df['cleaned_' + self.text_col] = elements_df[self.text_col].parallel_apply(self._cleaner)
        else:
            elements_df['cleaned_' + self.text_col] = elements_df[self.text_col].apply(self._cleaner)
        indexed_df = self.index_columns(elements_df, self.cols_to_index)
        return indexed_df








