import preprocessor as preproc
import spacy
import pandas as pd
from pandarallel import pandarallel
import multiprocessing


class SentencePreprocessor(object):
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
    def __init__(self, language):
        self.language = language
        self.sp = spacy.load(language)
        super().__init__()

    def _cleaner(self, text):
        cleaned = preproc.clean(text)
        return [
            word.lemma_.lower().strip() for word in self.sp(cleaned) if word.is_stop is False if word.is_punct is False]

    def tokenize(self, text):
        return [word.text for word in self.sp(text)]

    @staticmethod
    def _parse(text):
        return preproc.parse(text)

    def collect_parsed(self, text):
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
    def __init__(self, language, text_col):
        self.language = language
        self.text_col = text_col
        super().__init__(language)

    def collect_parsed(self, text):
        return pd.Series(super().collect_parsed(text))

    def parse_multiple(self, df, multiproc=False):
        if multiproc:
            pandarallel.initialize()
            elements_df = df[self.text_col].parallel_apply(self.collect_parsed)
        else:
            elements_df = df[self.text_col].apply(self.collect_parsed)
        multiple_elements_df = pd.concat([df, elements_df], axis=1)
        return multiple_elements_df

    def transform(self, df, multiproc=False):
        elements_df = self.parse_multiple(df, multiproc=False)
        if multiproc:
            pandarallel.initialize(nb_workers=multiprocessing.cpu_count()-1)
            elements_df['cleaned_' + self.text_col] = elements_df[self.text_col].parallel_apply(self._cleaner)
        else:
            elements_df['cleaned_' + self.text_col] = elements_df[self.text_col].apply(self._cleaner)
        return elements_df








