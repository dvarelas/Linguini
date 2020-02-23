import tensorflow as tf
import os


from linguini import BASE_PATH
from linguini.utils.callbacks import CyclicLR


class BinaryClassifier(object):
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        raise NotImplementedError('This is an abstract method.')

    @staticmethod
    def score(test_inputs, model_path):
        raise NotImplementedError('This is an abstract method.')


class MultiLabelClassifier(object):
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        raise NotImplementedError('This is an abstract method.')

    @staticmethod
    def score(test_input, model_path):
        raise NotImplementedError('This is an abstract method.')


class BertBinaryClassifier(BinaryClassifier):
    def __init__(self, optim_setup, bert_layer, max_len):
        self.optim_setup = optim_setup
        self.bert_layer = bert_layer
        self.max_len = max_len
        self.model = self.compile()
        super().__init__()

    def compile(self):
        input_word_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]

        hidden_1 = tf.keras.layers.Dense(256, activation='relu')(clf_output)

        hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)

        out = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_2)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(
            tf.keras.optimizers.Adam(lr=float(self.optim_setup['base_lr'])),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model

    def fit(self, x, y):
        clr_exp = CyclicLR(
            mode='exp_range',
            step_size=(int(self.optim_setup['num_epochs']) / 2) * (6090 / int(self.optim_setup['batch_size'])),
            # multiplier should be between 2-10, number of examples is divided by the batch size
            max_lr=float(self.optim_setup['max_lr']),
            base_lr=float(self.optim_setup['min_lr']),
            gamma=0.9995)

        print(self.model.summary())
        train_history = self.model.fit(
            x, y,
            validation_split=0.2,
            epochs=int(self.optim_setup['num_epochs']),
            batch_size=int(float(self.optim_setup['batch_size'])),
            callbacks=[clr_exp]
        )
        return train_history

    def save(self):
        if not os.path.exists(BASE_PATH + 'models/'):
            os.makedirs(BASE_PATH + 'trained_models/')
        self.model.save(BASE_PATH + 'trained_models/bert_model.h5')

    @staticmethod
    def score(test_input, model_path):
        import tensorflow_hub as hub
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        return model.predict(test_input)


class BertPlusBinaryClassifier(BertBinaryClassifier):
    def __init__(self, optim_setup, bert_layer, max_len):
        super().__init__(optim_setup, bert_layer, max_len)
        self.model = self.compile()

    def create_embedding_layer(self, n_items, dim, layer_name, trainable=False):
        """
        This method creates an embeddings layer for each item feature in the feature lookup
        dict

        :param feature_lookup: Dict of two dicts. Contains values and dimensionality of feature
        :param layer_name: String. The name of the layer
        :param trainable: Boolean. If embeddings should be trainable
        :return: Embedding Layer
        """
        emb_layer = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=dim,
            trainable=trainable,
            name=layer_name)
        return emb_layer

    def compile(self):
        input_word_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="segment_ids")
        input_keyword_ids = tf.keras.Input(shape=(1,), dtype=tf.int32, name='keyword_ids')

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]

        hidden_1 = tf.keras.layers.Dense(256, activation='relu')(clf_output)

        hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)

        text_latent = tf.keras.layers.Dense(1, activation='relu')(hidden_2)

        keyword_emb_layer = self.create_embedding_layer(
            n_items=500,
            dim=8,
            trainable=True,
            layer_name='keyword_embedding_layer')

        keyword_embedding = tf.keras.layers.GlobalAveragePooling1D()(keyword_emb_layer(input_keyword_ids))

        concat = tf.keras.layers.concatenate(
            [text_latent, keyword_embedding],
            name='concat_layer',
            axis=1
        )

        out = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids, input_keyword_ids], outputs=out)
        model.compile(
            tf.keras.optimizers.Adam(lr=float(self.optim_setup['base_lr'])),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model
