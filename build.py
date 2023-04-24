from keras import Sequential
from keras.datasets import imdb
from keras.layers import Concatenate as ConcatenateLayer
from keras.layers import Dense as DenseLayer
from keras.layers import Dropout as DropoutLayer
from keras.layers import Embedding as EmbeddingLayer
from keras.layers import Input as InputLayer
from keras.layers import Layer as KerasLayer
from keras.layers import LayerNormalization as NormalizationLayer
from keras.layers import MultiHeadAttention as MultiHeadAttentionLayer
from keras.layers import Reshape as ReshapeLayer
from keras.utils import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10_000)
word_index = imdb.get_word_index()

x_train = pad_sequences(x_train, maxlen=256)
x_test = pad_sequences(x_train, maxlen=256)

# Time to build the structure of the Transformer


class Transformer(KerasLayer):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, base_truth):
        encoder_result = self.encoder.call(inputs)
        decoder_result = self.decoder.call(base_truth, encoder_result)

        return decoder_result


class Encoder(KerasLayer):
    def __init__(self):
        super().__init__()

        self.mha_layer = ResidualNormalizedMHALayer()
        self.ff_layer = ResidualNormalizedFeedForwardLayer()

    def call(self, inputs):
        mha_result = self.mha_layer.call(inputs)
        ff_result = self.ff_layer.call(mha_result)

        return ff_result


class Decoder(KerasLayer):
    def __init__(self):
        super().__init__()

        self.self_attention_mha_layer = ResidualNormalizedMHALayer()
        self.concatenation_layer = ConcatenateLayer()
        self.mha_layer = ResidualNormalizedMHALayer()
        self.ff_layer = ResidualNormalizedFeedForwardLayer()

    def call(self, base_truth, encoder_result):
        self_attention_result = self.self_attention_mha_layer.call(base_truth)
        concatenated_input = self.concatenation_layer(
            axis=-1)([self_attention_result, encoder_result])
        mha_result = self.mha_layer.call(concatenated_input)
        ff_result = self.ff_layer.call(mha_result)

        return ff_result


class ResidualNormalizedFeedForwardLayer(KerasLayer):
    def __init__(self):
        super().__init__()

    def call(self):
        pass


class ResidualNormalizedMHALayer(KerasLayer):
    def __init__(self, num_heads, key_dim, dropout_rate):
        super().__init__()

        self.mha_layer = MultiHeadAttentionLayer(num_heads, key_dim)
        self.dropout_layer = DropoutLayer(dropout_rate)
        self.normalization_layer = NormalizationLayer()

    def call(self, inputs):
        mha_result = self.mha_layer(inputs, inputs)
        dropout_result = self.dropout_layer(mha_result)
        normalization_result = self.normalization_layer(
            dropout_result + inputs)

        return normalization_result


# Test the architecture
model = Sequential()

model.add(InputLayer(shape=(None, 256)))
model.add(EmbeddingLayer(
    input_dim=10_000,
    output_dim=64,
    input_length=256
))
model.add(ResidualNormalizedMHALayer(
    num_heads=3,
    key_dim=64,
    dropout_rate=0.1
))
model.add(DenseLayer(10_000, activation="softmax"))

print(model.summary())

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x=x_train, y=x_train, epochs=1, batch_size=16)
