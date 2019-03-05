import tensorflow as tf


class Autoencoder():
    def __init__(self, num_input, num_hidden_1, num_hidden_2):
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([num_input])),
        }

    def autoencode(self, x):
        # Construct model
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    # Building the encoder
    def encode(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decode(self, x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2

    def loss(self, x, decoded):
        # Define loss and optimizer, minimize the squared error
        return tf.reduce_mean(tf.pow(x - decoded, 2))
