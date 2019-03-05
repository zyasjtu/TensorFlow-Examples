import tensorflow as tf
import numpy as np
import utils
import autoencoder as ae
import sys
import random
import cv2


if __name__ == '__main__':
	# Training Data
	cfg = utils.get_cfg(sys.argv[1])
        img = utils.load_image(sys.argv[2], cfg['width'], cfg['height'], True)
        cv2.imwrite(cfg['sampledir'] + 'input.jpg', img)
        img = img / 255.0
        batch_1 = img.reshape(1, cfg['width'] * cfg['height'])

	# Network Parameters
	num_hidden_1 = cfg['hidden1num'] # 1st layer num features
	num_hidden_2 = cfg['hidden2num'] # 2nd layer num features (the latent dim)
	num_input = cfg['width'] * cfg['height'] * 1 # data input
	network = ae.Autoencoder(num_input, num_hidden_1, num_hidden_2)

	# tf Graph input (only pictures)
	x = tf.placeholder("float", [None, num_input])
	encoded, decoded = network.autoencode(x)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start a new TF session
	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)

		# Restore
                tf.train.Saver().restore(sess, cfg['modelpath'])

	        # Predict
                out = sess.run(decoded, feed_dict={x: batch_1})[0]
		out = out.reshape(cfg['width'], cfg['height'])

		# Save the sample
                cv2.imwrite(cfg['sampledir'] + 'output.jpg', out * 255.0)

