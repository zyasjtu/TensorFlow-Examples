import tensorflow as tf
import numpy as np
import utils
import autoencoder as ae
import sys
import random
import cv2

def minibatch(file_list, batchsize, w, h, rgb2gray):
        length = len(file_list)
        i = 0
        epoch = 0
        random.shuffle(file_list)
        while True:
                if i + batchsize >= length:
                        random.shuffle(file_list)
                        epoch += 1
                        i = 0
                images = []
                for j in range(i, i+batchsize):
                        content = file_list[j]
                        npos = content.index(',')
                        path = content[:npos]
                        classid = content[npos+1:len(content)-1]

                        image = utils.load_image(path, w, h, rgb2gray)
                        image = image / 255.0
                        images.append(image)
                i += batchsize
                images = np.array(images, dtype=np.float32)
                yield epoch, images


if __name__ == '__main__':
	# Training Data
	cfg = utils.get_cfg(sys.argv[1])
	file_list = utils.load_data(cfg['trainpath'])
        batch = minibatch(file_list, cfg['batchsize'], cfg['width'], cfg['height'], True)
	
	# Training Parameters
	learning_rate = cfg['learningrate']
	num_steps = cfg['maxepoch']
	batch_size = cfg['batchsize']

	# Network Parameters
	num_hidden_1 = cfg['hidden1num'] # 1st layer num features
	num_hidden_2 = cfg['hidden2num'] # 2nd layer num features (the latent dim)
	num_input = cfg['width'] * cfg['height'] * 1 # data input
	network = ae.Autoencoder(num_input, num_hidden_1, num_hidden_2)

	# tf Graph input (only pictures)
	x = tf.placeholder("float", [None, num_input])
	encoded, decoded = network.autoencode(x)

	# Define loss and optimizer, minimize the squared error
	loss = network.loss(x, decoded)
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start Training
	# Start a new TF session
	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)

		# Training
		i = 0
		epoch = 0
		while epoch < num_steps:
        		# Prepare Data
        		# Get the next batch of MNIST data (only images are needed, not labels)
        		epoch, batch_x = next(batch)
			batch_x = batch_x.reshape(batch_size, num_input)

	        	# Run optimization op (backprop) and cost op (to get loss value)
			_, l = sess.run([optimizer, loss], feed_dict={x: batch_x})
                        i += 1
        		# Display logs per step
                        if (i == 1 or i % 50 == 0):
			        print(epoch, ':Step %i: Minibatch Loss: %f' % (i, l))
		
		# Save the model
		tf.train.Saver().save(sess, cfg['modelpath'])

                n = 5
                canvas_orig = np.empty((28 * n, 28 * n))
                canvas_recon = np.empty((28 * n, 28 * n))
                for i in range(n):
                        _, batch_x = next(batch)
                        batch_x = batch_x.reshape(batch_size, num_input)
                        # Encode and decode the digit image
                        g = sess.run(decoded, feed_dict={x: batch_x})

                        # Display original images
                        for j in range(n):
                                # Draw the original digits
                                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
                                # Display reconstructed images
                        for j in range(n):
                                # Draw the reconstructed digits
                                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

                cv2.imwrite(cfg['sampledir'] + 'origin.jpg', canvas_orig * 255)
                cv2.imwrite(cfg['sampledir'] + 'reconstructed.jpg', canvas_recon * 255)
