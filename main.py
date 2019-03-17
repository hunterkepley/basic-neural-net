# Import `tensorflow`
import tensorflow as tf
import os
from argparse import ArgumentParser
import skimage
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import random

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".png")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

def parse_cmdline_args():
    parser = ArgumentParser(
        description='Command-line program for this neural network'
    )
    parser.add_argument(
        '-e', '--epochs',
        default='0',
        help='How many epochs to train for, 0 or exclusion will not train'
    )
    parser.add_argument(
        '-r', '--run',
        action='store_true',
        help='Pass this to run a test'
    )
    parser.add_argument(
        '-p', '--path',
        default='',
        help='Type a .ckpt file to load the data'
    )
    return parser.parse_args()

def train(epochs, sess, train_op, loss, images28, labels, saver, x, y, model_path):
    for i in range(epochs):
        print('EPOCH', i)
        _, c = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        print("Loss: ", c)
        print('DONE WITH EPOCH')

    save_model(model_path, saver, sess)

def run(test_data_directory, image_x, image_y, sess, correct_pred, x):
    # load the test data
    test_images, test_labels = load_data(test_data_directory)

    # transform the images to 28 by 28 pixels
    test_images28 = [transform.resize(image, (image_x, image_y)) for image in test_images]

    # convert to grayscale
    test_images28 = rgb2gray(np.array(test_images28))

    # run predictions against the full test set
    predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

    # calculate correct matches
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

    # calculate the accuracy
    accuracy = match_count / len(test_labels)

    # print the accuracy
    print("Accuracy of neural net on images: {:.3f}, or, {:.3f}%".format(accuracy, accuracy*100))

def load_model(model_path, saver, sess):
    if(model_path != ''):
        # Restore variables from disk
        try:
            saver.restore(sess, model_path)
            print("---\nModel restored\n---")
        except:
            print("Saved model not found, continuing anyway.")

def save_model(model_path, saver, sess):
    if(model_path != ''):
        # Save variables to disk
        save_path = saver.save(sess, model_path)
        print("---\nModel saved in path: %s\n---" % save_path)

        print("Done training, now testing...")

def main():
    # Get the CL arguments
    epochs = 0
    run_test = 0
    model_path = ''
    args = parse_cmdline_args()
    if args.epochs:
        epochs = int(args.epochs)
    if args.run:
        run_test = 1
    if args.path:
        model_path = args.path
    
    if(epochs == 0 and run_test == 0):
        print("No option selected, quitting program.")
        quit()

    # Data path
    ROOT_PATH = ""
    train_data_directory = os.path.join(ROOT_PATH, "Legos/train")
    test_data_directory = os.path.join(ROOT_PATH, "Legos/valid")

    images, labels = load_data(train_data_directory)

    # image size
    image_x = 48
    image_y = 48

    images28 = [transform.resize(image, (image_x, image_y)) for image in images] # make images 28x28
    images28 = rgb2gray(np.array(images28)) # turn to grayscale after converting images28 to array

    # initialize placeholders
    x = tf.placeholder(dtype = tf.float32, shape = [None, image_x, image_y])
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    # flatten the input data
    images_flat = tf.contrib.layers.flatten(x)

    # fully connected layer
    logits = tf.contrib.layers.fully_connected(images_flat, 16, tf.nn.relu)

    # define a loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

    # define and optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)

    #define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("Training time...")

    tf.set_random_seed(1234)
    sess = tf.Session()

    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess.run(init_op)

    load_model(model_path, saver, sess)

    if(epochs > 0):
        train(epochs, sess, train_op, loss, images28, labels, saver, x, y, model_path)

    if(run_test == 1):
        run(test_data_directory, image_x, image_y, sess, correct_pred, x)

    # Close this session
    sess.close()

if __name__ == '__main__':
    main()
