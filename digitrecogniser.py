"""MNIST Digit classifier using softmax regression
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import csv

def get_data(filename):
    df = pd.read_csv(filename)
    x = df[df.columns[1:]]
    x = x/255.0
    y = df[df.columns[0]]
    return x, y

def submit(answers):
	f = open("submission.csv", "w")
	writer = csv.writer(f)
	writer.writerow(["ImageId","Label"])
	for i in range(len(answers)):
		writer.writerow([i+1, answers[i]])

def train():
    learning_rate = 0.003
    batch_size = 100

    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x,w) + b

    trainX, trainY = get_data("train.csv")

    labels = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
    trainer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    epochs = len(trainY)/batch_size
    for i in range(epochs):
    	print "Epoch ",i+1
    	batch_x = trainX[epochs*i:epochs*i+batch_size]
    	batch_y = np.eye(10)[trainY[epochs*i:epochs*i+batch_size]]
    	#print batch_y.shape
    	sess.run(trainer, feed_dict={x: batch_x, labels: batch_y})
    	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print tf.scalar_summary("accuracy", accuracy)

    test_X = pd.read_csv("test.csv")
    answers = sess.run(tf.argmax(y, 1), feed_dict={x: test_X})
    submit(answers)
    print "File written"

train()
    


