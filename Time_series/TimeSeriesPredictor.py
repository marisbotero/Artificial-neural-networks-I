import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import time_series_preprocessor as tsp
import matplotlib.pyplot as plt


input_dim = 1
seq_size = 5
hidden_dim = 5

W_out = tf.get_variable("W_out", shape=[hidden_dim, 1], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None) 
b_out = tf.get_variable("b_out", shape=[1], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)
x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
y = tf.placeholder(tf.float32, [None, seq_size])

def LSTM_Model():
	cell = rnn_cell.BasicLSTMCell(hidden_dim)        
	outputs, states = rnn.dynamic_rnn(cell, x, dtype=tf.float32)        
	num_examples = tf.shape(x)[0]        
	W_repeated = tf.tile(tf.expand_dims(W_out, 0), [num_examples, 1, 1])        
	out = tf.matmul(outputs, W_repeated) + b_out        
	out = tf.squeeze(out)        
	return out

train_loss = []
test_loss = []
step_list = []

def trainNetwork(train_x, train_y, test_x, test_y):
	 with tf.Session() as sess:
	 	 tf.get_variable_scope().reuse_variables()
	 	 sess.run(tf.global_variables_initializer())
	 	 max_patience = 3
	 	 patience = max_patience
	 	 min_test_err = float('inf')
	 	 step = 0
	 	 while patience > 0:
	 	 	 _, train_err = sess.run([train_op, cost], feed_dict={x: train_x, y: train_y})
	 	 	if step % 100 == 0:
	 	 		test_err = sess.run(cost, feed_dict={x: test_x, y: test_y})
	 	 	 	print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
	 	 	 	train_loss.append(train_err)
	 	 	 	test_loss.append(test_err)
	 	 	 	step_list.append(step)
	 	 	 	if test_err < min_test_err:
	 	 	 		min_test_err = test_err
	 	 	 		patience = max_patience
	 	 	 	else:
	 	 	 		patience -= 1
	 	 	 	step += 1
	 	 	save_path = saver.save(sess, 'model.ckpt')
	 	 	print('Model saved to {}'.format(save_path))

cost = tf.reduce_mean(tf.square(LSTM_Model()- y))
train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)

saver = tf.train.Saver()


def testLSTM(sess, test_x):        
	tf.get_variable_scope().reuse_variables()        
	saver.restore(sess, 'model.ckpt')        
	output = sess.run(LSTM_Model(), feed_dict={x: test_x})       
	return output


def plot_results(train_x, predictions, actual, filename):
	plt.figure()
	num_train = len(train_x)
	plt.plot(list(range(num_train)), train_x, color='b', label='training data')
	plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
	plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
	plt.legend()
	if filename is not None:
		plt.savefig(filename)
	else:
		plt.show()
	

