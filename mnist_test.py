from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
slim = tf.contrib.slim

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
#one_hot labels
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

inputs = tf.reshape(x, [-1, 28, 28, 1])
net = slim.convolution2d(inputs, 32, [3, 3], stride=1, scope='conv1')
print(net)
net = slim.max_pool2d(net, [2, 2], stride=1, padding='SAME', scope='pool1')
print(net)
net = slim.convolution2d(net, 32, [3, 3], stride=2, scope='conv2')
print(net)
net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
print(net)
net = slim.convolution2d(net, 64, [3, 3], stride=1, scope='conv3')
print(net)
net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')
print(net)
net = slim.flatten(net, scope='flatten')
print(net)
net = slim.fully_connected(net, 64, scope='fc1')
print(net)
net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')
print(net)
output = net

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.summary.scalar('Cross entropy', cross_entropy)
#inference
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Train accuracy', accuracy)
merged = tf.summary.merge_all()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter('mnist_summary/', sess.graph)

  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    writer.add_summary(summary, i)

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
