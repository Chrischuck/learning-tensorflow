import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

################## Implementing Regression ##################

# img flattened into 784 dimensional vector
# 2D vector that can have dimensions of any length
x = tf.placeholder(tf.float32, [None, 784])

# weights, 784 dimesion vector, when we multiply, we'll get a 10 d one
W = tf.Variable(tf.zeros([784, 10]))
# shape of 10 so we can just dd it
b = tf.Variable(tf.zeros([10]))

# The model
y = tf.nn.softmax(tf.matmul(x, W) + b)

################## Training ##################

y_ = tf.placeholder(tf.float32, [None, 10])

# create cross_entropy? 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# train 1k times
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

################## Evaluating ##################

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
