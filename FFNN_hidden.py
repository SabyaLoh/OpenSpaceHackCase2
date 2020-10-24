import tensorflow as tf
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W0 = tf.Variable(tf.random_normal([784, 200]))
b0 = tf.Variable(tf.random_normal([200]))
hidden0 = tf.nn.relu(tf.matmul(x, W0) + b0)

W1 = tf.Variable(tf.random_normal([200, 200]))
b1 = tf.Variable(tf.random_normal([200]))
hidden1 = tf.nn.relu(tf.matmul(hidden0, W1) + b1)

W2 = tf.Variable(tf.random_normal([200, 200]))
b2 = tf.Variable(tf.random_normal([200]))
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

W3 = tf.Variable(tf.random_normal([200, 10]))
b3 = tf.Variable(tf.random_normal([10]))
y = tf.matmul(hidden2, W3) + b3

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if _ % 1000 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(_, session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('final:', session.run(accuracy, feed_dict={x: mnist.test.images,
            y_: mnist.test.labels}))

'''graf = plt.figure()
plt.plot(x, y)
plt.title('График функции')
plt.ylabel('Ось Y')
plt.xlabel('Ось X')
plt.grid(True)
plt.show()'''