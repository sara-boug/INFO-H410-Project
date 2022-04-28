import tensorflow as tf

from tensorflow.keras.datasets import mnist

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# def get_batch(x_data, y_data, batch_size):
#     idxs = np.random.randint(0, len(y_data), batch_size)
#     return x_data[idxs,:,:], y_data[idxs]
#
# def nn_model(x_input, W1, b1, W2, b2):
#     # flatten the input image from 28 x 28 to 784
#     x_input = tf.reshape(x_input, (x_input.shape[0], -1))
#     x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
#     x = tf.nn.relu(x)
#     logits = tf.add(tf.matmul(x, W2), b2)
#     return logits
#
# def loss_fn(logits, labels):
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
#     return cross_entropy

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')
    tf.test.is_gpu_available()
    print("TensorFlow version:", tf.__version__)
    """
    # Initialize two constants
    x1 = tf.constant([1, 2, 3, 4])
    x2 = tf.constant([5, 6, 7, 8])

    # Multiply
    result = tf.multiply(x1, x2)

    # Intialize the Session
    sess = tf.compat.v1.Session()

    # Print the result
    print(sess.run(result))

    # Close the session
    sess.close()
    """

    print("fin1")

    """
        #
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #
        # # Python optimisation variables
        # epochs = 10
        # batch_size = 100
        # # normalize the input images by dividing by 255.0
        # x_train = x_train / 255.0
        # x_test = x_test / 255.0
        # # convert x_test to tensor to pass through model (train data will be converted to
        # # tensors on the fly)
        # x_test = tf.Variable(x_test)
        #
        #
        #
        #
        # # now declare the weights connecting the input to the hidden layer
        # W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
        # b1 = tf.Variable(tf.random.normal([300]), name='b1')
        # # and the weights connecting the hidden layer to the output layer
        # W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='W2')
        # b2 = tf.Variable(tf.random.normal([10]), name='b2')
        #
        # # setup the optimizer
        # optimizer = tf.keras.optimizers.Adam()
        #
        #
        #
        # total_batch = int(len(y_train) / batch_size)
        # for epoch in range(epochs):
        #     avg_loss = 0
        #     for i in range(total_batch):
        #         batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
        #         # create tensors
        #         batch_x = tf.Variable(batch_x)
        #         batch_y = tf.Variable(batch_y)
        #         # create a one hot vector
        #         batch_y = tf.one_hot(batch_y, 10)
        #         with tf.GradientTape() as tape:
        #             logits = nn_model(batch_x, W1, b1, W2, b2)
        #             loss = loss_fn(logits, batch_y)
        #         gradients = tape.gradient(loss, [W1, b1, W2, b2])
        #         optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
        #         avg_loss += loss / total_batch
        #     test_logits = nn_model(x_test, W1, b1, W2, b2)
        #     max_idxs = tf.argmax(test_logits, axis=1)
        #     test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
        #     print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set      accuracy={test_acc * 100:.3f}%")
        # print("\nTraining complete!")
    """



    x = tf.compat.v1.placeholder(tf.float64, shape=[4, 2], name="x")
    # declaring a place holder for input x
    y = tf.compat.v1.placeholder(tf.float64, shape=[4, 1], name="y")
    # declaring a place holder for desired output y


    m = np.shape(x)[0]  # number of training examples
    n = np.shape(x)[1]  # number of features
    hidden_s = 2  # number of nodes in the hidden layer
    l_r = 0.01  # learning rate initialization

    theta1 = tf.cast(tf.Variable(tf.random.normal([n, hidden_s]), name="theta1"), tf.float64)
    theta2 = tf.cast(tf.Variable(tf.random.normal([hidden_s, 1]), name="theta2"), tf.float64)


    print(theta1)
    print(theta2)

    # conducting forward propagation
    a1 = x


    # the weights of the first layer are multiplied by the input of the first layer

    z1 = tf.matmul(a1, theta1)

    # the input of the second layer is the output of the first layer, passed through the activation function and column of biases is added

    a2 = tf.sigmoid(z1)

    # the input of the second layer is multiplied by the weights

    z3 = tf.matmul(a2, theta2)

    # the output is passed through the activation function to obtain the final probability

    h3 = tf.sigmoid(z3)

#    cost_func = -tf.reduce_sum(y * tf.math.log(h3) + (1 - y) * tf.math.log(1 - h3), axis=1)
#    cost_func = tf.reduce_sum(y-h3, axis=1)
    cost_func = tf.reduce_mean(tf.squared_difference(h3, y))

    print("cost function:")
    print(cost_func)


    
    # built in tensorflow optimizer that conducts gradient descent using specified learning rate to obtain theta values

#    optimiser = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=l_r).minimize(cost_func)
    optimiser = tf.compat.v1.train.AdamOptimizer(l_r).minimize(cost_func)

    # setting required X and Y values to perform XOR operation
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]

    # initializing all variables, creating a session and running a tensorflow session
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)
#############
    print("afficher cost_func:\n", sess.run(cost_func, feed_dict={x: X, y: Y}))
    print("afficher theta1:\n", sess.run(theta1, feed_dict={x: X, y: Y}))
    print("afficher theta2:\n", sess.run(theta2, feed_dict={x: X, y: Y}))
    print("afficher a1:\n", sess.run(a1, feed_dict={x: X, y: Y}))
    print("afficher z1:\n", sess.run(z1, feed_dict={x: X, y: Y}))
    print("afficher a2:\n", sess.run(a2, feed_dict={x: X, y: Y}))
    print("afficher z3:\n", sess.run(z3, feed_dict={x: X, y: Y}))
    print("afficher h3:\n", sess.run(h3, feed_dict={x: X, y: Y}))
    print("afficher x:\n", sess.run(x, feed_dict={x: X, y: Y}))
    print("afficher y:\n", sess.run(y, feed_dict={x: X, y: Y}))
#    sess.close()
#############


    # running gradient descent for each iteration and printing the hypothesis obtained using the updated theta values
    sol = []
    for i in range(100000):
        sess.run(optimiser, feed_dict={x: X, y: Y})  # setting place holder values using feed_dict
        if i % 1000 == 0:
            print("Epoch:", i)
            sol = sess.run(h3, feed_dict={x: X, y: Y})
            print("Hyp:", sol)

    print("sol:")
    print(np.round(sol, decimals=1))
    print("Y")
    print(Y)
    print("fin2")
#    print(y_test)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
