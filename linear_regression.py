import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(0)

def create_random_data(n):
    x = np.random.random(n)
    #Simple equation for a line.
    y = 7*x + 8 + np.random.normal(0,1,n) # Insert random noise to data
    return x, y
    

def make_latest_loss(loss_list, loss):
    if len(loss_list) == 5:
        _ = loss_list.pop(0)
        loss_list.append(loss)
    else:
        loss_list.append(loss)
    return loss_list


def check_loss_list(loss_list, threshold):
    if abs(np.mean(np.diff(loss_list))) < threshold:
        return True
    else:
        return False


def linear_optimizer(n, lr, epoch):
    x_train, y_train = create_random_data(n)
    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    w = tf.Variable(tf.ones(1))
    b = tf.Variable(tf.ones(1))
    
    pred = tf.add(tf.multiply(w, x_), b)
    
    loss = tf.reduce_mean(tf.square(y_ - pred))
    
    train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_list = []
        plt.ion()
        for i in range(epoch):
            _, loss_val = sess.run([train, loss], feed_dict={x_: x_train, y_: y_train})
            weight = sess.run(w)
            bias = sess.run(b)
            loss_list = make_latest_loss(loss_list, float(loss_val))
            # Plot best fit line on the points
            if not i % 20:
                if check_loss_list(loss_list, 0.0001):
                    break
                else:
                    pass
                plt.cla()
                plt.plot(x_train, y_train, 'rx')
                plt.plot(x_train, weight * x_train + bias)
                plt.show()
                plt.pause(0.1)
        plt.ioff()
        #plt.show()
        print("Weight: {0:.3f}, Bias: {1:.3f}, Loss: {2:.3f}"\
              .format(float(weight), float(bias), float(loss_val)))
                                                                         
                                                                         

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--num_points',
        type=int,
        default=100,
        help="Number of random data points"
    )
    args.add_argument(
        '--lr',
        type=float,
        default=0.5,
        help="Learning rate"
    )
    args.add_argument(
        '--epoch',
        type=int,
        default=100,
        help="Number of iterations"
    )
    params = vars(args.parse_args())
    n, lr, epoch = params['num_points'], params['lr'], params['epoch']
    linear_optimizer(n, lr, epoch)