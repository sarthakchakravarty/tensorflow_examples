import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

#np.random.seed(0)

def create_random_data(n):
    x = np.random.random(n)
    #Simple equation for a line.
    y = 7*x + 8 + np.random.normal(0,0.5,n) # Insert random noise to data
    return x, y
    

def update_cost_list(cost_list, cost):
    if len(cost_list) == 5:
        _ = cost_list.pop(0)
        cost_list.append(cost)
    else:
        cost_list.append(cost)
    return cost_list


def check_cost_list(cost_list, threshold):
    if abs(np.mean(np.diff(cost_list))) < threshold:
        return True
    else:
        return False


def linear_optimizer(x_data, y_data, n, lr, cost_lim):
    x_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    w = tf.Variable(tf.ones(1))
    b = tf.Variable(tf.ones(1))
    
    pred = tf.add(tf.multiply(w, x_), b)
    
    cost = tf.reduce_mean(tf.square(y_ - pred))
    
    train = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cost_list = []
        plt.ion()
        epoch = 0
        while True:
            _, cost_val = sess.run([train, cost], feed_dict={x_: x_data, y_: y_data})
            weight = sess.run(w)
            bias = sess.run(b)
            cost_list = update_cost_list(cost_list, float(cost_val))
            # Plot best fit line on the points
            if not epoch % 20:
                if check_cost_list(cost_list, cost_lim):
                    break
                else:
                    pass
                plt.cla()
                plt.plot(x_data, y_data, 'rx')
                plt.plot(x_data, weight * x_data + bias)
                plt.show()
                plt.pause(0.1)
            epoch += 1
        plt.ioff()
        #plt.show()
        return {"weight": float(weight), "bias": float(bias), 
                "cost": float(cost_val), "epoch": epoch}


def evaluate_model(x_data, y_data, weight, bias):
    """
    Calculating cost using calculated weights and bias.
    """
    y_pred = weight * x_data + bias
    cost = np.mean(np.square(y_data - y_pred))
    return cost


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        '--num-points',
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
        '--cost-limit',
        type=float,
        default=0.00001,
        help="Threshold upto which cost function should be calculated"
    )
    params = vars(args.parse_args())
    n, lr, cost_lim = params['num_points'], params['lr'], params['cost_limit']
    x_train, y_train = create_random_data(n)
    train_dict = linear_optimizer(x_train, y_train, n, lr, cost_lim)
    print("Weight: {0:.3f}, Bias: {1:.3f}, cost: {2:.3f}, Epoch: {3}"\
          .format(train_dict['weight'], train_dict['bias'], train_dict['cost'],
                  train_dict['epoch']))
    x_test, y_test = create_random_data(n)
    train_cost = evaluate_model(x_test, y_test, train_dict['weight'],
                                train_dict['bias'])
    print("Test cost: {:.3f}".format(train_cost))