
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import sys
import argparse
import matplotlib.pyplot as plt
import random


# In[ ]:

# hyperparameters
batch_iterations = 3000
batch_size = 50
full_iterations = 60
learning_rate = 0.01
reg_eta = 0.001

# dimensionalities
dim_lstm = 50
dim_word = 300
dim_aspect = 5
dim_sentence = 80
dim_polarity = 3

# setup utils object
isSample = False
u = utils.UTILS(batch_size, dim_sentence, dim_polarity, isSample)


# In[ ]:

# define tf placeholders
X = tf.placeholder(tf.int32, [None, dim_sentence])
y = tf.placeholder(tf.float32, [None, dim_polarity])
seqlen = tf.placeholder(tf.int32, [None])
lr = tf.placeholder(tf.float32, [])


# In[ ]:

# define tf variables
with tf.variable_scope('bilstm_vars'):
    with tf.variable_scope('weights', reuse = tf.AUTO_REUSE):
        lstm_w = tf.get_variable(
            name = 'softmax_w',
            shape = [dim_lstm * 2, dim_polarity],
            initializer = tf.random_uniform_initializer(-0.003, 0.003),
            regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
        )
    with tf.variable_scope('biases', reuse = tf.AUTO_REUSE):
        lstm_b = tf.get_variable(
            name = 'softmax_b',
            shape = [dim_polarity],
            initializer = tf.random_uniform_initializer(-0.003, 0.003),
            regularizer = tf.contrib.layers.l2_regularizer(reg_eta)
        )


# In[ ]:

# define lstm model
def dynamic_lstm(inputs, seqlen):
    inputs = tf.nn.dropout(inputs, keep_prob=1.0)
    with tf.name_scope('bilstm_model'):
        forward_lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
        backward_lstm_cell = tf.contrib.rnn.LSTMCell(dim_lstm)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            forward_lstm_cell,
            backward_lstm_cell,
            inputs = inputs,
            sequence_length = seqlen,
            dtype = tf.float32,
            scope = 'bilstm'
        )
        forward_outputs, backward_outputs = outputs
        backward_outputs = tf.reverse_sequence(backward_outputs, tf.cast(seqlen, tf.int64), seq_dim=1)
        outputs = tf.concat([forward_outputs, backward_outputs], 2)
        size = tf.shape(outputs)[0]
        index = tf.range(0, size) * dim_sentence + seqlen - 1
        output = tf.gather(tf.reshape(outputs, [-1, dim_lstm * 2]), index)  # batch_size * n_hidden * 2
    predict = tf.matmul(output, lstm_w) + lstm_b
    return predict


# In[ ]:

# define operations
# tf.reset_default_graph()
pred = dynamic_lstm(tf.nn.embedding_lookup(u.gloveDict, X), seqlen)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()


# In[ ]:
def batchTrain():
# batch training
     test_X, test_y, test_seqlen, _ = u.getData('test')
     train_X, train_y, train_seqlen, train_aspects = u.getData('train')
     lrval = 0.01

     costs_train = []
     costs_test = []
     with tf.Session() as sess:
         sess.run(init)
         for i in range(batch_iterations):
              batch_X, batch_y, batch_seqlen, _ = u.nextBatch(batch_size)
              lrval *= (1. / (1. + 0.2 * i))
              sess.run(optimizer, feed_dict = {X: batch_X, y: batch_y, seqlen: batch_seqlen, lr: lrval})

              if i % 10 ==0:
                  loss_train, accuracy_train = sess.run([loss, accuracy],
                                                        feed_dict={X: batch_X, y: batch_y, seqlen: batch_seqlen})
                  costs_train.append(loss_train)
                  loss_test, accuracy_test = sess.run([loss, accuracy],
                                                      feed_dict={X: test_X, y: test_y, seqlen: test_seqlen})
                  costs_test.append(loss_test)

              if i % 100 == 0:
                  loss_train, accuracy_train = sess.run([loss, accuracy], feed_dict = {X: batch_X, y: batch_y, seqlen: batch_seqlen})
                  print('step: %s, train loss: %s, train accuracy: %s' % (i, loss_train, accuracy_train))
                  loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict = {X: test_X, y: test_y, seqlen: test_seqlen})
                  print('step: %s, val loss: %s, val accuracy: %s' % (i, loss_test, accuracy_test))

         cost_train, acc_train = sess.run([loss, accuracy],
                                               feed_dict={X: train_X, y: train_y, seqlen: train_seqlen,
                                                          })
         cost_test, acc_test = sess.run([loss, accuracy],
                                             feed_dict={X: test_X, y: test_y, seqlen: test_seqlen,
                                                        })
         print('test loss: %s, test accuracy: %s' % (cost_test, acc_test))


     tt = np.arange(0, batch_iterations, 10)
     iter = int(batch_iterations / 10)
     err_train = np.empty((iter, 1))
     err_val = np.empty((iter, 1))
     for k in range(0, iter):
          err_train[k, 0] = costs_train[k]
          err_val[k, 0] = costs_test[k]

     fig, ax = plt.subplots()
     ax.plot(tt, err_train, 'r', label='Train_error')
     ax.plot(tt, err_val, 'b', label='Val_error')

     legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
     legend.get_frame().set_facecolor('#00FFCC')
     plt.ylabel('Error')
     plt.title('bi-lstm Learning Curve')
     ax.annotate('batch=' + str(batch_size) + ',iter=' + str(batch_iterations) + ',learning_rate=' + str(
                       learning_rate) + ',dim_lstm=' + str(dim_lstm) + ',dim_word=' + str(dim_word),
            xy=(0.5, 0), xytext=(0, 10),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=8, ha='center', va='bottom')
     plt.show()

     res = {'Train_accuracy': [acc_train], 'Test_accuracy': [acc_test], 'batch_size': [batch_size],
       'iterations': [batch_iterations], 'learning_rate': [learning_rate], 'dim_lstm': [dim_lstm],
       'dim_word': [dim_word]}
     df = pd.DataFrame(data=res)
     df.to_csv(r"bi-lstm_performance.csv", mode='a', header=True, index=False)

# In[ ]:

# full dataset training
def fullTrain():

     test_X, test_y, test_seqlen, _ = u.getData('test')
     train_X, train_y, train_seqlen, _ = u.getData('train')
     lrval = 0.01
     costs_train = []
     costs_val = []
     with tf.Session() as sess:
           sess.run(init)
           for i in range(full_iterations):
                lrval *= (1. / (1. + 0.2 * i))
                sess.run(optimizer, feed_dict = {X: train_X, y: train_y, seqlen: train_seqlen, lr: lrval})
                loss_train, accuracy_train = sess.run([loss, accuracy],
                                                          feed_dict={X: train_X, y: train_y, seqlen: train_seqlen,
                                                                     })
                costs_train.append(loss_train)
                loss_test, accuracy_test = sess.run([loss, accuracy],
                                                        feed_dict={X: test_X, y: test_y, seqlen: test_seqlen,
                                                                   })
                costs_val.append(loss_test)

                print('step: %s, train loss: %s, train accuracy: %s' % (i, loss_train, accuracy_train))

                print('step: %s, val loss: %s, val accuracy: %s' % (i, loss_test, accuracy_test))


     tt = np.arange(0, full_iterations, 1)
     iter = int(full_iterations)
     err_train = np.empty((iter, 1))
     err_val = np.empty((iter, 1))
     for k in range(0, iter):
         err_train[k, 0] = costs_train[k]
         err_val[k, 0] = costs_val[k]

     fig, ax = plt.subplots()
     ax.plot(tt, err_train, 'r', label='Train_error')
     ax.plot(tt, err_val, 'b', label='Val_error')

     legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
     legend.get_frame().set_facecolor('#00FFCC')
     plt.ylabel('Error')
     plt.title('bi-lstm Learning Curve')
     ax.annotate('batch=' + str(0) + ',iter=' + str(full_iterations) + ',learning_rate=' + str(
         learning_rate) + ',dim_lstm=' + str(dim_lstm),
                 xy=(0.5, 0), xytext=(0, 10),
                 xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points',
                 size=8, ha='center', va='bottom')
     plt.show()

     res = {'Train_accuracy': [accuracy_train], 'Test_accuracy': [accuracy_test], 'batch_size': 0,
            'iterations': [full_iterations], 'learning_rate': [learning_rate], 'dim_lstm': [dim_lstm],
            }
     df = pd.DataFrame(data=res)
     df.to_csv(r"bi-lstm_performance.csv", mode='a', header=True, index=False)


# In[ ]:
if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, choices=[0, 1], default=0)   # 0 for full train, 1 for batch train

    args, _ = parser.parse_known_args(argv)
    runDict = {0: fullTrain, 1: batchTrain}

    # start training
    runDict[args.run]()



