import numpy as np
import tensorflow as tf

from rnn.dynamic_vs_static.dataset import gen_epochs
from rnn.dynamic_vs_static.globals import Globals


"""
Placeholders
"""

x = tf.placeholder(tf.int32, [Globals.BATCH_SIZE, Globals.NUM_STEPS], name='input_placeholder')
y = tf.placeholder(tf.int32, [Globals.BATCH_SIZE, Globals.NUM_STEPS], name='label_placeholder')
init_state = tf.zeros([Globals.BATCH_SIZE, Globals.HIDDEN_STATE])

print("User input : {}".format(x.shape))
print("User output: {}".format(y.shape))

"""
RNN Inputs
"""

# Turn our x placeholder into a list of one-hot tensors:
# x_one_hot is of the format (BATCH_SIZE, NUM_STEPS, OUTPUT_CLASSES)
x_one_hot = tf.one_hot(x, Globals.OUTPUT_CLASSES)

# rnn_inputs is a list of NUM_STEPS tensors with shape (BATCH_SIZE, OUTPUT_CLASSES)
# This is the format static_rnn expects.
rnn_inputs = tf.unstack(x_one_hot, axis=1)

print("LSTM input : {} x {}".format(len(rnn_inputs), rnn_inputs[0].shape))
print("")

cell = tf.contrib.rnn.BasicRNNCell(Globals.HIDDEN_STATE)
# Static rnn adds every node for every time step to the graph before execution.
# To do so we need to pass a list (one element for each time step) of the form (BATCH_SIZE, OUTPUT_CLASSES)
# Dynamic rnn on the other hand can handle the case of 3D tensors.
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)


"""
Predictions, loss, training step

Losses is similar to the "sequence_loss"
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py#L30
"""
with tf.variable_scope('softmax'):
    # W and b are different from W and b defined in the previous scope
    W = tf.get_variable('W', [Globals.HIDDEN_STATE, Globals.OUTPUT_CLASSES])
    b = tf.get_variable('b', [Globals.OUTPUT_CLASSES], initializer=tf.constant_initializer(0.0))
# Let's create a list of logits.
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=Globals.NUM_STEPS, axis=1)

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
          for logit, label in zip(logits, y_as_list)]

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(Globals.LEARNING_RATE).minimize(total_loss)


"""
Train the network
"""


def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        training_losses_viz = []

        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((Globals.BATCH_SIZE, state_size))
            if verbose:
                print("\nEPOCH", (idx+1))
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                             feed_dict={x: X, y: Y, init_state: training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses_viz.append(training_loss/100)
                    training_loss = 0

    return training_losses_viz


training_losses = train_network(10, Globals.NUM_STEPS)
