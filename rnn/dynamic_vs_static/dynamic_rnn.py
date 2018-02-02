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

# We do not need to unpack anymore
rnn_inputs = x_one_hot
print("LSTM input : {}".format(rnn_inputs.shape))
print("")

cell = tf.contrib.rnn.BasicRNNCell(Globals.HIDDEN_STATE)
# Static rnn adds every node for every time step to the graph before execution.
# To do so we need to pass a list (one element for each time step) of the form (BATCH_SIZE, OUTPUT_CLASSES)
# Dynamic rnn on the other hand can handle the case of 3D tensors.
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)


print("LSTM output: {}".format(rnn_outputs.shape))
print("Final state: {}".format(final_state.shape))
print("")

"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [Globals.HIDDEN_STATE, Globals.OUTPUT_CLASSES])
    b = tf.get_variable('b', [Globals.OUTPUT_CLASSES], initializer=tf.constant_initializer(0.0))

# Lets remove time dimension and move it outside: e.g. instead of having
# (number_of_instances, timesteps, features) --> (number_of_instances x timesteps, features)
# Then multiply by W and b.
logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, Globals.HIDDEN_STATE]), W) + b,
            [Globals.BATCH_SIZE, Globals.NUM_STEPS, Globals.OUTPUT_CLASSES])

predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
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
                print("\nEPOCH", (idx + 1))
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
