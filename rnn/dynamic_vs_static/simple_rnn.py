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


"""
RNN Cell
"""
# Let's initialize our variables. The same variables will be used inside each RNN cell.
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [Globals.OUTPUT_CLASSES + Globals.HIDDEN_STATE, Globals.HIDDEN_STATE])
    b = tf.get_variable('b', [Globals.HIDDEN_STATE], initializer=tf.constant_initializer(0.0))


# This returns the exact same rnn_cell each time called.
def rnn_cell(rnn_input, previous_state):
    # Notice the reuse flag. It will retrive from scope the variables W and b upon request.
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [Globals.OUTPUT_CLASSES + Globals.HIDDEN_STATE, Globals.HIDDEN_STATE])
        b = tf.get_variable('b', [Globals.HIDDEN_STATE], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, previous_state], 1), W) + b)


"""
Build the graph
"""
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]


"""
Predictions, loss, training step

Losses is similar to the "sequence_loss"
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py#L30
"""
with tf.variable_scope('softmax'):
    # W and b are different from W and b defined in the previous scope
    W = tf.get_variable('W', [Globals.HIDDEN_STATE, Globals.OUTPUT_CLASSES])
    b = tf.get_variable('b', [Globals.OUTPUT_CLASSES], initializer=tf.constant_initializer(0.0))

# Let's create a list of logits. Each element in the list is a vector of shape (2,).
# The length of the list is the same as the one the length of the sequence
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

# Use softmax to transform to probabilities. Again it is a list with shape (2,) having as many elements
# as the length of the input sequence
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
