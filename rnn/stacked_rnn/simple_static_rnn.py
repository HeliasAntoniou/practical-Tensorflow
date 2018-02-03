import tensorflow as tf

from rnn.stacked_rnn.model import RNNModel


class SimpleStaticRNN(RNNModel):

    def __init__(self):
        super(SimpleStaticRNN, self).__init__()

    def _build_graph(self,
                     state_size=100,
                     num_steps=200,
                     num_layers=3):
        tf.reset_default_graph()

        x = tf.placeholder(tf.int32, [self.BATCH_SIZE, num_steps], name='input_placeholder')
        y = tf.placeholder(tf.int32, [self.BATCH_SIZE, num_steps], name='labels_placeholder')

        print("")
        print("Input placeholder : {}".format(x.shape))
        print("Output placeholder: {}".format(y.shape))
        print("")

        # Map your input to a different dimension. So it maps let's say number 45 to a vector with shape state_size.
        # embeddings is a tensor of shape (vocab_size, embedding_vector_length)
        embeddings = tf.get_variable('embedding_matrix', [self.NUM_CLASSES, state_size])

        # tf.nn.embedding_lookup(embeddings, x) will transform the input of the shape (training_instances, steps)
        # with each one being a number in the vocabulary (e.g. 421, 234, 21) to a 3D tensor with each number
        # transformed to the corresponding embedding with shape (training_instances, steps, embedding_size)
        rnn_inputs_3d = tf.nn.embedding_lookup(embeddings, x)

        # Lets create a list instead of a 3d vector. So instead of having a vector of shape:
        # (training_instances, steps, embedding_size) --> steps x (training_instances, embedding_size)
        rnn_inputs = [tf.squeeze(i) for i in tf.split(rnn_inputs_3d, num_steps, 1)]

        # Create the cells
        cells = []
        for i in range(num_layers):
            with tf.variable_scope("LSTM_{}".format(i)):
                cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
                cells.append(cell)

        # Stack the cells
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # Create init state. It is a tuple with each element the state of the stack RNN. So if we use 3 stacked RNNs
        # we will end up with a tuple of 3 elements. Each tuple will correspond to another tuple: h and c variables
        # of the RNN (hidden and cell).
        init_state = cell.zero_state(self.BATCH_SIZE, tf.float32)

        print("Init state             :")
        print("Number of stacks        : {}".format(len(init_state)))
        print("State elements per stack: {}".format(len(init_state[0])))
        print("Each state element      : {}".format(init_state[0][0].shape))
        print("")

        # Build static RNN graph.
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, self.NUM_CLASSES])
            b = tf.get_variable('b', [self.NUM_CLASSES], initializer=tf.constant_initializer(0.0))

        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

        y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]

        loss_weights = [tf.ones([self.BATCH_SIZE]) for _ in range(num_steps)]
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(total_loss)

        return dict(
            x=x,
            y=y,
            init_state=init_state,
            final_state=final_state,
            total_loss=total_loss,
            train_step=train_step
        )


rnn = SimpleStaticRNN()
graph = rnn.build_graph()
rnn.train_network(graph, 10)
