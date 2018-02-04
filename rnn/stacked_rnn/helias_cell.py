import tensorflow as tf

from tensorflow.contrib.rnn.python.ops import core_rnn_cell


class HeliasCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, num_weights):
        super(HeliasCell, self).__init__()

        self._num_units = num_units
        self._num_weights = num_weights

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return self._num_units

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        Every `RNNCell` must have the properties below and implement `call` with
        the signature `(output, next_state) = call(input, state)`.  The optional
        third input argument, `scope`, is allowed for backwards compatibility
        purposes; but should be left off for new subclasses.

        This operation results in an output matrix with `self.output_size` columns.
        If `self.state_size` is an integer, this operation also results in a new
        state matrix with `self.state_size` columns.  If `self.state_size` is a
        (possibly nested tuple of) TensorShape object(s), then it should return a
        matching structure of Tensors having shape `[batch_size].concatenate(s)`
        for each `s` in `self.batch_size`.

        :param inputs: `2-D` tensor with shape `[batch_size, input_size]`
        :param state: If `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`
        :param scope: VariableScope for the created subgraph; defaults to class name.
        :return:
            A pair containing:
            - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
            - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`.
        """
        with tf.variable_scope(scope or type(self).__name__):
            print("")
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                # ru is pretty much a concatenation of input and previous state.
                # From those values we will evaluate the reset and update gate!
                ru = core_rnn_cell._linear([inputs, state], 2 * self._num_units, True)

                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(ru, 2, 1)

                print("Reset Gate     : {}".format(r.shape))
                print("Update Gate    : {}".format(u.shape))
            with tf.variable_scope("Candidate"):
                # Lambdas are the weights. We have as many lambdas as the nu_weights variable
                lambdas = core_rnn_cell._linear([inputs, state], self._num_weights, True)
                lambdas = tf.split(tf.nn.softmax(lambdas), self._num_weights, 1)
                print("Lambdas        : {} x {}".format(len(lambdas), lambdas[0].shape))

                # Ws determine which lambda we will favor.
                Ws = tf.get_variable("Ws",
                                     shape=[self._num_weights, inputs.get_shape()[1], self._num_units])
                Ws = [tf.squeeze(i) for i in tf.split(Ws, self._num_weights, 0)]
                print("Lambdas weights: {} x {}".format(len(Ws), Ws[0].shape))

                candidate_inputs = []

                # We multiply each lambda with the corresponding W.
                # We have num_weights candidates which we will add up
                for idx, W in enumerate(Ws):
                    candidate_inputs.append(tf.matmul(inputs, W) * lambdas[idx])

                Wx = tf.add_n(candidate_inputs)
                print("Weight matrix  : {}".format(Wx.shape))

                with tf.variable_scope("second"):
                    # Let's use reset gate to reset the previous state to some degree and add the result of our
                    # lambdas multiplication.
                    c = tf.nn.tanh(Wx + core_rnn_cell._linear([r * state], self._num_units, True))
            # Use update gate to estimate our new output.
            new_h = u * state + (1 - u) * c
            print("New output     : {}".format(new_h.shape))
            print("New state      : {}".format(c.shape))
            print("")
        return new_h, c
