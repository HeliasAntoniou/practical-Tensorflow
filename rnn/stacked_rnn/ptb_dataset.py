import numpy as np
import os
import urllib.request


class PTBDataset(object):

    def __init__(self):
        file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
        file_name = 'tinyshakespeare.txt'
        if not os.path.exists(file_name):
            urllib.request.urlretrieve(file_url, file_name)

        with open(file_name, 'r') as f:
            raw_data = f.read()
            print("Data length:", len(raw_data))

        vocab = set(raw_data)
        idx_to_vocab = dict(enumerate(vocab))
        vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

        self.data = [vocab_to_idx[c] for c in raw_data]

        self.vocab_size = len(vocab)

        del raw_data

    """
    Code downloaded from https://gist.github.com/spitis/2dd1720850154b25d2cec58d4b75c4a0
    """

    @staticmethod
    def ptb_iterator(raw_data_ptb, batch_size, num_steps, steps_ahead=1):
        """
        Iterate on the raw PTB data.
        This generates batch_size pointers into the raw PTB data, and allows
        minibatch iteration along these pointers.
        Args:
        raw_data_ptb: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
        Raises:
        ValueError: if batch_size or num_steps are too high.
        """
        raw_data_ptb = np.array(raw_data_ptb, dtype=np.int32)

        data_len = len(raw_data_ptb)

        # Integer division
        batch_len = data_len // batch_size

        # Now we create a data set with batch_size columns but each row being a very large row having
        # batch_len elements. The idea is to split each row in mini-batches later on!
        data = np.zeros([batch_size, batch_len], dtype=np.int32)

        # Pick an offset to start from. So we will skip the very first offset observations.
        offset = 0
        if data_len % batch_size:
            offset = np.random.randint(0, data_len % batch_size)

        # Populate each column of the data matrix
        for i in range(batch_size):
            data[i] = raw_data_ptb[batch_len * i + offset:batch_len * (i + 1) + offset]

        # How many batches we will yield per epoch.
        epoch_size = (batch_len - steps_ahead) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        # Yield each batch. Note that y vector is just x slided left by one.
        for i in range(epoch_size):
            x = data[:, i * num_steps:(i + 1) * num_steps]
            y = data[:, i * num_steps + 1:(i + 1) * num_steps + steps_ahead]
            yield (x, y)

        # In case you do not care last batch not to have the same length as the rest of the batches
        # if epoch_size * num_steps < batch_len - steps_ahead:
        #     yield (data[:, epoch_size * num_steps: batch_len - steps_ahead], data[:, epoch_size * num_steps + 1:])

    def gen_epochs(self, n, num_steps, batch_size):
        for _ in range(n):
            yield PTBDataset.ptb_iterator(self.data, batch_size, num_steps)