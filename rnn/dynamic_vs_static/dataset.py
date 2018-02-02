import numpy as np

from rnn.dynamic_vs_static.globals import Globals


def __gen_data(size=1000000):
    """
    It generates two np arrays based on the requirements of our application and the dependencies we want to inject.
    :param size: The size of our data set
    :return: A tuple X, and Y
    """
    # Create an np.array of shape (size, ) having randomly 0s and 1s
    x = np.asarray(np.random.choice(2, size=(size,)))
    y = []
    for i in range(size):
        threshold = 0.5
        if x[i-2] == 1:
            threshold += 0.5
        if x[i-6] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            y.append(0)
        else:
            y.append(1)
    return x, np.asarray(y)


def __gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    # It returns a generator that for each epoch returns a generator
    for i in range(n):
        yield __gen_batch(__gen_data(), Globals.BATCH_SIZE, num_steps)
