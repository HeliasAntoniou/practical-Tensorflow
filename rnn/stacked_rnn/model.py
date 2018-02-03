import time

import tensorflow as tf

from rnn.stacked_rnn.ptb_dataset import PTBDataset


def timeit(func):
    def func_wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print("Execution time of the method: {:.3f}secs".format((time.time() - start) / 1.000))
        return res

    return func_wrapper


class RNNModel(object):

    def __init__(self):
        self.BATCH_SIZE = 100

        self.ptb = PTBDataset()

        self.NUM_CLASSES = self.ptb.vocab_size

        self.LEARNING_RATE = 1e-4

    @timeit
    def build_graph(self,
                    state_size=100,
                    num_steps=200,
                    num_layers=3):
        res_dict = self._build_graph(state_size, num_steps, num_layers)

        expected_keys = ["x", "y", "init_state", "final_state", "total_loss", "train_step"]
        are_present = [el in res_dict.keys() for el in expected_keys]
        assert all(are_present), "Please return valid return arguments"

        return res_dict

    def _build_graph(self,
                     state_size=100,
                     num_steps=200,
                     num_layers=3):
        raise NotImplementedError("Please implement the method in the subclasses")

    def train_network(self, g, num_epochs, num_steps=200, verbose=True, save=False):
        tf.set_random_seed(2345)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            for idx, epoch in enumerate(self.ptb.gen_epochs(num_epochs, num_steps, self.BATCH_SIZE)):
                training_loss = 0
                steps = 0
                training_state = None
                for X, Y in epoch:
                    if X.shape != (self.BATCH_SIZE, num_steps) or Y.shape != (self.BATCH_SIZE, num_steps):
                        print("Mini-batch with non expected shape! We move on!")
                        continue
                    steps += 1

                    feed_dict = {g['x']: X, g['y']: Y}
                    if training_state is not None:
                        feed_dict[g['init_state']] = training_state
                    training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                                  g['final_state'],
                                                                  g['train_step']],
                                                                 feed_dict)
                    training_loss += training_loss_
                if verbose:
                    print("Average training loss for Epoch", idx, ":", training_loss / steps)
                training_losses.append(training_loss / steps)

            if isinstance(save, str):
                g['saver'].save(sess, save)

        return training_losses
