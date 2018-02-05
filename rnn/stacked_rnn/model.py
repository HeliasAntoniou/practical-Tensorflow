import time

import numpy as np
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

    def __init__(self, num_steps):
        self.BATCH_SIZE = 100

        self.ptb = PTBDataset()

        self.NUM_CLASSES = self.ptb.vocab_size

        self.LEARNING_RATE = 1e-4

        self.NUM_STEPS = num_steps

    @timeit
    def build_graph(self,
                    state_size=100,
                    num_layers=3):
        res_dict = self._build_graph(state_size, num_layers)

        expected_keys = ["x", "y", "init_state", "predictions", "final_state", "total_loss", "train_step"]
        are_present = [el in res_dict.keys() for el in expected_keys]
        assert all(are_present), "Please return valid return arguments"

        res_dict["saver"] = tf.train.Saver()

        return res_dict

    def _build_graph(self,
                     state_size=100,
                     num_layers=3):
        raise NotImplementedError("Please implement the method in the subclasses")

    def train_network(self, g, num_epochs, verbose=True, save=False):
        tf.set_random_seed(2345)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            for idx, epoch in enumerate(self.ptb.gen_epochs(num_epochs, self.NUM_STEPS, self.BATCH_SIZE)):
                training_loss = 0
                steps = 0
                training_state = None
                for X, Y in epoch:
                    if X.shape != (self.BATCH_SIZE, self.NUM_STEPS) or Y.shape != (self.BATCH_SIZE, self.NUM_STEPS):
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
                    print("Average training loss for Epoch", idx+1, ":", training_loss / steps)
                training_losses.append(training_loss / steps)

            if isinstance(save, str):
                path = g['saver'].save(sess, save)
                print("Checkpoints saved at: {}".format(path))

        return training_losses

    def generate_characters(self, g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
        """ Accepts a current character, initial state"""

        assert len(prompt) >= self.NUM_STEPS, "Please provide input at least as large as the num of steps"

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g['saver'].restore(sess, checkpoint)

            state = None
            chars = [self.ptb.vocab_to_idx[el] for el in prompt]

            for i in range(num_chars):
                current_chars = chars[-self.NUM_STEPS:]

                if state is not None:
                    feed_dict = {g['x']: np.asarray([current_chars for _ in range(self.BATCH_SIZE)]),
                                 g['init_state']: state}
                else:
                    feed_dict = {g['x']: np.asarray([current_chars for _ in range(self.BATCH_SIZE)])}

                preds, state = sess.run([g['predictions'], g['final_state']], feed_dict)
                preds = preds[0, :]

                if pick_top_chars is not None:
                    p = np.squeeze(preds)
                    p[np.argsort(p)[:-pick_top_chars]] = 0
                    p = p / np.sum(p)
                    current_char = np.random.choice(self.ptb.vocab_size, 1, p=p)[0]
                else:
                    current_char = np.random.choice(self.ptb.vocab_size, 1, p=np.squeeze(preds))[0]

                chars.append(current_char)

        chars = map(lambda x: self.ptb.idx_to_vocab[x], chars)
        print("".join(chars))
        return "".join(chars)
