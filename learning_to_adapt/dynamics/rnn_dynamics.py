from learning_to_adapt.dynamics.core.layers import RNN
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.utils import tensor_utils
from learning_to_adapt.logger import logger
import time


class RNNDynamicsModel(Serializable):
    """
    Class for RNN continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(512,),
                 cell_type='lstm',
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 backprop_steps=50,
                 ):

        Serializable.quick_init(self, locals())
        self.recurrent = True

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency
        self.backprop_steps = backprop_steps

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = name
        self._dataset_train = None
        self._dataset_test = None

        # Determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        """ computation graph for training and simple inference """
        with tf.variable_scope(name):
            # Placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, None, obs_space_dims), name='obs_ph')
            self.act_ph = tf.placeholder(tf.float32, shape=(None, None, action_space_dims), name='act_ph')
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, None, obs_space_dims),
                                           name='delta_ph')

            # Concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=2)

            # Create RNN
            rnns = []
            delta_preds = []
            self.obs_next_pred = []
            self.hidden_state_ph = []
            self.next_hidden_state_var = []
            self.cell = []
            with tf.variable_scope('rnn_model'):
                rnn = RNN(name,
                          output_dim=self.obs_space_dims,
                          hidden_sizes=hidden_sizes,
                          hidden_nonlinearity=hidden_nonlinearity,
                          output_nonlinearity=output_nonlinearity,
                          input_var=self.nn_input,
                          input_dim=self.obs_space_dims + self.action_space_dims,
                          cell_type=cell_type,
                          )

            self.delta_pred = rnn.output_var
            self.hidden_state_ph = rnn.state_var
            self.next_hidden_state_var = rnn.next_state_var
            self.cell = rnn.cell
            self._zero_state = self.cell.zero_state(1, tf.float32)

            self.loss = tf.reduce_mean(tf.square(self.delta_pred - self.delta_ph))
            params = list(rnn.get_params().values())
            self._gradients_ph = [tf.placeholder(shape=param.shape, dtype=tf.float32) for param in params]
            self._gradients_vars = tf.gradients(self.loss, params)
            applied_gradients = zip(self._gradients_ph, params)
            self.train_op = optimizer(self.learning_rate).apply_gradients(applied_gradients)


            # Tensor_utils
            self.f_delta_pred = tensor_utils.compile_function([self.obs_ph, self.act_ph, self.hidden_state_ph],
                                                              [self.delta_pred, self.next_hidden_state_var])

        self._networks = [rnn]

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False):
        assert obs.ndim == 3 and obs.shape[2] == self.obs_space_dims
        assert obs_next.ndim == 3 and obs_next.shape[2] == self.obs_space_dims
        assert act.ndim == 3 and act.shape[2] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 3
        else:
            delta = obs_next - obs

        obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                         test_split_ratio=valid_split_ratio)

        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test)
            self._dataset_train = dict(obs=obs_train, act=act_train, delta=delta_train)
        else:
            self._dataset_test['obs'] = np.concatenate([self._dataset_test['obs'], obs_test])
            self._dataset_test['act'] = np.concatenate([self._dataset_test['act'], act_test])
            self._dataset_test['delta'] = np.concatenate([self._dataset_test['delta'], delta_test])

            self._dataset_train['obs'] = np.concatenate([self._dataset_train['obs'], obs_train])
            self._dataset_train['act'] = np.concatenate([self._dataset_train['act'], act_train])
            self._dataset_train['delta'] = np.concatenate([self._dataset_train['delta'], delta_train])

            # create data queue
        if self.next_batch is None:
            self.next_batch, self.iterator = self._data_input_fn(self._dataset_train['obs'],
                                                                 self._dataset_train['act'],
                                                                 self._dataset_train['delta'],
                                                                 batch_size=self.batch_size)

        valid_loss_rolling_average = None
        epoch_times = []

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            # initialize data queue
            feed_dict = {self.obs_dataset_ph: self._dataset_train['obs'],
                         self.act_dataset_ph: self._dataset_train['act'],
                         self.delta_dataset_ph: self._dataset_train['delta']}

            sess.run(self.iterator.initializer, feed_dict=feed_dict)

            # preparations for recording training stats
            batch_losses = []

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            t0 = time.time()
            while True:
                try:
                    obs_batch, act_batch, delta_batch = sess.run(self.next_batch)
                    hidden_batch = self.get_initial_hidden(obs_batch.shape[0])
                    seq_len = obs_batch.shape[1]

                    # run train op
                    all_grads = []
                    for i in range(0, seq_len, self.backprop_steps):
                        end_i = i + self.backprop_steps
                        feed_dict = {self.obs_ph: obs_batch[:, i:end_i, :],
                                     self.act_ph: act_batch[:, i:end_i, :],
                                     self.delta_ph: delta_batch[:, i:end_i, :]}
                        hidden_feed_dict = dict(zip(self.hidden_state_ph, hidden_batch))
                        feed_dict.update(hidden_feed_dict)

                        batch_loss, grads, hidden_batch = sess.run([self.loss, self._gradients_vars,
                                                              self.next_hidden_state_var], feed_dict=feed_dict)

                        all_grads.append(grads)
                        batch_losses.append(batch_loss)

                    grads = [np.mean(grad, axis=0) for grad in zip(*all_grads)]
                    feed_dict = dict(zip(self._gradients_ph, grads))
                    _ = sess.run(self.train_op, feed_dict=feed_dict)

                except tf.errors.OutOfRangeError:
                    obs_test = self._dataset_test['obs']
                    act_test = self._dataset_test['act']
                    delta_test = self._dataset_test['delta']
                    hidden_batch = self.get_initial_hidden(obs_test.shape[0])

                    # compute validation loss
                    feed_dict = {self.obs_ph: obs_test,
                                 self.act_ph: act_test,
                                 self.delta_ph: delta_test,
                                 self.hidden_state_ph: hidden_batch}
                    valid_loss = sess.run(self.loss, feed_dict=feed_dict)

                    if valid_loss_rolling_average is None:
                        valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = 2 * valid_loss
                        if valid_loss < 0:
                            valid_loss_rolling_average = valid_loss/1.5  # set initial rolling to a higher value avoid too early stopping
                            valid_loss_rolling_average_prev = valid_loss/2

                    valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                                 + (1.0-rolling_average_persitency)*valid_loss

                    epoch_times.append(time.time() - t0)
                    if verbose:
                        logger.log("Training RNNDynamicsModel - finished epoch %i --"
                                   "train loss: %.4f  valid loss: %.4f  valid_loss_mov_avg: %.4f  epoch time: %.2f"
                                   % (epoch, np.mean(batch_losses), valid_loss, valid_loss_rolling_average,
                                      time.time() - t0))
                    break

            if valid_loss_rolling_average_prev < valid_loss_rolling_average or epoch == epochs - 1:
                logger.log('Stopping Training of Model since its valid_loss_rolling_average decreased')
                break

            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv('AvgModelEpochTime', np.mean(epoch_times))
            logger.logkv('Epochs', epoch)

    def predict(self, obs, act, hidden_state):
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        obs, act = np.expand_dims(obs, 1), np.expand_dims(act, 1)

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta, next_hidden_state = self.f_delta_pred(obs, act, hidden_state)
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta, next_hidden_state = self.f_delta_pred(obs, act, hidden_state)

        delta = delta[:, 0, :]
        pred_obs = obs_original + delta

        return pred_obs, next_hidden_state

    def _data_input_fn(self, obs, act, delta, batch_size=500, buffer_size=100000):

        assert obs.ndim == act.ndim == delta.ndim == 3, "inputs must have 3 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == act.shape[1] == delta.shape[1], "inputs must have same length along axis 1"
        assert obs.shape[2] == delta.shape[2], "obs and obs_next must have same length along axis 1 "

        self.obs_dataset_ph = tf.placeholder(tf.float32, (None, None, obs.shape[2]))
        self.act_dataset_ph = tf.placeholder(tf.float32, (None, None, act.shape[2]))
        self.delta_dataset_ph = tf.placeholder(tf.float32, (None, None, delta.shape[2]))

        dataset = tf.data.Dataset.from_tensor_slices((self.obs_dataset_ph, self.act_dataset_ph, self.delta_dataset_ph))
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def get_initial_hidden(self, batch_size):
        sess = tf.get_default_session()
        state = sess.run(self._zero_state)
        if isinstance(state, tuple) and not isinstance(state, tf.contrib.rnn.LSTMStateTuple):
            hidden = []
            for _s in state:
                if isinstance(_s, tf.contrib.rnn.LSTMStateTuple):
                    _hidden_c = np.concatenate([_s.c] * batch_size)
                    _hidden_h = np.concatenate([_s.h] * batch_size)
                    _hidden = tf.contrib.rnn.LSTMStateTuple(_hidden_c, _hidden_h)
                else:
                    _hidden = np.concatenate([state] * batch_size)
                hidden.append(_hidden)
        else:
            if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
                hidden_c = np.concatenate([state.c] * batch_size)
                hidden_h = np.concatenate([state.h] * batch_size)
                hidden = tf.contrib.rnn.LSTMStateTuple(hidden_c, hidden_h)
            else:
                hidden = np.concatenate([state] * batch_size)
        return hidden

    def compute_normalization(self, obs, act, obs_next):
        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]
        assert obs.shape[1] == obs_next.shape[1] == act.shape[1]
        delta = obs_next - obs

        assert delta.ndim == 3 and delta.shape[2] == obs_next.shape[2] == obs.shape[2]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=(0, 1)), np.std(obs, axis=(0, 1)))
        self.normalization['delta'] = (np.mean(delta, axis=(0, 1)), np.std(delta, axis=(0, 1)))
        self.normalization['act'] = (np.mean(act, axis=(0, 1)), np.std(act, axis=(0, 1)))

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def train_test_split(obs, act, delta, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :]
