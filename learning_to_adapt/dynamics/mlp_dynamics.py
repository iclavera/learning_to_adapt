from learning_to_adapt.dynamics.core.layers import MLP
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.utils import tensor_utils
from learning_to_adapt.logger import logger
import time


class MLPDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """

    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = name
        self._dataset_train = None
        self._dataset_test = None

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]

        with tf.variable_scope(name):
            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # create MLP
            with tf.variable_scope('ff_model'):
                mlp = MLP(name,
                          output_dim=obs_space_dims,
                          hidden_sizes=hidden_sizes,
                          hidden_nonlinearity=hidden_nonlinearity,
                          output_nonlinearity=output_nonlinearity,
                          input_var=self.nn_input,
                          input_dim=obs_space_dims+action_space_dims)

            self.delta_pred = mlp.output_var  # shape: (batch_size, ndim_obs, n_models)

            self.loss = tf.reduce_mean(tf.square(self.delta_ph - self.delta_pred))
            self.optimizer = optimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = tensor_utils.compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        self._networks = [mlp]

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False):


        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 2
        else:
            delta = obs_next - obs

        # split into valid and test set
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
            t0 = time.time()

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            while True:
                try:
                    obs_batch, act_batch, delta_batch = sess.run(self.next_batch)
                    # run train op
                    batch_loss, _ = sess.run([self.loss, self.train_op],
                                                   feed_dict={self.obs_ph: obs_batch,
                                                              self.act_ph: act_batch,
                                                              self.delta_ph: delta_batch})

                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:
                    obs_test = self._dataset_test['obs']
                    act_test = self._dataset_test['act']
                    delta_test = self._dataset_test['delta']

                    # compute validation loss
                    feed_dict = {self.obs_ph: obs_test,
                                 self.act_ph: act_test,
                                 self.delta_ph: delta_test}
                    valid_loss = sess.run(self.loss, feed_dict=feed_dict)

                    if valid_loss_rolling_average is None:
                        valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = 2 * valid_loss
                        if valid_loss < 0:
                            valid_loss_rolling_average = valid_loss/1.5  # set initial rolling to a higher value avoid too early stopping
                            valid_loss_rolling_average_prev = valid_loss/2

                    valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                                 + (1.0-rolling_average_persitency)*valid_loss

                    if verbose:
                        logger.log("Training DynamicsModel - finished epoch %i --"
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

    def predict(self, obs, act):
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = np.array(self.f_delta_pred(obs, act))
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = np.array(self.f_delta_pred(obs, act))

        assert delta.ndim == 2

        pred_obs = obs_original + delta

        return pred_obs

    def _data_input_fn(self, obs, act, delta, batch_size=500, buffer_size=100000):

        assert obs.ndim == act.ndim == delta.ndim == 2, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == delta.shape[1], "obs and obs_next must have same length along axis 1 "

        self.obs_dataset_ph = tf.placeholder(tf.float32, (None, obs.shape[1]))
        self.act_dataset_ph = tf.placeholder(tf.float32, (None, act.shape[1]))
        self.delta_dataset_ph = tf.placeholder(tf.float32, (None, delta.shape[1]))

        dataset = tf.data.Dataset.from_tensor_slices((self.obs_dataset_ph, self.act_dataset_ph, self.delta_dataset_ph))
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized

    def compute_normalization(self, obs, act, obs_next):
        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]
        delta = obs_next - obs
        assert delta.ndim == 2 and delta.shape[0] == obs_next.shape[0]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=0), np.std(obs, axis=0))
        self.normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        self.normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))


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
