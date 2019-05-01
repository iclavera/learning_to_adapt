import tensorflow as tf
import numpy as np
import scipy


def compile_function(inputs, outputs, log_name=None):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return run


def flatten(xs):
    return [x for y in xs for x in y]


def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [np.reshape(pair[0], pair[1]) for pair in zip(np.split(flattened, indices), tensor_shapes)]


def pad_tensor(x, max_len, mode='zero'):
    padding = np.zeros_like(x[0])
    if mode == 'last':
        padding = x[-1]
    return np.concatenate([
        x,
        np.tile(padding, (max_len - len(x),) + (1,) * np.ndim(x[0]))
    ])


def pad_tensor_n(xs, max_len):
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len, mode='zero'):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len, mode=mode)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len, mode=mode)
    return ret


def flatten_first_axis_tensor_dict(tensor_dict):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = flatten_first_axis_tensor_dict(tensor_dict[k])
        else:
            old_shape = tensor_dict[k].shape
            ret[k] = tensor_dict[k].reshape((-1,) + old_shape[2:])
    return ret


def high_res_normalize(probs):
    return [x / sum(map(float, probs)) for x in list(map(float, probs))]


def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def concat_tensor_list_subsample(tensor_list, f):
    return np.concatenate(
        [t[np.random.choice(len(t), int(np.ceil(len(t) * f)), replace=False)] for t in tensor_list], axis=0)


def concat_tensor_dict_list_subsample(tensor_dict_list, f):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list_subsample([x[k] for x in tensor_dict_list], f)
        else:
            v = concat_tensor_list_subsample([x[k] for x in tensor_dict_list], f)
        ret[k] = v
    return ret


def concat_tensor_list(tensor_list, recurrent=False):
    if recurrent:
        return np.array(tensor_list)
    else:
        return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def temporal_list_dict_to_paths_list_dict(tensor_list_dict):
    keys = list(tensor_list_dict[0].keys())
    num_paths = tensor_list_dict[0][keys[0]].shape[0]
    ret = [{} for _ in range(num_paths)]
    for k in keys:
        aux = np.array([d[k] for d in tensor_list_dict])
        dim = len(aux.shape)
        aux = aux.transpose((1, 0) + tuple(range(2, dim)))
        _ = [d.update({k: v}) for d, v in zip(ret, aux)]
    return ret

def split_tensor_dict_list(tensor_dict):
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def truncate_tensor_list(tensor_list, truncated_len):
    return tensor_list[:truncated_len]


def truncate_tensor_dict(tensor_dict, truncated_len):
    ret = dict()
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            ret[k] = truncate_tensor_dict(v, truncated_len)
        else:
            ret[k] = truncate_tensor_list(v, truncated_len)
    return ret

def from_onehot(v):
    return np.nonzero(v)[0][0]


def from_onehot_n(v):
    if len(v) == 0:
        return []
    return np.nonzero(v)[1]


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret


def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret

def cat_entropy(x):
    return -np.sum(x * np.log(x), axis=-1)


# compute perplexity for each row
def cat_perplexity(x):
    return np.exp(cat_entropy(x))


def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)

def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_return(x, discount):
    return np.sum(x * (discount ** np.arange(len(x))))

def weighted_sample(weights, objects):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    """
    # An array of the weights, cumulatively summed.
    cs = np.cumsum(weights)
    # Find the index of the first weight over a random value.
    idx = sum(cs < np.random.rand())
    return objects[min(idx, len(objects) - 1)]


def weighted_sample_n(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((-1, 1))).sum(axis=1)
    n_items = len(items)
    return items[np.minimum(k, n_items - 1)]


# compute softmax for each row
def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expx = np.exp(shifted)
    return expx / np.sum(expx, axis=-1, keepdims=True)

def center_advantages(advantages):
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)


def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8
