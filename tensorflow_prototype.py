import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

raw_var = tf.constant(np.array([
    [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]
    ]
], np.float64))

weight_one = tf.constant(np.ones((3, 4)) * 2)
weight_two = tf.constant(np.ones((3, 4)) * 3)

var_one = raw_var * weight_one
var_two = raw_var * weight_two

# checkpoint
_var_one = var_one.eval()
_var_two = var_two.eval()

var_one = tf.expand_dims(
    var_one,
    2
)

var_two = tf.expand_dims(
    var_two,
    1
)

var_sum = var_one + var_two

tanh_sum = tf.tanh(var_sum)

adopter_weight = tf.constant(np.ones((4,)))

weight_matrix = tf.tensordot(tanh_sum, adopter_weight, [3, 0])

# checkpoint
_weight_matrix = weight_matrix.eval()

exp_weight_matrix = tf.exp(weight_matrix)

exp_weight_sum = tf.reduce_sum(exp_weight_matrix, 2, keepdims=True)

normed_weight = exp_weight_matrix / exp_weight_sum

weighted_var_matrix = tf.expand_dims(normed_weight, -1) * tf.expand_dims(raw_var, 1)

weighted_var = tf.reduce_sum(weighted_var_matrix, 1)

# checkpoint
_weighted_var = weighted_var.eval()
print(_weighted_var)
