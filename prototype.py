import numpy as np


raw_var = np.array([
    [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]
    ]
])

weight_one = np.ones((3, 4)) * 2
weight_two = np.ones((3, 4)) * 3

var_one = raw_var * weight_one
var_two = raw_var * weight_two


var_one = np.expand_dims(
    var_one,
    2
)

var_two = np.expand_dims(
    var_two,
    1
)

var_sum = var_one + var_two

tanh_sum = np.tanh(var_sum)

adopter_weight = np.ones((4,))

weight_matrix = np.dot(tanh_sum, adopter_weight)

exp_weight_matrix = np.exp(weight_matrix)

exp_weight_sum = np.sum(exp_weight_matrix, 2, keepdims=True)

normed_weight = exp_weight_matrix / exp_weight_sum

weighted_var_matrix = np.expand_dims(normed_weight, -1) * np.expand_dims(raw_var, 1)

weighted_var = np.sum(weighted_var_matrix, 1)

print("")