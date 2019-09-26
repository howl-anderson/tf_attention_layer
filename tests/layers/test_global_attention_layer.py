from tf_attention_layer.layers.global_attentioin_layer import GlobalAttentionLayer
from tensorflow.python.keras import testing_utils


def test_global_attention_layer():
    testing_utils.layer_test(
        GlobalAttentionLayer,
        kwargs={},
        input_shape=(1, 3, 4)
    )


def test_global_attention_layer_correctness():
    pass