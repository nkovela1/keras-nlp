import os

os.environ["KERAS_BACKEND"] = "jax"

import tensorflow as tf
import jax

from jax.experimental import jax2tf
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.tests.test_case import TestCase

class GPT2ExportTest(TestCase):
    def test_export(self):
        tf.debugging.disable_traceback_filtering()

        m = tf.__internal__.tracking.AutoTrackable()

        gpt_model = GPT2Backbone.from_preset("gpt2_base_en")
        state_vars = tf.nest.map_structure(tf.Variable, gpt_model.variables)
        m.vars = tf.nest.flatten(state_vars)

        call_fn = jax2tf.convert(
            gpt_model.call,
            polymorphic_shapes=[{'token_ids': '(b, d)', 'padding_mask': '(b, d)'}]
        )

        @tf.function(
            autograph=False,
            input_signature=[{
                "token_ids": tf.TensorSpec(shape=(None, None), dtype="int32"),
                "padding_mask": tf.TensorSpec(shape=(None, None), dtype="int32"),
            }]
        )
        def call(inputs):
            return call_fn(inputs)

        m.call = call
        tf.saved_model.save(m, "my_gpt_model")