"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function


import numpy as np
from functools import partial
from keras.models import Model
from keras.regularizers import L1L2
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Dropout, Input, Reshape, concatenate, dot, Lambda, Conv2D
from ame_starter.models.losses import *
from ame_starter.models.mixture_soft_attention import MixtureSoftAttention


class ModelBuilder(object):
    @staticmethod
    def compile_model(model, learning_rate, optimizer="adam", loss_weights=list([1.0, 1.0, 1.0, 1.0]),
                      main_loss="mse", extra_loss=None, metrics={}, gradient_clipping_threshold=100):

        losses = main_loss

        if loss_weights is not None:
            losses = [losses] * len(loss_weights)

        if extra_loss is not None:
            if isinstance(extra_loss, list):
                for i in range(1, 1 + len(extra_loss)):
                    losses[i] = extra_loss[i - 1]
            else:
                losses[1] = extra_loss

        if optimizer == "rmsprop":
            opt = RMSprop(lr=learning_rate, clipvalue=gradient_clipping_threshold)
        elif optimizer == "sgd":
            opt = SGD(lr=learning_rate, nesterov=True, momentum=0.9, clipvalue=gradient_clipping_threshold)
        else:
            opt = Adam(lr=learning_rate, clipvalue=gradient_clipping_threshold)

        model.compile(loss=losses,
                      loss_weights=loss_weights,
                      optimizer=opt,
                      metrics=metrics)
        return model

    @staticmethod
    def build_mlp(input_layer, num_units=16, activation="selu", n_layers=1, p_dropout=0.0,
                  l2_weight=0.0, with_bn=False, with_bias=True, **kwargs):
        last_layer = input_layer
        for i in range(n_layers):
            last_layer = Dense(num_units,
                               kernel_regularizer=L1L2(l2=l2_weight),
                               bias_regularizer=L1L2(l2=l2_weight),
                               use_bias=with_bias)(last_layer)
            if with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=l2_weight),
                                                gamma_regularizer=L1L2(l2=l2_weight))(last_layer)
            last_layer = Activation(activation)(last_layer)

            if p_dropout != 0.0:
                last_layer = Dropout(p_dropout)(last_layer)
        return last_layer

    @staticmethod
    def build_mlp_expert(i, input_shape, output_dim, output_activation,
                         l2_weight=0.0, topmost_hidden_state=None, **kwargs):
        if topmost_hidden_state is None:
            input_layer = Input(shape=input_shape)
            topmost_hidden_state = ModelBuilder.build_mlp(input_layer, l2_weight=l2_weight, **kwargs)
        else:
            input_layer = topmost_hidden_state

        auxiliary_output = Dense(output_dim,
                                 name="auxiliary" + str(i),
                                 activation=output_activation)(topmost_hidden_state)

        return Model(inputs=[input_layer], outputs=[topmost_hidden_state, auxiliary_output])

    @staticmethod
    def get_output_activation(is_regression, output_dim):
        if is_regression:
            output_activation = "linear"
            output_tf_activation = lambda x: x
        else:
            if output_dim == 1:
                output_activation = "sigmoid"
                output_tf_activation = tf.nn.sigmoid
            else:
                output_activation = "softmax"
                output_tf_activation = tf.nn.softmax
        return output_activation, output_tf_activation

    @staticmethod
    def _attention_dot(x):
        a, r = x
        if len(a.shape) == 2:  # at least 3d
            a = K.expand_dims(a, axis=-1)
        weighted_input = a * r
        return K.sum(weighted_input, axis=1)

    @staticmethod
    def _get_expert_outputs_unoptimized(input_num_dimensions, last_layer, make_expert_fn,
                                        output_dim, output_activation, l2_weight, topmost_hidden_states=None,
                                        **kwargs):
        """
        An unoptimised routine for obtaining expert outputs in an AME. Provided for didactic reasons.
        This routine creates a large number of operations in the TF graph if there are more than >100 experts and
        can therefore be slow to compile. Use _get_expert_outputs_optimized for problems that require more than
        100 experts in a single model.
        """
        outputs, topmost_hidden_states = [], []
        for i in range(input_num_dimensions):
            expert_input_layer = Lambda(lambda x: x[:, i:i + 1])(last_layer)
            expert = make_expert_fn(i, (1,), output_dim, output_activation, l2_weight=l2_weight,
                                    topmost_hidden_state=topmost_hidden_states[i],
                                    **kwargs)
            topmost_hidden_state, auxiliary_output = expert(expert_input_layer)
            topmost_hidden_states.append(topmost_hidden_state)
            outputs.append(auxiliary_output)
        return outputs, topmost_hidden_states, []

    @staticmethod
    def _get_expert_auxiliary_predictions_unoptimized(output_dim, output_activation, topmost_hidden_states):
        """
        An unoptimised routine for obtaining experts' auxiliary predictions in an AME. Provided for didactic reasons.
        Like _get_expert_outputs_unoptimized, this routine creates a large number of operations in the TF graph and
        can therefore be slow to compile. Use _get_expert_auxiliary_predictions_optimized for problems that require
        more than 100 experts in a single model.
        """
        all_but_one_auxiliary_outputs = []
        for i in range(len(topmost_hidden_states)):
            all_but_one_output = concatenate([topmost_hidden_state
                                              for j, topmost_hidden_state
                                              in enumerate(topmost_hidden_states) if j != i])
            all_but_one_output = Dense(output_dim,
                                       activation=output_activation,
                                       name="1vk_auxiliary" + str(i))(all_but_one_output)
            all_but_one_auxiliary_outputs.append(all_but_one_output)
        return all_but_one_auxiliary_outputs, []

    @staticmethod
    def _get_expert_outputs_optimized(input_num_dimensions, last_layer, num_units,
                                      output_dim, output_tf_activation, topmost_hidden_states=None):
        """
        Method for obtaining expert outputs in an AME optimised for faster compilation speed.
        Reduces the number of operations in the TF graph to a number independent of the chosen number of experts
        by sharing the per-expert operations and looping over their associated weights instead.
        This method is less flexible than the unoptimised version since it
        requires that all experts share the same architecture.
        """
        from keras.initializers import he_normal, zeros

        extra_trainable_weights = []

        has_prebuilt_hidden_states = topmost_hidden_states is not None
        if not has_prebuilt_hidden_states:
            num_experts = input_num_dimensions
            w1 = tf.Variable(he_normal()((num_experts, 1, num_units)))
            b1 = tf.Variable(zeros()((num_experts, num_units)))
            extra_trainable_weights += [w1, b1]
        else:
            num_experts = len(topmost_hidden_states)
        w2 = tf.Variable(he_normal()((num_experts, num_units, output_dim)))
        b2 = tf.Variable(zeros()((num_experts, output_dim)))
        extra_trainable_weights += [w2, b2]

        if not has_prebuilt_hidden_states:
            topmost_hidden_states = tf.TensorArray(dtype=tf.float32, size=num_experts, dynamic_size=False)
        else:
            topmost_hidden_states = tf.stack(topmost_hidden_states, axis=0)

        outputs = tf.TensorArray(dtype=tf.float32, size=num_experts, dynamic_size=False)

        def loop_fun(x):
            i = tf.constant(0)

            c = lambda i, ths, o: tf.less(i, num_experts)

            def loop_body(i, topmost_hidden_states, outputs):
                if has_prebuilt_hidden_states:
                    topmost_hidden_state = topmost_hidden_states[i]
                else:
                    topmost_hidden_state = tf.nn.selu(tf.matmul(x[:, i:i + 1], w1[i]) + b1[i])
                    topmost_hidden_states = topmost_hidden_states.write(i, topmost_hidden_state)
                auxiliary_output = output_tf_activation(tf.matmul(topmost_hidden_state, w2[i]) + b2[i])
                outputs = outputs.write(i, auxiliary_output)
                return i + 1, topmost_hidden_states, outputs

            _, hidden_states, aux_outputs = tf.while_loop(c, loop_body, [i, topmost_hidden_states, outputs])
            return [hidden_states.stack() if not has_prebuilt_hidden_states else topmost_hidden_states,
                    aux_outputs.stack()]

        topmost_hidden_states, outputs = Lambda(loop_fun,
                                                output_shape=lambda _: [(num_experts, None, num_units),
                                                                        (num_experts, None, output_dim)])(last_layer)
        topmost_hidden_states = Lambda(lambda x: tf.unstack(x, num=num_experts, axis=0))(topmost_hidden_states)
        outputs = Lambda(lambda x: tf.unstack(x, num=num_experts, axis=0))(outputs)
        return outputs, topmost_hidden_states, extra_trainable_weights

    @staticmethod
    def _get_expert_auxiliary_predictions_optimized(output_dim, output_tf_activation, topmost_hidden_states):
        """
        Method for obtaining experts' auxiliary predictions in an AME optimised for faster compilation speed.
        Like _get_expert_outputs_unoptimized, this routine creates a large number of operations in the TF graph and
        can therefore be slow to compile. Use _get_expert_auxiliary_predictions_optimized for problems that require
        more than 100 experts in a single model.
        """
        from keras.initializers import he_normal, zeros

        num_experts = len(topmost_hidden_states)
        step_size = K.int_shape(topmost_hidden_states[0])[-1]
        w3 = tf.Variable(he_normal()((num_experts, step_size * (num_experts - 1), output_dim)))
        b3 = tf.Variable(zeros()((num_experts, output_dim)))
        extra_trainable_weights = [w3, b3]

        def apply_fully_connected(idx, x):
            return output_tf_activation(tf.matmul(x, w3[idx]) + b3[idx])

        def get_all_but_one_auxiliary_outputs(x):
            all_outputs = tf.concat(x, axis=-1)
            return [apply_fully_connected(
                idx,
                tf.concat((all_outputs[:, :idx * step_size],
                           all_outputs[:, (idx + 1) * step_size:]),
                          axis=-1)
            ) for idx in range(num_experts)]

        all_but_one_auxiliary_outputs = Lambda(get_all_but_one_auxiliary_outputs)(topmost_hidden_states)
        return all_but_one_auxiliary_outputs, extra_trainable_weights

    @staticmethod
    def build_ame_model(input_dim, output_dim, make_expert_fn=None, l2_weight=0.0, num_softmaxes=1,
                        num_units=36, granger_loss_weight=0.03, is_regression=True, fast_compile=True,
                        is_image=True, downsample_factor=4, learning_rate=0.0001, dropout=0.0, attention_dropout=0.2,
                        **kwargs):
        if make_expert_fn is None:
            make_expert_fn = ModelBuilder.build_mlp_expert

        output_activation, output_tf_activation = ModelBuilder.get_output_activation(is_regression, output_dim)

        input_layer, input_num_dimensions = Input(shape=input_dim), int(np.prod(input_dim))
        last_layer = input_layer

        if is_image:
            last_num_units = num_units
            for _ in range(downsample_factor // 2):
                # Apply downsampling convolutions for image data to reduce the number of total experts.
                # This reduces the compilation and training time at the cost of resolution
                # in the attention map.
                last_layer = Conv2D(last_num_units, kernel_size=2, strides=2, activation="elu",
                                    kernel_regularizer=L1L2(l2=l2_weight))(last_layer)
                if dropout != 0.0:
                    last_layer = Dropout(dropout)(last_layer)

                last_num_units *= 2

            num_units = K.int_shape(last_layer)[-1]
            num_pixel_experts = np.prod(K.int_shape(last_layer)[1:3])
            last_layer = Reshape((num_pixel_experts, num_units))(last_layer)
            topmost_hidden_states = Lambda(lambda x: tf.unstack(x, axis=1))(last_layer)
        else:
            topmost_hidden_states = None

        if fast_compile:
            outputs, topmost_hidden_states, extra_trainable_weights1 = \
                ModelBuilder._get_expert_outputs_optimized(input_num_dimensions, last_layer, num_units,
                                                           output_dim, output_tf_activation,
                                                           topmost_hidden_states=topmost_hidden_states)
            all_but_one_auxiliary_outputs, extra_trainable_weights2 = \
                ModelBuilder._get_expert_auxiliary_predictions_optimized(output_dim, output_tf_activation,
                                                                         topmost_hidden_states)
        else:
            outputs, topmost_hidden_states, extra_trainable_weights1 = \
                ModelBuilder._get_expert_outputs_unoptimized(input_num_dimensions, last_layer, make_expert_fn,
                                                             output_dim, output_activation, l2_weight,
                                                             topmost_hidden_states=topmost_hidden_states,
                                                             **kwargs)
            all_but_one_auxiliary_outputs, extra_trainable_weights2 = \
                ModelBuilder._get_expert_auxiliary_predictions_unoptimized(output_dim,
                                                                           output_activation,
                                                                           topmost_hidden_states)
        extra_trainable_weights = extra_trainable_weights1 + extra_trainable_weights2

        all_auxiliary_outputs = concatenate(topmost_hidden_states)
        all_auxiliary_outputs_layer = Dense(output_dim,
                                            activation=output_activation,
                                            name="all_auxiliary")

        # Extra trainable weights must be added to a trainable layer.
        # See https://stackoverflow.com/questions/46544329/keras-add-external-trainable-variable-to-graph
        all_auxiliary_outputs_layer.trainable_weights.extend(extra_trainable_weights)
        all_auxiliary_outputs = all_auxiliary_outputs_layer(all_auxiliary_outputs)

        combined_hidden_state = concatenate(topmost_hidden_states + outputs)

        attention_weights = MixtureSoftAttention(num_softmaxes=num_softmaxes,
                                                 num_independent_attention_mechanisms=len(outputs),
                                                 attention_dropout=attention_dropout,
                                                 name="mixture_attention_1",
                                                 u_regularizer=L1L2(l2=l2_weight),
                                                 w_regularizer=L1L2(l2=l2_weight),
                                                 activation="tanh",
                                                 normalised=True)(combined_hidden_state)

        if is_regression:
            concatenated_residuals = concatenate(outputs, axis=-1)
            concatenated_residuals = Reshape((len(outputs),))(concatenated_residuals)
            attention_weights = Reshape((len(outputs),), name="soft_attention_1")(attention_weights)
            output = dot([attention_weights, concatenated_residuals], axes=-1, name="combined")
        else:
            concatenated_residuals = Lambda(lambda x: K.stack(x, axis=-2))(outputs)
            output = Lambda(ModelBuilder._attention_dot, name="combined")([attention_weights, concatenated_residuals])

        granger_output = Lambda(lambda x: x, name="granger")(output)
        repeat_output = Lambda(lambda x: x, name="repeat")(all_auxiliary_outputs)

        if is_regression:
            main_loss = "mse"
            auxiliary_loss = absolute_error_loss
            metrics = {}
        else:
            main_loss = "binary_crossentropy" if output_dim == 1 else "categorical_crossentropy"
            auxiliary_loss = categorical_loss
            metrics = {"combined": "accuracy"}

        granger_loss = partial(granger_causal_loss,
                               attention_weights=attention_weights,
                               auxiliary_outputs=all_auxiliary_outputs,
                               all_but_one_auxiliary_outputs=all_but_one_auxiliary_outputs,
                               loss_function=auxiliary_loss)
        granger_loss.__name__ = "granger_causal_loss"

        # We optimise compilation speed by using one shared loss function for all auxiliary outputs.
        repeat_loss = partial(repeat_output_loss,
                              outputs=outputs + all_but_one_auxiliary_outputs + [all_auxiliary_outputs],
                              main_loss=main_loss)
        repeat_loss.__name__ = "repeat_loss"

        extra_losses = [granger_loss, repeat_loss]
        outputs = [output, granger_output, repeat_output]

        auxiliary_loss_weight = 1.0
        loss_weights = [(1 - granger_loss_weight), granger_loss_weight, auxiliary_loss_weight]

        model = Model(inputs=input_layer, outputs=outputs)
        return ModelBuilder.compile_model(model,
                                          learning_rate=learning_rate,
                                          main_loss=main_loss,
                                          extra_loss=extra_losses,
                                          loss_weights=loss_weights,
                                          metrics=metrics,
                                          # We found gradient clipping useful to combat exploding gradients
                                          # when using unbounded outputs, e.g. in the regression setting.
                                          gradient_clipping_threshold=100 if is_regression else 0)
