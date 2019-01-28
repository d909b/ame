#!/usr/bin/env python2
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

import sys
import pandas as pd
from os.path import join
from keras.callbacks import *
from keras.utils import to_categorical
from ame_starter.apps.util import time_function
from keras.datasets import mnist, cifar10, boston_housing
from ame_starter.models.model_builder import ModelBuilder
from ame_starter.apps.evaluate import EvaluationApplication
from ame_starter.apps.parameters import clip_percentage, parse_parameters
from ame_starter.data_access.generator import make_generator, get_last_row_id


class MainApplication(EvaluationApplication):
    def __init__(self, args):
        super(MainApplication, self).__init__(args)
        self.training_set, self.validation_set, self.input_shape, self.output_dim = self.get_data()

    def setup(self):
        super(MainApplication, self).setup()

    def get_data(self):
        dataset = self.args["dataset"].lower()
        if dataset == "mnist":
            print("INFO: Loading MNIST data.", file=sys.stderr)
            return self.get_data_mnist()
        elif dataset == "boston_housing":
            return self.get_data_boston_housing()

    def load_image_data(self, img_channels, img_rows, img_cols, num_classes, dataset):
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        if K.image_dim_ordering() == "th":
            input_shape = (img_channels, img_rows, img_cols)
        else:  # K.image_dim_ordering() == "tf":
            input_shape = (img_rows, img_cols, img_channels)

        x_train = x_train.reshape((x_train.shape[0],) + input_shape)
        x_train = x_train.astype('float32')
        x_train /= 255.

        min = np.min(y_train)
        mean, std = np.mean(x_train), np.std(x_train)
        x_train = (x_train - mean) / std

        # If 1 indexed y (e.g. SVHN)
        y_train -= min
        y_test -= min

        x_test = x_test.reshape((x_test.shape[0],) + input_shape)
        x_test = x_test.astype('float32')
        x_test /= 255.

        x_test = (x_test - mean) / std

        return (x_train, to_categorical(np.squeeze(y_train))), \
               (x_test, to_categorical(np.squeeze(y_test))), \
               input_shape, num_classes

    def get_data_mnist(self):
        return self.load_image_data(img_channels=1, img_rows=28, img_cols=28, num_classes=10, dataset=mnist)

    def get_data_cifar10(self):
        return self.load_image_data(img_channels=3, img_rows=32, img_cols=32, num_classes=10, dataset=cifar10)

    def get_data_svhn(self):
        outer_self = self

        class svhn(object):
            @staticmethod
            def load_data():
                from scipy.io import loadmat
                path = outer_self.args["svhn_dataset"]
                train_data = loadmat(join(path, "train_32x32.mat"))
                test_data = loadmat(join(path, "test_32x32.mat"))
                return (train_data["X"].T, train_data["y"]), (test_data["X"].T, test_data["y"])

        return self.load_image_data(img_channels=3, img_rows=32, img_cols=32, num_classes=10, dataset=svhn)

    def get_data_boston_housing(self):
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        mean, std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        return (x_train, np.squeeze(y_train)), (x_test, np.squeeze(y_test)), \
               (x_train.shape[-1],), 1

    def get_num_losses(self):
        return 3  # One output each for combined, granger and all auxiliary outputs.

    def make_train_generator(self, randomise=True, stratify=True):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        train_generator, train_steps = make_generator(dataset=self.training_set,
                                                      batch_size=batch_size,
                                                      num_losses=num_losses,
                                                      shuffle=randomise)

        return train_generator, train_steps

    def make_validation_generator(self, randomise=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        val_generator, val_steps = make_generator(dataset=self.validation_set,
                                                  batch_size=batch_size,
                                                  num_losses=num_losses,
                                                  shuffle=randomise)
        return val_generator, val_steps

    def make_test_generator(self, randomise=False, do_not_sample_equalised=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        test_generator, test_steps = make_generator(dataset=self.validation_set,
                                                    batch_size=batch_size,
                                                    num_losses=num_losses,
                                                    shuffle=randomise)
        return test_generator, test_steps

    def get_best_model_path(self):
        return join(self.args["output_directory"], "model.npz")

    def get_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.csv")

    def get_attribution_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_attributions.csv")

    def get_hyperopt_parameters(self):
        hyper_params = {}

        base_params = {
            # "seed": [0, 2**32-1],
            # "dropout": [0.0, 0.7],
            # "num_layers": [0, 2],
            # "l2_weight": (0.0, 0.0001, 0.00001),
            # "batch_size": (4, 8, 16, 32),
        }

        hyper_params.update(base_params)

        return hyper_params

    @time_function("time_steps")
    def time_steps(self, generator, num_steps=10):
        for _ in range(num_steps):
            _ = next(generator)

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started training feature extraction.", file=sys.stderr)

        with_tensorboard = self.args["with_tensorboard"]
        output_directory = self.args["output_directory"]
        n_jobs = int(np.rint(self.args["n_jobs"]))
        num_epochs = int(np.rint(self.args["num_epochs"]))
        learning_rate = float(self.args["learning_rate"])
        l2_weight = float(self.args["l2_weight"])
        batch_size = int(np.rint(self.args["batch_size"]))
        early_stopping_patience = int(np.rint(self.args["early_stopping_patience"]))
        num_layers = int(np.rint(self.args["num_layers"]))
        num_units = int(np.rint(self.args["num_units"]))
        dropout = float(self.args["dropout"])
        granger_loss_weight = float(self.args["granger_loss_weight"])
        best_model_path = self.get_best_model_path()

        network_params = {
            "output_directory": output_directory,
            "early_stopping_patience": early_stopping_patience,
            "num_layers": num_layers,
            "num_units": num_units,
            "dropout": dropout,
            "input_dim": self.input_shape,
            "output_dim": self.output_dim,
            "batch_size": batch_size,
            "best_model_path": best_model_path,
            "l2_weight": l2_weight,
            "learning_rate": learning_rate,
            "granger_loss_weight": granger_loss_weight,
            "with_tensorboard": with_tensorboard,
            "n_jobs": n_jobs
        }

        assert train_steps > 0, "You specified a batch_size that is bigger than the size of the train set."
        assert val_steps > 0, "You specified a batch_size that is bigger than the size of the validation set."

        if with_tensorboard:
            tb_folder = join(self.args["output_directory"], "tensorboard")
            tmp_generator, tmp_steps = val_generator, val_steps
            tb = [MainApplication.build_tensorboard(tmp_generator, tb_folder)]
        else:
            tb = []

        network_params["tensorboard_callback"] = tb

        model = ModelBuilder.build_ame_model(is_regression=self.args["dataset"] == "boston_housing",
                                             is_image=self.dataset_is_image(),
                                             **network_params)

        if self.args["load_existing"]:
            print("INFO: Loading existing model from", self.args["load_existing"], file=sys.stderr)
            model.load(self.args["load_existing"])

        if self.args["do_train"]:
            monitor = "val_combined_loss"
            history = model.fit_generator(generator=train_generator,
                                          steps_per_epoch=train_steps,
                                          epochs=num_epochs,
                                          validation_data=val_generator,
                                          validation_steps=val_steps,
                                          verbose=2,
                                          callbacks=[
                                              EarlyStopping(monitor=monitor,
                                                            patience=early_stopping_patience),
                                              ModelCheckpoint(best_model_path,
                                                              monitor=monitor,
                                                              save_weights_only=True),
                                              ReduceLROnPlateau(monitor=monitor, factor=np.sqrt(0.1),
                                                                cooldown=0, patience=9, min_lr=1e-5, verbose=1)
                                          ])

            # Reset to best encountered weights.
            model.load_weights(best_model_path)
        else:
            history = {
                "val_acc": [],
                "val_loss": [],
                "val_combined_loss": [],
                "acc": [],
                "loss": [],
                "combined_loss": []
            }
        return model, history

    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="test"):
        if with_print:
            print("INFO: Started evaluation.", file=sys.stderr)

        scores = model.evaluate_generator(test_generator, test_steps)
        scores = dict(zip(model.metrics_names, scores))
        print("Eval score:", scores, file=sys.stderr)
        return scores

    def save_predictions(self, model):
        print("INFO: Saving model predictions.", file=sys.stderr)

        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])

        main_output_index = 0
        generators = [self.make_train_generator, self.make_validation_generator, self.make_test_generator]
        generator_names = ["train", "val", "test"]
        for generator_fun, generator_name in zip(generators, generator_names):
            generator, steps = generator_fun(randomise=False)
            steps = int(np.rint(steps * fraction_of_data_set))

            predictions = []
            for step in range(steps):
                x, y = next(generator)
                last_id = get_last_row_id()
                predictions.append([last_id, np.squeeze(model.predict(x)[main_output_index])])
            row_ids = np.hstack(map(lambda x: x[0], predictions))
            outputs = np.concatenate(map(lambda x: x[1], predictions), axis=0)
            file_path = self.get_prediction_path(generator_name)

            num_predictions = 1 if len(outputs.shape) == 1 else outputs.shape[-1]
            columns = ["prediction_" + str(i) for i in range(num_predictions)]
            df = pd.DataFrame(outputs, columns=columns, index=row_ids)
            df.to_csv(file_path)
            print("INFO: Saved model predictions to", file_path, file=sys.stderr)

    @staticmethod
    def get_attention_factors(model, x, layer_name):
        get_attention_output = K.function([model.layers[0].input, K.learning_phase()],
                                          [model.get_layer(layer_name).output])
        attention_factors = get_attention_output([x, 0])[0]
        return attention_factors

    def save_attributions(self, model, save_images=True, max_images_per_generator=20):
        print("INFO: Saving model attributions.", file=sys.stderr)

        attention_layer_name = "mixture_attention_1"
        output_directory = self.args["output_directory"]
        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])

        generators = [self.make_train_generator, self.make_validation_generator, self.make_test_generator]
        generator_names = ["train", "val", "test"]
        for generator_fun, generator_name in zip(generators, generator_names):
            generator, steps = generator_fun(randomise=False)
            steps = int(np.rint(steps * fraction_of_data_set))

            predictions = []
            for step in range(steps):
                x, y = next(generator)
                last_id = get_last_row_id()
                attention_factors = np.squeeze(
                    MainApplication.get_attention_factors(model, x, attention_layer_name)
                )
                predictions.append([last_id, attention_factors])

                if save_images and self.dataset_is_image():
                    from visualisation import plot_image

                    reshape_failed = False
                    sqrt_dim = int(np.sqrt(attention_factors.shape[-1]))
                    try:
                        attention_factors = attention_factors.reshape((attention_factors.shape[0], sqrt_dim, sqrt_dim))
                    except ValueError:
                        reshape_failed = True
                        print("ERROR: Could not reshape attention factors:", sys.exc_info()[0], file=sys.stderr)

                    if not reshape_failed:
                        input_shape = model.input_shape[1:3]
                        while attention_factors.shape[1] != input_shape[0]:
                            attention_factors = attention_factors.repeat(2, axis=1).repeat(2, axis=2)

                        for idx, sample, explanation in zip(last_id, x, attention_factors):
                            if idx > max_images_per_generator:
                                break
                            plot_image(np.squeeze(sample), generator_name + str(idx) + "_sample.png", output_directory)
                            plot_image(explanation, generator_name + str(idx) + "_attention.png", output_directory)

            row_ids = np.hstack(map(lambda x: x[0], predictions))
            outputs = np.concatenate(map(lambda x: x[1], predictions), axis=0)
            file_path = self.get_attribution_path(generator_name)
            df = pd.DataFrame(outputs, index=row_ids, columns=["a_" + str(i) for i in range(outputs.shape[-1])])
            df.to_csv(file_path)
            print("INFO: Saved model predictions to", file_path, file=sys.stderr)

    def dataset_is_image(self):
        dataset = self.args["dataset"]
        return dataset == "svhn" or dataset == "mnist" or dataset == "cifar"

    @staticmethod
    def build_tensorboard(tmp_generator, tb_folder):
        for a_file in os.listdir(tb_folder):
            file_path = join(tb_folder, a_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e, file=sys.stderr)

        tb = TensorBoard(tb_folder, write_graph=False, histogram_freq=1, write_grads=True, write_images=False)
        x, y = next(tmp_generator)

        tb.validation_data = x
        tb.validation_data[1] = np.expand_dims(tb.validation_data[1], axis=-1)
        if isinstance(y, list):
            num_targets = len(y)
            tb.validation_data += [y[0]] + y[1:]
        else:
            tb.validation_data += [y]
            num_targets = 1

        tb.validation_data += [np.ones(x[0].shape[0])] * num_targets + [0.0]
        return tb


if __name__ == '__main__':
    app = MainApplication(parse_parameters())
    app.run()
