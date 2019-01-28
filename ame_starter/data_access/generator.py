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
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

LAST_ID = -1


def get_last_row_id():
    return LAST_ID


def random_cycle_generator(array):
    while 1:
        samples = np.random.permutation(array)
        for sample in samples:
            yield sample


def make_generator(dataset, batch_size, shuffle=True, label_limit=None, num_losses=1):
    x, y = dataset

    if label_limit is not None:
        # Fixed random state to prevent different splits across runs.
        sss = StratifiedShuffleSplit(n_splits=1, test_size=label_limit, random_state=0)
        _, test_index = next(sss.split(x, y))
        x, y = x[test_index], y[test_index]

    num_steps = int(np.ceil(len(x) / float(batch_size)))
    num_samples = len(x)

    print("INFO: Loaded generator with", num_samples, "samples. Doing", num_steps, "steps of size", batch_size,
          file=sys.stderr)

    def inner_generator():
        global LAST_ID
        while True:
            if shuffle:
                samples = np.random.permutation(num_samples)
            else:
                samples = np.arange(num_samples)

            for _ in range(num_steps):
                batch_indices = samples[:batch_size]
                samples = samples[batch_size:]

                LAST_ID = batch_indices

                batch_x = x[batch_indices]
                batch_y = y[batch_indices]

                if num_losses != 1:
                    batch_y = [batch_y] * num_losses

                yield batch_x, batch_y

    return inner_generator(), num_steps
