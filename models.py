import functools
from functools import partial  # pylint: disable=g-importing-member
from typing import Any, Callable, Optional, Tuple

import jax
import optax
from flax import linen as nn
from flax.linen.activation import tanh
from flax.linen.initializers import orthogonal
from flax.linen.initializers import zeros
from flax.linen.linear import Dense
from flax.linen.linear import default_kernel_init
from flax.linen.recurrent import RNNCellBase
from jax import numpy as jnp, value_and_grad, jacfwd
from jax import random

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any


class RNNCell(RNNCellBase):
    activation_fn: Callable[..., Any] = tanh
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = orthogonal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        h = carry
        n_hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(Dense,
                          features=n_hidden_features,
                          use_bias=True,
                          kernel_init=self.recurrent_kernel_init,
                          bias_init=self.bias_init,
                          dtype=self.dtype,
                          param_dtype=self.param_dtype)
        dense_i = partial(Dense,
                          features=n_hidden_features,
                          use_bias=False,
                          kernel_init=self.kernel_init,
                          dtype=self.dtype,
                          param_dtype=self.param_dtype)
        new_h = self.activation_fn(dense_i(name='ig')(inputs) + dense_h(name='hg')(h))
        return new_h, new_h

    @staticmethod
    def initialize_carry(rng, batch_dims, size, init_fn=zeros):
        mem_shape = batch_dims + (size,)
        return init_fn(rng, mem_shape)


class SimpleRNN(nn.Module):
    """A simple unidirectional RNN."""

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        rnn = RNNCell()
        fn = lambda rnn, carry, x: rnn(carry, x)
        new_carry, new_out = fn(rnn, carry, x)
        J = jacfwd(fn, argnums=(0, 1), has_aux=False)(rnn, carry, x)
        return new_carry, (new_out, J)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        # Use fixed random key since default state init fn is just zeros.
        return RNNCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, hidden_size)


class SimpleRNNClassifier(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, carry, x):
        rnn_state, (y, J) = SimpleRNN()(carry, x)
        logits = nn.Dense(features=self.output_size)(y)
        # Sample the predicted token using a categorical distribution over the
        # logits.
        categorical_rng = self.make_rng('rnn')
        predicted_token = jax.random.categorical(categorical_rng, logits)
        # Convert to one-hot encoding.
        prediction = jax.nn.one_hot(
            predicted_token, self.output_size, dtype=jnp.float32)
        return logits, prediction, J

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        return SimpleRNN.initialize_carry(batch_dims, hidden_size)


if __name__ == '__main__':
    seq_len, batch_size, in_size, hidden_size, output_size = 11, 7, 3, 5, 2
    key_1, key_2, key_3 = random.split(random.PRNGKey(0), 3)

    xs = random.uniform(key_1, (batch_size, seq_len, in_size))
    ys = random.randint(key_1, (batch_size, 1), 0, 2)


    def cross_entropy_loss(*, logits, labels):
        labels_onehot = jax.nn.one_hot(labels, num_classes=output_size)
        return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


    def compute_metrics(*, logits, labels):
        loss = cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
        }
        return metrics


    model = SimpleRNNClassifier(output_size=output_size)
    init_carry = SimpleRNNClassifier.initialize_carry((batch_size,), hidden_size)
    params = model.init({'params': key_2, 'rnn': key_3}, init_carry, xs)
    logits, prediction, J = model.apply(params, init_carry, xs, rngs={'rnn': key_3})

    loss = cross_entropy_loss(logits=logits[:, -1, ...], labels=ys)
    metrics = compute_metrics(logits=logits[:, -1, ...], labels=ys)

    print(metrics)
    # print(J)
