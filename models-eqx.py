import math
from typing import Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from equinox import Module, static_field
from jax import vmap, value_and_grad
from jaxtyping import Array
import equinox as eqx


class RNNCell(Module):
    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    input_size: int = static_field()
    hidden_size: int = static_field()

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            key: Optional["jax.random.PRNGKey"],
            **kwargs
    ):
        super().__init__(**kwargs)

        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        self.weight_ih = jrandom.uniform(
            ihkey, (hidden_size, input_size), minval=-lim, maxval=lim
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (hidden_size, hidden_size), minval=-lim, maxval=lim
        )
        self.bias = jrandom.uniform(
            bkey, (hidden_size,), minval=-lim, maxval=lim
        )

        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
            self, input: Array, hidden: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        new = jnn.tanh(self.weight_ih @ input + self.weight_hh @ hidden + self.bias)
        return new


class SimpleRNN(Module):
    cell: eqx.Module

    def __init__(self, **kwargs):
        # self.cell = RNNCell(**kwargs)
        self.cell = eqx.nn.GRUCell(**kwargs)

    def __call__(self, xs):
        scan_fn = lambda state, input: (self.cell(input, state), None)
        init_state = jnp.zeros(self.cell.hidden_size)
        final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
        return final_state


class SimpleRNNClassifier(Module):
    output_size: int = static_field()
    simple_rnn: eqx.Module
    readout: eqx.nn.Linear
    bias: jnp.ndarray

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 key: Optional["jax.random.PRNGKey"]):
        rkey, lkey = jrandom.split(key, 2)
        self.output_size = output_size
        self.simple_rnn = SimpleRNN(input_size=input_size, hidden_size=hidden_size, key=rkey)
        self.readout = eqx.nn.Linear(hidden_size, output_size, key=lkey)
        self.bias = jnp.zeros(output_size)

    def __call__(self, x):
        final_state = self.simple_rnn(x)
        logits = self.readout(final_state) + self.bias

        return logits


def get_data(dataset_size, seq_len, *, key):
    # t = jnp.linspace(0, 2 * math.pi, seq_len)
    # offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    # x1 = jnp.sin(t + offset) / (1 + t)
    # x2 = jnp.cos(t + offset) / (1 + t)
    # y = jnp.ones((dataset_size, 1))
    #
    # half_dataset_size = dataset_size // 2
    # x1 = x1.at[:half_dataset_size].multiply(-1)
    # y = y.at[:half_dataset_size].set(0)
    # x = jnp.stack([x1, x2], axis=-1)

    x = jrandom.uniform(key, (dataset_size, seq_len, in_size))
    y = jrandom.randint(key, (dataset_size, 1), 0, 2)

    return x, y


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


if __name__ == '__main__':
    dataset_size = 10000
    seed = 5678

    seq_len, batch_size, in_size, hidden_size, output_size = 11, 32, 2, 16, 2
    data_key, model_key, loader_key = jrandom.split(jrandom.PRNGKey(seed), 3)

    xs, ys = get_data(dataset_size, seq_len=seq_len, key=data_key)


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


    model = SimpleRNNClassifier(input_size=in_size, hidden_size=hidden_size, output_size=output_size, key=model_key)
    # logits = vmap(model)(xs)

    # loss = cross_entropy_loss(logits=logits, labels=ys)
    # metrics = compute_metrics(logits=logits, labels=ys)

    # print(metrics)

    learning_rate = 3e-3
    steps = 200


    @eqx.filter_value_and_grad
    def compute_loss(model, xs, ys):
        logits = vmap(model)(xs)
        loss = cross_entropy_loss(logits=logits, labels=ys)
        return loss


    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    for step, (x, y) in zip(range(steps), iter_data):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum(jnp.argmax(pred_ys, axis=-1) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")
