import math
from typing import Optional, Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from equinox import Module, static_field
from jaxtyping import Array

from jax import custom_jvp


@custom_jvp
def event_fn(x):
    return jnp.heaviside(x, x)


@event_fn.defjvp
def f_jvp(primals, tangents):
    dampening_factor = 0.7
    pseudo_derivative_width = 1.
    x, = primals
    x_dot, = tangents
    primal_out = event_fn(x)
    tangent_out = dampening_factor * jnp.maximum(1 - pseudo_derivative_width * jnp.abs(x), 0.) * x_dot
    return primal_out, tangent_out


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


def get_data(dataset_size, seq_len, *, key):
    t = jnp.linspace(0, 2 * math.pi, seq_len)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y


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
        # new = jnn.tanh(self.weight_ih @ input + self.weight_hh @ hidden + self.bias)
        # fn = lambda w_hh, h: event_fn(jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        fn = lambda w_hh, h: event_fn(jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        new = fn(self.weight_hh, hidden)
        jac = jax.jacfwd(fn, argnums=(0, 1))
        bar_M, J = jac(self.weight_hh, hidden)
        # print(self.weight_hh.shape, hidden.shape, j1.shape, j2.shape)
        return new, (new, bar_M, J)


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jnp.ndarray

    # init_fn: Callable
    # apply_fn: Callable

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        # self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.cell = RNNCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            carry, out = self.cell(inp, carry)
            return carry, out

        final_state, outs = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(final_state) + self.bias), outs


def train(
        dataset_size=10000,
        batch_size=32,
        learning_rate=3e-3,
        steps=1000,
        hidden_size=16,
        seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_value_and_grad(has_aux=True)
    def compute_loss_and_outputs(model, x, y):
        pred_y, outs = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y)), outs

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        (loss, outs), grads = compute_loss_and_outputs(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, outs

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state, outs = make_step(model, x, y, opt_state)
        print(outs)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


def main(
        dataset_size=10000,
        seq_len=11,
        batch_size=32,
        hidden_size=16,
        seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, seq_len=seq_len, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_jit
    def compute_loss_and_outputs(model, x, y):
        pred_y, outs = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y)), outs

    x, y = next(iter_data)
    loss, (states, bar_Ms, Js) = compute_loss_and_outputs(model, x, y)
    print("Shape states & bar_Ms & Js: ", states.shape, bar_Ms.shape, Js.shape)
    # print(Js[:, 0])
    loss = loss.item()
    print(f"loss={loss}")

    def compute_influence_matrix(carry, inp):
        M_prev = carry
        J, bar_M = inp
        M = jnp.einsum('bkl,blij->bkij', J, M_prev) + bar_M
        return M, M

    Js_tr = jnp.transpose(Js, (1, 0, 2, 3))
    bar_Ms_tr = jnp.transpose(bar_Ms, (1, 0, 2, 3, 4))

    print("Transpose shape bar_Ms & Js: ", bar_Ms_tr.shape, Js_tr.shape)

    _, Ms_tr = lax.scan(compute_influence_matrix, jnp.zeros_like(bar_Ms[:, 0]), (Js_tr, bar_Ms_tr))

    Ms = jnp.transpose(Ms_tr, (1, 0, 2, 3, 4))

    print("Ms.shape: ", Ms.shape)

    print("Mean value of states: ", jnp.mean(states))
    print("Percent zeros in states: ", jnp.mean(states == 0.))

    print("Percent zeros in Js: ", jnp.mean(jnp.isclose(Js, 0.)))
    print("Percent zeros in bar_Ms: ", jnp.mean(jnp.isclose(bar_Ms, 0.)))

    print("Percent zeros in Ms: ", jnp.mean(jnp.isclose(Ms, 0.)))

    # pred_ys, _ = jax.vmap(model)(xs)
    # num_correct = jnp.sum((pred_ys > 0.5) == ys)
    # final_accuracy = (num_correct / dataset_size).item()
    # print(f"final_accuracy={final_accuracy}")


# train()  # All right, let's run the code.
main()  # All right, let's run the code.
