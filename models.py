import math
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module, static_field
from jax import custom_jvp
from jaxtyping import Array


@custom_jvp
def event_fn(x):
    return jnp.heaviside(x, x)


@event_fn.defjvp
def f_jvp(primals, tangents):
    dampening_factor = 0.7
    pseudo_derivative_width = 0.5
    x, = primals
    x_dot, = tangents
    primal_out = event_fn(x)
    tangent_out = x_dot * dampening_factor * jnp.maximum(1. - jnp.abs(x) / pseudo_derivative_width, 0.)
    return primal_out, tangent_out


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
        fn = lambda h, w_hh: event_fn(jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))

        # fn = lambda w_hh, h: (jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        new = fn(hidden, self.weight_hh)

        jac = jax.jacfwd(fn, argnums=(0, 1))
        J, bar_M = jac(hidden, self.weight_hh)

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
