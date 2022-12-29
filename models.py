import math
from typing import Optional, Tuple

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

    def init_carry(self):
        return jnp.zeros((self.hidden_size,))


class EGRUCell(Module):
    weight_ih: Array
    weight_hh: Array
    threshold: Array
    bias: Optional[Array]
    input_size: int = static_field()
    hidden_size: int = static_field()
    alpha: float = 0.001

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            key: Optional["jax.random.PRNGKey"],
            **kwargs
    ):
        super().__init__(**kwargs)

        ihkey, hhkey, bkey, thkey = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        self.weight_ih = jrandom.uniform(
            ihkey, (3 * hidden_size, input_size), minval=-lim, maxval=lim
        )
        self.weight_hh = jrandom.uniform(
            hhkey, (3 * hidden_size, hidden_size), minval=-lim, maxval=lim
        )
        self.bias = jrandom.uniform(
            bkey, (3 * hidden_size,), minval=-lim, maxval=lim
        )

        self.threshold = jrandom.normal(
            thkey, (hidden_size,)
        )

        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
            self, input: Array, state: Tuple[Array, Array, Array], *, key: Optional["jax.random.PRNGKey"] = None
    ):
        thr = jnn.sigmoid(self.threshold)
        c, o, i = state
        hidden = o * c
        c_reset = c - (o * thr)

        iu, ir, ic = jnp.split(i, 3, -1)

        def fn(hh, w_hh):
            lin_x = self.weight_ih @ input
            lin_h = w_hh @ hh
            xu, xr, xc = jnp.split(lin_x, 3, -1)
            hu, hr, hc = jnp.split(lin_h, 3, -1)
            bu, br, bc = jnp.split(self.bias, 3)

            new_iu = self.alpha * iu + (1 - self.alpha) * (xu + hu + bu)
            new_u = jnn.sigmoid(new_iu)
            new_ir = self.alpha * ir + (1 - self.alpha) * (xr + hr + br)
            new_r = jnn.sigmoid(new_ir)
            new_ic = self.alpha * ic + (1 - self.alpha) * (xc + new_r * hc + bc)
            new_z = jnn.tanh(new_ic)

            new_i = jnp.concatenate([new_iu, new_ir, new_ic], -1)
            new_c = (1 - new_u) * c_reset + new_u * new_z

            new_o = event_fn(new_c - thr)
            # new_h = new_o * new_c

            return new_c, new_o, new_i

        # fn = lambda w_hh, h: (jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        new_c, new_o, new_i = fn(hidden, self.weight_hh)

        jac = jax.jacfwd(fn, argnums=(0, 1))
        # res = jac(hidden, self.weight_hh)
        res = (None, None), (None, None), (None, None)
        (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi) = res

        # print(self.weight_hh.shape, hidden.shape, j1.shape, j2.shape)
        return (new_c, new_o, new_i), (new_c, new_o, new_i, (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi))

    def init_carry(self):
        return jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)), jnp.zeros((3 * self.hidden_size,))


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
        # self.cell = RNNCell(in_size, hidden_size, key=ckey)
        self.cell = EGRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = self.cell.init_carry()

        def f(carry, inp):
            carry, out = self.cell(inp, carry)
            return carry, out

        # final_state, outs = lax.scan(f, hidden, input)
        (c, o, i), outs = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(c * o) + self.bias), outs
