import math
from enum import Enum
from typing import Optional, Tuple, Callable

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
    pseudo_derivative_width = 1.
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
    output_jac: bool = static_field()

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            key: Optional["jax.random.PRNGKey"],
            output_jac=True,
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
        self.output_jac = output_jac

    def __call__(
            self, input: Array, hidden: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        # new = jnn.tanh(self.weight_ih @ input + self.weight_hh @ hidden + self.bias)
        fn = lambda h, w_hh: event_fn(jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        # fn = lambda h, w_hh: (jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))

        # fn = lambda w_hh, h: (jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        new = fn(hidden, self.weight_hh)

        if self.output_jac:
            jac = jax.jacfwd(fn, argnums=(0, 1))
            J, bar_M = jac(hidden, self.weight_hh)
        else:
            J, bar_M = None, None

        # print(self.weight_hh.shape, hidden.shape, j1.shape, j2.shape)
        return new, (new, bar_M, J)

    def init_carry(self):
        return jnp.zeros((self.hidden_size,))


class EGRUCell(Module):
    weight_ih: Array
    weight_hh: Array
    mask_hh: Array = static_field()
    threshold: Array
    bias: Optional[Array]
    input_size: int = static_field()
    hidden_size: int = static_field()
    alpha: float = 0.001
    output_jac: bool = static_field()
    output_fn: Callable = static_field()

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            key: Optional["jax.random.PRNGKey"],
            activity_sparse=True,
            weight_sparsity=0.,
            output_jac=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        ihkey, hhkey, bkey, thkey = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        xavier_uniform_initializer = jax.nn.initializers.glorot_normal()

        # self.weight_ih = jrandom.uniform(
        #     ihkey, (3 * hidden_size, input_size), minval=-lim, maxval=lim
        # )
        self.weight_ih = xavier_uniform_initializer(
            ihkey, (3 * hidden_size, input_size), jnp.float32
        )

        hhwkey, hhmkey = jrandom.split(hhkey, 2)
        # self.weight_hh = jrandom.uniform(
        #     hhwkey, (3 * hidden_size, hidden_size), minval=-lim, maxval=lim
        # )
        self.weight_hh = xavier_uniform_initializer(
            hhwkey, (3 * hidden_size, hidden_size), jnp.float32
        )
        s = weight_sparsity
        self.mask_hh = jrandom.choice(
            hhmkey, jnp.array([True, False]), (3 * hidden_size, hidden_size), p=jnp.array([1 - s, s])
        )

        self.bias = jrandom.uniform(
            bkey, (3 * hidden_size,), minval=-lim, maxval=lim
        )

        self.threshold = jrandom.normal(
            thkey, (hidden_size,)
        ) - 1.

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_jac = output_jac

        if activity_sparse:
            self.output_fn = event_fn
        else:
            self.output_fn = lambda x: jnp.ones_like(x)

    def __call__(
            self, input: Array, state: Tuple[Array, Array, Array], *, key: Optional["jax.random.PRNGKey"] = None
    ):
        thr = jnn.sigmoid(self.threshold)
        h, c, o, i = state
        hidden = o * c
        c_reset = c - (o * thr)

        iu, ir, ic = jnp.split(i, 3, -1)

        def fn(hh, model):
            lin_x = model.weight_ih @ input

            # w_hh = model.weight_hh[model.mask_hh]
            w_hh = jnp.where(model.mask_hh, model.weight_hh, jnp.zeros_like(model.weight_hh))
            # jax.debug.print("Percent zeros in w_hh: {m}", m=jnp.mean(jnp.isclose(w_hh, 0)))
            # w_hh = model.weight_hh
            lin_h = w_hh @ hh

            xu, xr, xc = jnp.split(lin_x, 3, -1)
            hu, hr, hc = jnp.split(lin_h, 3, -1)
            bu, br, bc = jnp.split(model.bias, 3)

            new_iu = model.alpha * iu + (1 - model.alpha) * (xu + hu + bu)
            new_u = jnn.sigmoid(new_iu)
            new_ir = model.alpha * ir + (1 - model.alpha) * (xr + hr + br)
            new_r = jnn.sigmoid(new_ir)
            new_ic = model.alpha * ic + (1 - model.alpha) * (xc + new_r * hc + bc)
            new_z = jnn.tanh(new_ic)

            new_i = jnp.concatenate([new_iu, new_ir, new_ic], -1)
            new_c = (1 - new_u) * c_reset + new_u * new_z

            new_o = self.output_fn(new_c - thr)
            new_h = new_o * new_c
            # new_h = new_o

            return (new_h, new_c, new_o, new_i), (new_h, new_c, new_o, new_i)

        # fn = lambda w_hh, h: (jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        # new_h, new_c, new_o, new_i = fn(hidden, self)

        if self.output_jac:
            jac = jax.jacfwd(fn, argnums=(0, 1), has_aux=True)
        else: 
            jac = fn

        res, (new_h, new_c, new_o, new_i) = jac(hidden, self)

        if not self.output_jac:
            res = (None, None), (None, None), (None, None), (None, None)

        (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi) = res

        # import ipdb
        # ipdb.set_trace()

        # print(self.weight_hh.shape, hidden.shape, j1.shape, j2.shape)
        return (new_h, new_c, new_o, new_i), (new_h, new_c, new_o, new_i, (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo), (Ji, bar_Mi))

    def init_carry(self):
        return jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)), jnp.zeros((3 * self.hidden_size,))


class CellType(Enum):
    RNN = 'rnn'
    EqxGRU = 'eqxgru'
    EGRU = 'egru'


class EqxRNN(eqx.Module):
    hidden_size: int = static_field()
    cell: eqx.Module
    linear: eqx.nn.Linear

    # init_fn: Callable
    # apply_fn: Callable

    def __init__(self, cell_type, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)

        self.hidden_size = hidden_size

        if cell_type == CellType.EqxGRU:
            self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        else:
            raise RuntimeError(f"Unknown cell type {cell_type}")

        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            c = self.cell(inp, carry)
            return c, c

        final_state, outs = lax.scan(f, hidden, input)

        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(final_state)), outs


class RNN(eqx.Module):
    hidden_size: int = static_field
    cell: eqx.Module

    ## NOTE: Wan't to keep linear layer here so that the RNN class is always compatible with eqx.* RNNs
    linear: eqx.nn.Linear

    # init_fn: Callable
    # apply_fn: Callable

    def __init__(self, 
                cell_type, 
                 in_size: int, 
                 out_size: int, 
                 hidden_size: int, 
                *, 
                key: Optional["jax.random.PRNGKey"],
                **kwargs
                 ):
        ckey, lkey = jrandom.split(key)

        self.hidden_size = hidden_size

        if cell_type == CellType.RNN:
            self.cell = RNNCell(in_size, hidden_size, key=ckey, **kwargs)
        elif cell_type == CellType.EGRU:
            self.cell = EGRUCell(in_size, hidden_size, key=ckey, **kwargs)
        else:
            raise RuntimeError(f"Unknown cell type {cell_type}")

        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)

    def __call__(self, input):
        hidden = self.cell.init_carry()

        def f(carry, inp):
            carry, out = self.cell(inp, carry)
            return carry, out

        if isinstance(self.cell, RNNCell):
            final_state, outs = lax.scan(f, hidden, input)
        elif isinstance(self.cell, EGRUCell):
            (h, c, o, i), outs = lax.scan(f, hidden, input)
            final_state = h

        pred = jax.nn.sigmoid(self.linear(final_state))
        return pred, outs

