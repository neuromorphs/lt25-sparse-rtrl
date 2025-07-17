import math
from enum import Enum
from functools import partial
from typing import Optional, Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module
from jax import custom_jvp
from jaxtyping import Array
import aqt.jax.v2.flax.aqt_flax as aqt
import aqt.jax.v2.config as aqt_config

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
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    output_jac: bool = eqx.field(static=True)

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

        return new, (new, bar_M, J)

    def init_carry(self):
        return jnp.zeros((self.hidden_size,))

from aqt.jax.v2.aqt_dot_general import CalibrationMode

fully_quantized = partial(
    aqt_config.fully_quantized, use_stochastic_rounding=False,
)

def q_dot_maybe(lhs_bits: Optional[int], rhs_bits: Optional[int], return_cfg=False):
    if lhs_bits is None and rhs_bits is None:
        return jnp.dot if not return_cfg else None
    else:
        assert lhs_bits == rhs_bits
        precision = (lhs_bits, rhs_bits)
        bwd_bits = max([e for e in precision if e is not None])
        dot_general = fully_quantized(fwd_bits=lhs_bits, bwd_bits=bwd_bits)
        if return_cfg:
            return dot_general
        else:
            return quant_dot_for_dot(dot_general)

def quant_dot_for_dot(general_dot):
    """Generate a jitted general_dot function to be used for dot products.
    Will contract on the last dimension of a, and the first dimension of b.
    This means that there are no batch dimensions, and all dimensions will be used
    for calibration in the quantization."""
    def _dot(a, b):
        # contr_dims = ((a.ndim-1,), (1,))  # batched version (not used)
        # batch_dims = ((0,), (0,))  # batched version (not used)
        contr_dims = ((a.ndim-1,), (0,))
        batch_dims = ((), ())
        return general_dot(a, b, (contr_dims, batch_dims))
    return jax.jit(_dot)

class EGRUCell(Module):
    weight_ih: Array
    weight_hh: Array
    threshold: Array
    bias: Array
    recurrent_bias: Array

    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    quantize: bool = eqx.field(static=True)
    output_jac: bool = eqx.field(static=True)
    output_fn: Callable = eqx.field(static=True)
    dot_general: Callable = eqx.field(static=True)

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            *,
            key: Optional["jax.random.PRNGKey"],
            quantize=False,
            activity_sparse=True,
            weight_sparsity=0.,
            output_jac=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.quantize = quantize
        if quantize:
            print("Quantizing")
            self.dot_general = q_dot_maybe(8, 8)
        else:
            self.dot_general = jnp.dot

        ihkey, hhkey, bkey, thkey = jrandom.split(key, 4)
        # lim = math.sqrt(1 / hidden_size)

        xavier_initializer = jax.nn.initializers.glorot_normal()

        # self.weight_ih = jrandom.uniform(
        #     ihkey, (3 * hidden_size, input_size), minval=-lim, maxval=lim
        # )
        self.weight_ih = xavier_initializer(
            ihkey, (3 * hidden_size, input_size), jnp.float32
        )

        hhwkey, hhmkey = jrandom.split(hhkey, 2)
        # self.weight_hh = jrandom.uniform(
        #     hhwkey, (3 * hidden_size, hidden_size), minval=-lim, maxval=lim
        # )
        self.weight_hh = xavier_initializer(
            hhwkey, (3 * hidden_size, hidden_size), jnp.float32
        )

        # s = weight_sparsity
        # self.mask_hh = jrandom.choice(
        #     hhmkey, jnp.array([True, False]), (3 * hidden_size, hidden_size), p=jnp.array([1 - s, s])
        # )

        # self.bias = jrandom.uniform(
        #     bkey, (3 * hidden_size,), minval=-lim, maxval=lim
        # )
        self.bias = jnp.zeros((3 * hidden_size,))
        self.recurrent_bias = jnp.zeros((3 * hidden_size,))
        # self.bias = jrandom.uniform(
        #     bkey, (3 * hidden_size,), minval=-lim, maxval=lim
        # )

        beta = 3
        thr_mean = 0.3
        alpha = beta * thr_mean / (1 - thr_mean)
        self.threshold = jrandom.beta(
            thkey, alpha, beta, (hidden_size,)
        ) - 1.

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_jac = output_jac

        if activity_sparse:
            self.output_fn = event_fn
        else:
            self.output_fn = lambda x: x

    def __call__(
            self, input_: Array, state: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        # h, c, o = state
        # hidden = h # o * c
        # c_reset = c - (o * thr)

        # iu, ir, ic = jnp.split(i, 3, -1)

        lin_x = self.dot_general(self.weight_ih, input_) + self.bias
        xu, xr, xc = jnp.split(lin_x, 3, -1)

        w_hh = self.weight_hh

        # Sould pass in o & c here for RTRL to get the right gradients (I think)
        def fn(state, model):
            c = state
            o, h, thr = self.c_to_oh(c)
            c_reset = c - (o * thr)

            # w_hh = model.weight_hh[model.mask_hh]
            # w_hh = jnp.where(model.mask_hh, model.weight_hh, jnp.zeros_like(model.weight_hh))
            # w_hh = model.weight_hh
            lin_h = self.dot_general(w_hh , h) + model.recurrent_bias
            # lin_h = w_hh @ h + model.recurrent_bias

            hu, hr, hc = jnp.split(lin_h, 3, -1)
            # bu, br, bc = jnp.split(model.bias, 3)

            # new_ir = model.alpha * ir + (1 - model.alpha) * (xr + hr) #  + br)
            new_r = jnn.sigmoid(xr + hr)

            # new_ic = model.alpha * ic + (1 - model.alpha) * (xc + new_r * hc) #  + bc)
            new_z = jnn.tanh(xc + new_r * hc)

            # new_iu = model.alpha * iu + (1 - model.alpha) * (xu + hu) #  + bu)
            new_u = jnn.sigmoid(xu + hu)
            new_c = new_u * c_reset + (1 - new_u) * new_z

            # return (new_h, new_c, new_o), (new_h, new_c, new_o)
            return new_c, new_c

        # fn = lambda w_hh, h: (jnn.tanh(self.weight_ih @ input + w_hh @ h + self.bias))
        # new_h, new_c, new_o, new_i = fn(hidden, self)

        if self.output_jac:
            jac = jax.jacfwd(fn, argnums=(0, 1), has_aux=True)
        else:
            jac = fn

        res, new_c = jac(state, self)

        if not self.output_jac:
            res = (None, None)

        Jc, bar_Mc = res

        # return (new_h, new_c, new_o), (new_h, new_c, new_o, (Jh, bar_Mh), (Jc, bar_Mc), (Jo, bar_Mo))
        return new_c, (new_c, (Jc, bar_Mc))

    def init_carry(self):
        return jnp.zeros((self.hidden_size,))

    def c_to_oh(self, c):
        thr = jax.lax.clamp(0., self.threshold, 1.)
        o = self.output_fn(c - thr)
        if not self.quantize:
            h = o * c
        else:
            h = o
        return o, h, thr



class MultiLayerCell(Module):
    cells: list[Module]

    def __call__(self, input_: Array, states: list[Array], *, key: Optional["jax.random.PRNGKey"] = None):
        inp = input_
        new_states = []
        outs = []
        for cell, state in zip(self.cells, states):
            o = c = None
            if isinstance(cell, eqx.nn.GRUCell):
                new_state = cell(inp, state)
                o = c = new_state
            elif isinstance(cell, EGRUCell):
                new_state, out = cell(inp, state)
                c, _ = out
                o, h, _ = cell.c_to_oh(c)
            inp = o
            outs.append(o)
            new_states.append(c)

        return new_states, outs


class CellType(Enum):
    RNN = 'rnn'
    EqxGRU = 'eqxgru'
    EGRU = 'egru'


class RNN(eqx.Module):
    multilayer_cell: MultiLayerCell

    ## NOTE: Want to keep linear layer here so that the RNN class is always compatible with eqx.* RNNs
    linear: eqx.nn.Linear

    hidden_size: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)


    def __init__(self,
                 cell_type,
                 in_size: int,
                 out_size: int,
                 hidden_size: int | list[int],
                 *,
                 key: Optional["jax.random.PRNGKey"],
                 **kwargs
                 ):
        ckey, lkey = jrandom.split(key)
        self.in_size = in_size

        if isinstance(hidden_size, int):
            # print(f"Initialising cell with input {in_size} and hidden size {hidden_size}")
            hidden_size = [hidden_size]

        cells = []
        for h in hidden_size:
            print(f"Initialising cell with input {in_size} and hidden size {h}")
            match cell_type:
                case CellType.RNN:
                    cell = RNNCell(in_size, h, key=ckey, **kwargs)
                case CellType.EGRU:
                    cell = EGRUCell(in_size, h, key=ckey, **kwargs)
                case CellType.EqxGRU:
                    cell = eqx.nn.GRUCell(in_size, h, key=ckey, **kwargs)
                case _:
                    raise RuntimeError(f"Unknown cell type {cell_type}")
            cells.append(cell)
            in_size = h

        self.multilayer_cell = MultiLayerCell(cells)

        self.linear = eqx.nn.Linear(hidden_size[-1], out_size, key=lkey)
        self.hidden_size = hidden_size

    def __call__(self, input_):
        final_state, outs = None, None

        hiddens = []
        for i, hsz in enumerate(self.hidden_size):
            if isinstance(self.multilayer_cell.cells[i], eqx.nn.GRUCell):
                hidden = jnp.zeros((hsz,))
            else:
                hidden = self.multilayer_cell.cells[i].init_carry()
            hiddens.append(hidden)

        def f(carry, inp):
            cs, os = self.multilayer_cell(inp, carry)
            return cs, os

        final_cs, outs = lax.scan(f, hiddens, input_)

        last_cell = self.multilayer_cell.cells[-1]
        if isinstance(last_cell, EGRUCell):
            alpha = 0.9
            tr_, _ = lax.scan(lambda carry, inp: (alpha * carry + (1 - alpha) * inp, None),
                              jnp.zeros((last_cell.hidden_size)), outs[-1])
            final_state = tr_
        else:
            final_state = final_cs[-1]
        pred = jax.nn.softmax(self.linear(final_state))
        return pred, outs
