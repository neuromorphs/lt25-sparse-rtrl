import jax.numpy as jnp
import numpy as np
from jax import custom_jvp, jacfwd, jacrev, grad
import jax.nn as jnn

from models import event_fn


def test_scalar_event_fn():
    xs, ys, grad_xs = [], [], []
    for x in np.linspace(-2, 2, 100):
        xs.append(x)
        ys.append(event_fn(x))
        grad_xs.append(grad(event_fn)(x))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xs, ys, label='Event Fn')
    ax.plot(xs, grad_xs, label='Gradient')
    fig.show()


def test_scalar_event_fn_plus():
    fn = lambda x: event_fn(jnn.tanh(x))

    xs, ys, grad_xs = [], [], []
    for x in np.linspace(-3, 3, 100):
        xs.append(x)
        ys.append(fn(x))
        grad_xs.append(grad(fn)(x))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xs, ys, label='Event Fn')
    ax.plot(xs, grad_xs, label='Gradient')
    fig.show()


def test_event_fn_direct():
    x = np.random.randn(3)
    W = np.random.randn(3, 3)

    z = event_fn(x)
    jz = jacfwd(event_fn)(x)

    print("Ho ", z, "\n", jz)


def test_lambda():
    x = np.random.randn(3)
    W = np.random.randn(3, 3)
    ## First, use sigmoid directly

    fn1 = lambda xx: event_fn(jnn.tanh(W @ xx))
    jac_fn_1 = jacfwd(fn1)

    ye = fn1(x)
    j1e = jac_fn_1(x)

    ## Now, add one level of indirection using a lambda function

    fn2 = lambda xx, w: event_fn(jnn.tanh(w @ xx))
    jac_fn2 = jacfwd(fn2)

    y = fn2(x, W)
    j1 = jac_fn2(x, W)

    # print(y, ye)
    # print(j1, j1e)

    assert (y == ye).all()
    assert (j1 == j1e).all()


# test_scalar_event_fn()
test_scalar_event_fn_plus()
