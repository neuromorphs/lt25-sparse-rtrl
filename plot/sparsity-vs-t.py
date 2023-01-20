import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax

with open('../sparse-data-egru-100.p', 'rb') as f:
    obj = pickle.load(f)

Ms = obj['Ms']
bar_Ms = obj['bar_Ms']
Js = obj['Js']
states = obj['states']

b = 1

nt = Ms.shape[1]
nb = Ms.shape[0]
nh = Ms.shape[2]

states_flat = states.reshape((nb * nt, nh))
states_sparsities = lax.map(lambda M: jnp.mean(jnp.isclose(M, 0)), states_flat).reshape((nb, nt, 1))

bar_Ms_flat = bar_Ms.reshape((nb * nt, nh, 3 * nh, nh))
bar_M_sparsities = lax.map(lambda M: jnp.mean(jnp.isclose(M, 0)), bar_Ms_flat).reshape((nb, nt, 1))

Ms_flat = Ms.reshape((nb * nt, nh, 3 * nh, nh))
M_sparsities = lax.map(lambda M: jnp.mean(jnp.isclose(M, 0)), Ms_flat).reshape((nb, nt, 1))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax = plt.subplots()
x = range(nt)
y, y_err = jnp.mean(M_sparsities, axis=0).ravel(), jnp.std(M_sparsities, axis=0).ravel()
ax.plot(x, y, marker='.', color=colors[1], label="M sparsity")
ax.fill_between(x, y - y_err, y + y_err,  alpha=0.2, interpolate=True, color=colors[1])

y, y_err = jnp.mean(states_sparsities, axis=0).ravel(), jnp.std(states_sparsities, axis=0).ravel()
ax.plot(x, y, marker='.', color=colors[2], label="States sparsity")
ax.fill_between(x, y - y_err, y + y_err,  alpha=0.2, interpolate=True, color=colors[2])

y, y_err = jnp.mean(bar_M_sparsities, axis=0).ravel(), jnp.std(bar_M_sparsities, axis=0).ravel()
ax.plot(x, y, marker='.', color=colors[3], label="bar_M sparsity")
ax.fill_between(x, y - y_err, y + y_err,  alpha=0.2, interpolate=True, color=colors[3])
ax.legend()
ax.set(ylim=[0,1])
fig.show()

