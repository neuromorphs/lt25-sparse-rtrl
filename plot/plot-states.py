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

fig, ax = plt.subplots()
im = ax.imshow(states[b].T)
fig.colorbar(im)
fig.show()