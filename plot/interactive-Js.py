import pickle

import numpy as np
import plotly.graph_objects as go

# with open('sparse-data.p', 'wb') as f:
#     pickle.dump(dict(Ms=Ms, bar_Ms=bar_Ms, Js=Js, states=states), f)

with open('../sparse-data-100.p', 'rb') as f:
    obj = pickle.load(f)

Ms = np.array(obj['Ms'])
bar_Ms = obj['bar_Ms']
Js = obj['Js']
states = obj['states']

b = 1

# Create figure
fig = go.Figure()

nt = Js.shape[1]
nh = Js.shape[2]

print(np.min(Js[b]), np.max(Js[b]))

# Add traces, one for each slider step
for t in np.arange(0, nt):
    fig.add_trace(go.Heatmap(z=Js[b, t], zmin=-0.2, zmax=0.2, colorscale='Rainbow'))

# Make 10th trace visible
fig.data[1].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()
