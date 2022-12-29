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

# fig = go.Figure(data=go.Heatmap(z=Ms[b, 0]),
#                frames=[go.Frame(data=go.Heatmap(z=Ms[b, i])) for i in range(Ms.shape[1])])
# fig.update_layout(
#     updatemenus=[
#         dict(type="buttons", visible=True,
#         buttons=[dict(label="Play", method="animate", args=[None])]
#             )])
# fig.show()

# Create figure
fig = go.Figure()

nt = Ms.shape[1]
nh = Ms.shape[2]

# Add traces, one for each slider step
for t in np.arange(0, nt):
    # fig.add_trace(go.Heatmap(z=Ms[b, t].reshape(nh, nh ** 2), zmin=np.min(Ms[b]), zmax=np.max(Ms[b]), colorscale='Jet'))
    fig.add_trace(go.Heatmap(z=np.logical_not(np.isclose(Ms[b, t], 0)).astype(np.int8).reshape(nh, nh ** 2), zmin=0, zmax=1)) # , zmin=0, zmax=0.5, colorscale='Viridis'))

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
