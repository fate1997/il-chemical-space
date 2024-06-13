from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go
import numpy as np
import pathlib

app = Dash(__name__)
server = app.server


this_dir = pathlib.Path(__file__).parent
feature_path = this_dir.parent / 'data/il_features.npz'

features = np.load(feature_path, allow_pickle=True)
# Extract the first 100 features
# features = {k: v[:50] for k, v in features.items()}
labels = features['cls']
images = features['images']
images_dict = {}
images_dict[0] = images[:500]
images_dict[1] = images[500:1000]
images_dict[2] = images[1000:]

traces = []
id2name = {0: 'Original', 1: 'Ion-level', 2: 'Overall-level'}
id2color = {0: '#219C90', 1: '#FF6969', 2: '#799351'}
for i in range(3):
    traces.append(go.Scatter(
        x=features['tsne-1'][i*500:(i+1)*500],
        y=features['tsne-2'][i*500:(i+1)*500],
        name=id2name[i],
        mode="markers",
        marker=dict(
            color=id2color[i],
            size=features['molar_weights'][i*500:(i+1)*500],
            sizeref=30,
            sizemode="diameter",
            opacity=0.8,
        ),
    ))
fig = go.Figure(data=traces)

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)
""
fig.update_layout(
    xaxis=dict(title='TSNE-1'),
    yaxis=dict(title='TSNE-2'),
    plot_bgcolor='rgba(255,255,255,0.1)',
    width=800,
    height=800,
    
)
""
app.title = "TSNE of IL features for generated novel molecules (white) and non-generated (blue)"
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    curve_number = pt["curveNumber"]

    im_url = images_dict[curve_number][num]

    children = [
        html.Div(children=[
            html.Img(src=im_url, style={"width": "100%"}),
        ],
        style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True, port=8057)