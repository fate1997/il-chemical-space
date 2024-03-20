from dash import dcc, html, Input, Output, no_update, Dash
import plotly.graph_objects as go
import numpy as np
import pathlib

app = Dash(__name__)
server = app.server


this_dir = pathlib.Path(__file__).parent
feature_path = this_dir.parent / 'il_features.npz'

features = np.load(feature_path, allow_pickle=True)
labels = list(map(lambda x: int(~x), features['cls']))

fig = go.Figure(data=[
    go.Scatter(
        x=features['tsne-1'],
        y=features['tsne-2'],
        mode="markers",
        marker=dict(
            colorscale='blues',
            color=labels,
            size=features['molar_weights'],
            colorbar={"title": "Is<br>generated?"},
            line={"color": "#444"},
            reversescale=True,
            sizeref=45,
            sizemode="diameter",
            opacity=0.8,
        ),
    )
])

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
app = Dash(__name__)
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

    im_url = features['images'][num]

    children = [
        html.Div(children=[
            html.Img(src=im_url, style={"width": "100%"}),
        ],
        style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True, port=8057)