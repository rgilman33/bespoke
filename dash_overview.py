from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

from constants import *
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import io
import base64
import pickle
import gzip

import numpy as np

from PIL import Image

df = pd.read_pickle(f"{BESPOKE_ROOT}/tmp/latents.pkl")
images = np.load(f"{BESPOKE_ROOT}/tmp/dr_imgs.npy")

def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def update_tsne_fig_layout(fig):
    fig.update_layout(
        # yaxis_title=title,
        autosize=False,
        width=600,
        height=600,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=20, r=20, t=0, b=0)
    )
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        marker=dict(
            size=3, line=dict(width=1, color="DarkSlateGrey")
        ),
    )

def get_scatter(colorby):
    fig = px.scatter_3d(
            df, x="tsne1", y="tsne2", z='tsne3',
            color=colorby, 
            # symbol='is_rw', wtf this was causing bug. Messes up hover num.
            #color_continuous_scale=scale
            )
    update_tsne_fig_layout(fig)
    return fig

colorby_options = ['is_rw', 'tire_angle', 'unc_p', 'unc_p_normed', 'approaching_stop', 'has_lead', 'is_rw_pred']
fig = get_scatter(colorby_options[0])


# fig_unc = px.scatter(
#         df, x="traj_max_angle", y="unc_p",
#         color='is_rw',
#         )
fig_unc = px.scatter(
        df, x="is_rw_pred", y="unc_p",
        color='is_rw',
        )
fig_unc.update_layout(
    # yaxis_title=title,
    autosize=False,
    width=800,
    height=600,
    xaxis_visible=True,
    yaxis_visible=True,
    margin=dict(l=20, r=20, t=0, b=0)
)
fig_unc.update_traces(
    hoverinfo="none",
    hovertemplate=None,
    marker=dict(
        size=6, line=dict(width=1, color="DarkSlateGrey")
    ),
)

app = Dash(__name__)

im_url = np_image_to_base64(images[0])
app.layout = html.Div(
    className="container",
    children=[
        html.Img(id='img', src=im_url, style={"width": f"{IMG_WIDTH}px", 'display': 'inline-block', 'margin': '0 auto'}),
        dcc.Dropdown(
                colorby_options,
                value=colorby_options[0],
                id='groupby_prop',
                style={"width": "200px"}
            ),
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True, style={"float":"left"}),
        dcc.Graph(id="graph-6", figure=fig_unc, clear_on_unhover=True, style={"float":"right"}),
    ],
)
@app.callback(
    Output("img", "src"),
    Input("graph-5", "hoverData"),
    Input("graph-6", "hoverData"),
)
def display_hover(hd0, hd1):
    hover_data = [h for h in [hd0, hd1] if h is not None]
    
    if len(hover_data)==0:
        return np_image_to_base64(images[0])
    else:
        hover_data = hover_data[0]["points"][0]

    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    return im_url


@app.callback(
    Output('graph-5', 'figure'),
    Input('groupby_prop', 'value'))
def update_graph(groupby_prop):

    fig = get_scatter(groupby_prop)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8060)

