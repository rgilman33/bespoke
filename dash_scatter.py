from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

from constants import *
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import io
import base64
import dash_bootstrap_components as dbc

import numpy as np

from PIL import Image

df = pd.read_pickle(f"{BESPOKE_ROOT}/tmp/dr_df.pkl")
images = np.load(f"{BESPOKE_ROOT}/tmp/dr_imgs.npy")
aux = na(np.load(f"{BESPOKE_ROOT}/tmp/dr_aux.npy"), AUX_PROPS)
wps = np.load(f"{BESPOKE_ROOT}/tmp/dr_wps.npy")
wps_p = np.load(f"{BESPOKE_ROOT}/tmp/dr_wps_p.npy")
aux_targets_p = na(np.load(f"{BESPOKE_ROOT}/tmp/aux_targets_p.npy"), AUX_TARGET_PROPS)
obsnet_outs = na(np.load(f"{BESPOKE_ROOT}/tmp/obsnet_outs.npy"), OBSNET_PROPS)

def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def update_hist(fig):
    fig.update_layout(
        # yaxis_title=title,
        autosize=False,
        width=400,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0)
    )

colorby_options = ['is_rw', 'tire_angle', 'unc_p', 'unc_p_normed', 'has_stop', 'has_lead', 'is_rw_pred']

x_prop, y_prop, colorby = 'traj_max_angle_p', 'unc_p', 'is_rw'
fig_scatter, hist0, hist1 = None, None, None
def refresh_charts():
    global fig_scatter, hist0, hist1, x_prop, y_prop, colorby
    fig_scatter = px.scatter(
            df, x=x_prop, y=y_prop,
            color=colorby,
            )
    fig_scatter.update_layout(
        # yaxis_title=title,
        autosize=False,
        width=800,
        height=500,
        xaxis_visible=True,
        yaxis_visible=True,
        margin=dict(l=20, r=20, t=0, b=0)
    )
    fig_scatter.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        marker=dict(
            size=6, line=dict(width=1, color="DarkSlateGrey")
        ),
    )

    hist0 = px.histogram(df, x=x_prop, color="is_rw", histnorm='probability density')
    hist1 = px.histogram(df, x=y_prop, color="is_rw", histnorm='probability density')
    hist0.update_layout(
        # yaxis_title=title,
        autosize=False,
        width=400,
        height=280,
        yaxis_visible=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    hist1.update_layout(
        # yaxis_title=title,
        autosize=False,
        width=400,
        height=280,
        yaxis_visible=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )


refresh_charts()

from models import *
from viz_utils import *

m = EffNet().to(device) 
model_stem = "1.18_avg"
m.load_state_dict(torch.load(f"{BESPOKE_ROOT}/models/m{model_stem}.torch"), strict=False)
m.set_for_viz()
m.eval()

actgrad_source_options = list(range(8))
actgrad_source = 5
actgrad_target_options = ['traj', 'stop', 'lead', 'none']
actgrad_target = 'traj'
def get_actgrad_by_ix(ix):
    m.viz_ix = actgrad_source
    actgrad = get_actgrad(m, images[ix], aux[ix], actgrad_target=actgrad_target, viz_loc=actgrad_source)
    return actgrad

dim_options = list(df.columns)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

blank_img = np.zeros_like(images[0, :,:,:3])
blank_img[:,:,:] = 200
im_url = np_image_to_base64(blank_img)
app.layout = html.Div(
    className="container",
    children=[
        dbc.Row(
            [
                dbc.Col([
                    html.Img(id='img', src=im_url, style={"width": f"{IMG_WIDTH}px"}),
                ]),
            ],
        ),
        dbc.Row(
            [
                dcc.Dropdown(
                        colorby_options,
                        value=colorby_options[0],
                        id='_colorby',
                        style={"width": "200px", "float":"left"}
                    ),
                dcc.Dropdown(
                        dim_options,
                        value=dim_options[dim_options.index("traj_max_angle_p")],
                        id='_x_prop',
                        style={"width": "200px", "float":"left"}
                    ),
                dcc.Dropdown(
                        dim_options,
                        value=dim_options[dim_options.index("unc_p")],
                        id='_y_prop',
                        style={"width": "200px", "float":"left"}
                    ),
                dcc.Dropdown(
                        actgrad_target_options,
                        value=actgrad_target_options[0],
                        id='_actgrad_target',
                        style={"width": "200px", "float":"left"}
                    ),
                dcc.Dropdown(
                        actgrad_source_options,
                        value=5,
                        id='_actgrad_source',
                        style={"width": "200px", "float":"left"}
                    ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    dcc.Graph(id="graph-5", figure=fig_scatter, clear_on_unhover=True, style={"float":"bottom"}),
                ]),
                dbc.Col([
                    dbc.Row([dcc.Graph(id="hist-0", figure=hist0)]),
                    dbc.Row([dcc.Graph(id="hist-1", figure=hist1)]),
                ]),
                dbc.Col([
                    html.Div(id="m0")
                ]),
            ]
        )
    ],
)
@app.callback(
    Output("img", "src"),
    Output("m0", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hd0):
    hover_data = [h for h in [hd0] if h is not None]
    
    if len(hover_data)==0:
        return np_image_to_base64(blank_img), str(0)
    else:
        hover_data = hover_data[0]["points"][0]

    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    img = images[num][:,:,:3]
    # Actgrad
    if actgrad_target=='none':
        img_for_view = img
    else:
        actgrad = get_actgrad_by_ix(num)
        img_for_view = combine_img_actgrad(img, actgrad)

    _aux = aux[num]
    _wps = None if df.iloc[num]["is_rw"] else wps[num]
    img_for_view = enrich_img(img=img_for_view, wps=_wps, wps_p=wps_p[num], aux=_aux,  
                                aux_targets_p=aux_targets_p[num], obsnet_out=obsnet_outs[num])

    im_url = np_image_to_base64(img_for_view)
    # info_box = info_box_from_ix(num)

    return im_url, f"ID: {str(num)}"

def info_box_from_ix(ix):
    return [html.P(f"{k}: {v}") for k,v in df.iloc[ix].items()]

@app.callback(
    Output('graph-5', 'figure'),
    Output('hist-0', 'figure'),
    Output('hist-1', 'figure'),

    Input('_colorby', 'value'),
    Input('_x_prop', 'value'),
    Input('_y_prop', 'value'),
    Input('_actgrad_target', 'value'),
    Input('_actgrad_source', 'value'),
    )
def update_graph(_colorby, _x_prop, _y_prop, _actgrad_target,_actgrad_source):
    global fig_scatter, x_prop, y_prop, colorby, actgrad_target, actgrad_source
    colorby, x_prop, y_prop, actgrad_target, actgrad_source = _colorby, _x_prop, _y_prop, _actgrad_target, _actgrad_source
    refresh_charts()

    return fig_scatter, hist0, hist1


if __name__ == '__main__':
    app.run_server(debug=True, port=8070)

