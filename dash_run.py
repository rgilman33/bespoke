from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

from constants import *
from dash import dcc, html, Input, Output, no_update, ctx
import plotly.graph_objects as go
import io
import base64
import pickle
import gzip

import numpy as np

from PIL import Image
from dash_utils import *

run_ids = ["run_555b", "run_556d", "run_555a", "run_556a", "run_556b", "run_556c", "none"]
model_stem = "1.29_avg"
run_id = run_ids[0]
run_id_2 = 'none'

rollout = None
def update_rollout():
    global rollout
    rollout = load_object(f"{BESPOKE_ROOT}/tmp/{run_id}_{model_stem}_rollout.pkl")

update_rollout()

def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def get_scatter(name, prop, color, opacity=1., size=2):
    s = go.Scatter(
        name=name,
        x=list(range(len(prop))), 
        y=prop,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        )
    )
    return s
def update_timeline_fig_layout(fig, title='', xaxis_visible=False, height=120):
    fig.update_layout(
        yaxis_title=title,
        autosize=False,
        width=1800,
        height=height,
        xaxis_visible=xaxis_visible,
        margin=dict(l=20, r=20, t=0, b=0)
    )
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

fig, fig_unc, fig_lane_width, clfs, speed, dagger_shift = None, None, None, None, None, None
def refresh_charts():
    global fig, fig_unc, fig_lane_width, clfs, speed, dagger_shift
    # Tire angle
    fig = go.Figure(data=[get_scatter('tire_angle', rollout.aux[:, "tire_angle"]*-1, 'blue'), 
                            get_scatter('tire_angle_p', rollout.additional_results[:, "tire_angle_p"]*-1, 'orange')])
    update_timeline_fig_layout(fig, "Tire Angle")

    fig_unc = go.Figure(data=[
                            # get_scatter('loss', tl, 'aqua'), 
                            get_scatter('unc', rollout.obsnet_outs[:, "unc_p"], 'red'), 
                            # get_scatter('acts mean', rollout.final_acts_mean/rollout.final_acts_mean.max(), 'orange'),
                            # get_scatter('acts std', rollout.final_acts_std/rollout.final_acts_std.max(), 'salmon')
                            ])
    update_timeline_fig_layout(fig_unc, "Uncertainty", height=80)

    # Stop and lead
    clfs = go.Figure(data=[get_scatter('has stop p', sigmoid_python(rollout.aux_targets_p[:, "has_stop"]), 'red'),
        get_scatter('has lead p', sigmoid_python(rollout.aux_targets_p[:, "has_lead"]), 'blue')])
    update_timeline_fig_layout(clfs, "Stops + Leads", height=80)

    # Speed
    speed = go.Figure(data=[get_scatter('speed', rollout.aux[:,"speed"], 'blue'), 
                            get_scatter('ccs', rollout.additional_results[:,"ccs_p"], 'orange')])
    update_timeline_fig_layout(speed, "Speed", xaxis_visible=True, height=80)

    # Dagger shift
    dagger_shift = go.Figure(data=[get_scatter('dagger_shift', rollout.aux_targets_p[:,"dagger_shift"], 'blue')])
    update_timeline_fig_layout(dagger_shift, "dagger_shift", xaxis_visible=True, height=80)
    
    fig_lane_width = go.Figure(data=[get_scatter('lane_width', rollout.aux_targets_p[:,"lane_width"], 'blue'), ])
    update_timeline_fig_layout(fig_lane_width, "lane_width", height=80)
refresh_charts()


# Image
image = np.zeros_like(rollout.img[0, :,:,:3])
image[:,:,:] = 200
image_display = image.copy()
fig_img = None
def update_fig_img():
    global fig_img, image, image_display
    fig_img = px.imshow(image_display)
    fig_img.update_layout(
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        xaxis_visible=False,
        yaxis_visible=False,
        dragmode="drawrect",
        margin=dict(l=0, r=0, t=0, b=0)
    )
update_fig_img()

from models import *
from viz_utils import *

m = EffNet().to(device) 
m.load_state_dict(torch.load(f"{BESPOKE_ROOT}/models/m{model_stem}.torch"), strict=False)
m.set_for_viz()
m.eval()

actgrad_source_options = list(range(8))
actgrad_source = 5
actgrad_target_options = ['traj', 'stop', 'lead', 'none']
actgrad_target = 'traj'

def get_actgrad_by_ix(ix):
    m.viz_ix = actgrad_source
    actgrad = get_actgrad(m, rollout.img[ix], rollout.aux[ix], actgrad_target=actgrad_target, viz_loc=actgrad_source)
    return actgrad

def rerun_m(m, img, aux):
    m.zero_grad()
    with torch.cuda.amp.autocast(): wps_p, aux_targets_p, obsnet_outs = m(prep_img(img[None,None, :,:,:]), prep_aux(aux[None,None,:])) 
    wps_p = unprep_wps(wps_p)
    aux_targets_p = unprep_aux_targets(aux_targets_p)
    obsnet_outs = unprep_obsnet(obsnet_outs)
    return wps_p[0,0], aux_targets_p[0,0], obsnet_outs[0,0]


import dash_bootstrap_components as dbc
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# blank_img = np.zeros_like(rollout.img[0, :,:,:3])
# blank_img[:,:,:] = 200
# im_url = np_image_to_base64(blank_img)

plots = [fig, fig_unc, fig_lane_width, clfs, speed, dagger_shift]

app.layout = html.Div(
    className="container",
    children=[
        dbc.Row(
            [
                dbc.Col([
                    # html.Img(id='img', src=im_url, style={"width": f"{IMG_WIDTH}px"}),
                    dcc.Graph(id="graph-image", figure=fig_img),
                    html.Pre(id="annotations-data"),
                ]),
                dbc.Col([
                    dcc.Dropdown(
                            run_ids,
                            value=run_ids[0],
                            id='_rollout1',
                            style={"width": "200px"}
                        ),
                    dcc.Dropdown(
                            run_ids,
                            value=run_ids[-1],
                            id='_rollout2',
                            style={"width": "200px"}
                        ),
                    dcc.Dropdown(
                            actgrad_target_options,
                            value=actgrad_target_options[0],
                            id='_actgrad_target',
                            style={"width": "200px"}
                    ),
                    dcc.Dropdown(
                            actgrad_source_options,
                            value=5,
                            id='_actgrad_source',
                            style={"width": "200px"}
                        ),
                    html.Button('clear frozen', id='clear-frozen', n_clicks=0),
                ], style={"position":"absolute", "top": "0px", "left": "0px", "width": "200px"}),
            ],
        ),
        dbc.Row(
            [
                dbc.Col([
                    dcc.Graph(id=f"graph-{i}", figure=f, clear_on_unhover=True) for i, f in enumerate(plots)
                ], style={"position":"absolute", "top": "380px", "left": "0px"}),
            ],
        )
    ])

num = 0

n_plots = len(plots)
img_frozen = False


from viz_utils import *
@app.callback(
    # Output("img", "src"),
    Output("graph-image", "figure"),
    [[Input(f"graph-{i}", "hoverData") for i in range(len(plots))], [Input(f"graph-{i}", "clickData") for i in range(len(plots))], Input("graph-image", "relayoutData")],
    prevent_initial_call=True,
)
def display_hover(hover_data, click_data, relayout_data):
    global image, image_display, fig_img, num, img_frozen
    triggered_id = ctx.triggered_id

    #print(img_frozen, "\n", hover_data, "\n", click_data, "\n", "\n", )
    if triggered_id=="graph-image":
        if "shapes" in relayout_data:
            d = relayout_data["shapes"]
            img_model = rollout.img[num].copy()
            for rect in d:  
                x0 = int(rect["x0"])
                x1 = int(rect["x1"])
                y0 = int(rect["y0"])
                y1 = int(rect["y1"])
                print(x0, x1, y0, y1)
                image_display[y0:y1, x1:x0, :] = 0
                img_model[y0:y1, x1:x0, :] = 0
                
            image = img_model[:,:,:3] #TODO combine these. We're doing twice through the model
            # Actgrad
            if actgrad_target=='none':
                image_display = image
            else:
                m.viz_ix = actgrad_source
                actgrad = get_actgrad(m, img_model, rollout.aux[num], actgrad_target=actgrad_target, viz_loc=actgrad_source)
                image_display = combine_img_actgrad(image, actgrad)

            wps_p, aux_targets_p, obsnet_outs = rerun_m(m, img_model, rollout.aux[num])
            image_display = enrich_img(img=image_display, wps_p=wps_p, aux_targets_p=aux_targets_p, 
                                aux=rollout.aux[num], obsnet_out=obsnet_outs)
            update_fig_img()
        else:
            print("nothing") #return no_update
        return fig_img

    elif not img_frozen:
        hoverclick_data = [h for h in (hover_data + click_data) if h is not None]
        if len(hoverclick_data)==0:
            return fig_img #np_image_to_base64(blank_img)
        else:
            hoverclick_data = hoverclick_data[0]["points"][0]

        bbox = hoverclick_data["bbox"]
        num = hoverclick_data["pointNumber"]
        
        image = rollout.img[num][:,:,:3]
        # Actgrad
        if actgrad_target=='none':
            image_display = image
        else:
            actgrad = get_actgrad_by_ix(num)
            image_display = combine_img_actgrad(image, actgrad)

        image_display = enrich_img(img=image_display, wps_p=rollout.wps_p[num], aux_targets_p=rollout.aux_targets_p[num], 
                                        aux=rollout.aux[num],obsnet_out=rollout.obsnet_outs[num])

        update_fig_img()
        #im_url = np_image_to_base64(img)

        if any(click_data):
            print("CLICK", click_data)
            img_frozen = True

        return fig_img
    else:
        return fig_img


outputs = [Output(f'graph-{i}', 'figure') for i in range(len(plots))]
@app.callback(
    *outputs,
    Input('_rollout1', 'value'),
    Input('_rollout2', 'value'),
    Input('_actgrad_source', 'value'),
    Input('_actgrad_target', 'value'),
    )
def update_graph(_rollout1, _rollout2, _actgrad_source, _actgrad_target):
    global run_id, run_id_2, actgrad_source, actgrad_target
    run_id, run_id_2 = _rollout1, _rollout2
    actgrad_source = _actgrad_source
    actgrad_target = _actgrad_target
    update_rollout()
    refresh_charts()

    return fig, fig_unc, fig_lane_width, clfs, speed, dagger_shift


@app.callback(
    Output('clear-frozen', 'value'),
    [Output(f"graph-{i}", "clickData") for i in range(len(plots))], # NOTE have to manually clear the clickData, otherwise it remains
    Input('clear-frozen', 'n_clicks'),
    prevent_initial_call=True,
)
def clear_frozen(n_clicks):
    print("clearing frozen")
    global img_frozen
    img_frozen = False
    return str(n_clicks), *[None]*len(plots)


if __name__ == '__main__':
    app.run_server(debug=True)
