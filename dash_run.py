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

run_ids = ["run_626"] #["run_621", "run_622"] #["run_596"] #["run_555b", "run_556d", "run_555a", "run_556a", "run_556b", "run_556c", "run_567", "none"]
run_id = run_ids[0]
model_stem = "4.28_e118"
model_stem_b = None #"4.17_e27" #"4.13_e33" #"4.5_e68"

MIN_PT = 0
L = 4000
MAX_PT = MIN_PT+L

rollout, rollout_b, img_paths = None, None, None
def update_rollout():
    global rollout, rollout_b, img_paths
    rollout = load_object(f"{BESPOKE_ROOT}/tmp/{run_id}_{model_stem}_rollout.pkl")
    if model_stem_b is not None:
        rollout_b = load_object(f"{BESPOKE_ROOT}/tmp/{run_id}_{model_stem_b}_rollout.pkl")
    
    img_paths = sorted(glob.glob(f"{SSD_ROOT}/bespoke_logging/{run_id}/img/*"))
    print(len(img_paths))

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
        margin=dict(l=20, r=20, t=0, b=0), 
        showlegend=False,
    )
    # fig.update_layout(legend=dict(
    #     yanchor="top",
    #     y=0.99,
    #     xanchor="left",
    #     x=0.01
    # ))
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

fig, fig_unc, fig_lane_width, stop, stop_dist, lead, speed, dagger_shift, rd_is_lined = None, None, None, None, None, None, None, None, None
def refresh_charts():
    global fig, fig_unc, fig_lane_width, stop, stop_dist, lead, speed, dagger_shift, rd_is_lined

    # Tire angle
    tire_angle_data = [get_scatter('tire_angle', rollout.aux[MIN_PT:MAX_PT, "tire_angle"]*-1, 'blue')]
    if model_stem_b is not None:
        tire_angle_data += [get_scatter('tire_angle_p2', rollout_b.additional_results[MIN_PT:MAX_PT, "tire_angle_p"]*-1, 'brown')]
    tire_angle_data += [get_scatter('tire_angle_p', rollout.additional_results[MIN_PT:MAX_PT, "tire_angle_p"]*-1, 'red')]
    fig = go.Figure(data=tire_angle_data)
    update_timeline_fig_layout(fig, "Tire Angle")

    # Speed
    speed_data = [get_scatter('speed', rollout.aux[MIN_PT:MAX_PT,"speed"], 'blue')]
    if model_stem_b is not None:
        speed_data += [get_scatter('ccs', rollout_b.additional_results[MIN_PT:MAX_PT,"ccs_p"], 'brown')]
    speed_data += [get_scatter('ccs', rollout.additional_results[MIN_PT:MAX_PT,"ccs_p"], 'red')]
    speed = go.Figure(data=speed_data)
    update_timeline_fig_layout(speed, "Speed", xaxis_visible=True, height=80)

    # Uncertainty
    c = lambda x : np.clip(x, 0, .002)
    unc_data = [get_scatter('uncertainty p2', c(rollout_b.additional_results[MIN_PT:MAX_PT, "te"]), 'brown')] if model_stem_b is not None else []
    unc_data += [get_scatter('uncertainty p', c(rollout.additional_results[MIN_PT:MAX_PT, "te"]), 'red')]
    fig_unc = go.Figure(data=unc_data)
    update_timeline_fig_layout(fig_unc, "te", height=50)

    # Stop
    c = lambda x : np.clip(x, -5, 5)
    _has_stop = c(rollout.aux_targets_p[MIN_PT:MAX_PT, "has_stop"])
    stop_data = [get_scatter('has stop p2', c(rollout_b.aux_targets_p[MIN_PT:MAX_PT, "has_stop"]), 'brown')] if model_stem_b is not None else []
    stop_data += [get_scatter('has stop p', _has_stop, 'red')]
    stop = go.Figure(data=stop_data)
    update_timeline_fig_layout(stop, "Stop", height=50)

    # Stop dist
    _stop_dist = rollout.aux_targets_p[MIN_PT:MAX_PT, "stop_dist"].copy() # copy otherwise alters underlying rollout
    _stop_dist[_has_stop < 0] = 70 # manually peg to max when no stop is indicated. For visual cleanliness
    stop_dist_data = [get_scatter('stop dist p2', rollout_b.aux_targets_p[MIN_PT:MAX_PT, "stop_dist"], 'brown')] if model_stem_b is not None else []
    stop_dist_data += [get_scatter('stop dist', _stop_dist, 'red')]
    stop_dist = go.Figure(data=stop_dist_data)
    update_timeline_fig_layout(stop_dist, "Stop dist", height=50)

    # Lead
    lead_data = [get_scatter('has lead p2', c(rollout_b.aux_targets_p[MIN_PT:MAX_PT, "has_lead"]), 'brown')] if model_stem_b is not None else []
    lead_data += [get_scatter('has lead p', c(rollout.aux_targets_p[MIN_PT:MAX_PT, "has_lead"]), 'red')]
    lead = go.Figure(data=lead_data)
    update_timeline_fig_layout(lead, "Lead", height=50)

    # Dagger shift
    dagger_data = [get_scatter('dagger shift p2', rollout_b.aux_targets_p[MIN_PT:MAX_PT, "dagger_shift"], 'brown')] if model_stem_b is not None else []
    dagger_data += [get_scatter('dagger shift p', rollout.aux_targets_p[MIN_PT:MAX_PT, "dagger_shift"], 'red')]
    dagger_shift = go.Figure(data=dagger_data)
    update_timeline_fig_layout(dagger_shift, "dagger", xaxis_visible=True, height=80)
    
    # is-lined
    rd_is_lined_data = [get_scatter('is-lined p2', c(rollout_b.aux_targets_p[MIN_PT:MAX_PT, "rd_is_lined"]), 'brown')] if model_stem_b is not None else []
    rd_is_lined_data += [get_scatter('is-lined p', c(rollout.aux_targets_p[MIN_PT:MAX_PT, "rd_is_lined"]), 'red')]
    rd_is_lined = go.Figure(data=rd_is_lined_data)
    update_timeline_fig_layout(rd_is_lined, "is-lined", xaxis_visible=True, height=80)

    # lane width
    lane_width_data = [get_scatter('lane_width', rollout_b.aux_targets_p[MIN_PT:MAX_PT,"lane_width"], 'brown')] if model_stem_b is not None else []
    lane_width_data += [get_scatter('lane_width', rollout.aux_targets_p[MIN_PT:MAX_PT,"lane_width"], 'red')]
    fig_lane_width = go.Figure(data=lane_width_data)
    update_timeline_fig_layout(fig_lane_width, "L width", height=50)
refresh_charts()


# Image
image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
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
m.use_rnn = False # will give us different results on traj compared w rollout versions

actgrad_source_options = list(range(8))
actgrad_source = 5
actgrad_target_options = ['none', 'traj', 'stop', 'lead']
actgrad_target = 'traj'

def get_actgrad_by_ix(ix):
    m.viz_ix = actgrad_source
    _img = np.load(img_paths[ix])
    actgrad = get_actgrad(m, _img, rollout.aux[ix], actgrad_target=actgrad_target, viz_loc=actgrad_source)
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

plots = (fig, speed, fig_unc, stop, stop_dist, lead, dagger_shift, rd_is_lined, fig_lane_width)

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
                            id='_run_id',
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
                    dcc.Dropdown(
                            [0, 4000, 8000, 12000, 16000, 20000],
                            value=0,
                            id='_start_ix',
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
            img_model = np.load(img_paths[num]).copy()
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
        num = hoverclick_data["pointNumber"] + MIN_PT
        image = np.load(img_paths[num])[:,:,:3]
        # Actgrad
        if actgrad_target=='none':
            image_display = image
        else:
            actgrad = get_actgrad_by_ix(num)
            image_display = combine_img_actgrad(image, actgrad)

        image_display = enrich_img(img=image_display, 
                                   wps_p=rollout.wps_p[num], wps_p2=rollout_b.wps_p[num] if rollout_b is not None else None,
                                   aux_targets_p=rollout.aux_targets_p[num], aux_targets_p2=rollout_b.aux_targets_p[num] if rollout_b is not None else None,
                                   aux=rollout.aux[num],
                                   obsnet_out=rollout.obsnet_outs[num])

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
    Input('_run_id', 'value'),
    Input('_actgrad_source', 'value'),
    Input('_actgrad_target', 'value'),
    Input('_start_ix', 'value'),
    )
def update_graph(_run_id, _actgrad_source, _actgrad_target, _start_ix):
    global run_id, actgrad_source, actgrad_target, MIN_PT, MAX_PT
    triggered_id = ctx.triggered_id
    run_id = _run_id
    actgrad_source = _actgrad_source
    actgrad_target = _actgrad_target
    MIN_PT = _start_ix
    MAX_PT = MIN_PT + L
    print(triggered_id)
    if triggered_id == "_run_id":
        MIN_PT = 0
        MAX_PT = MIN_PT + L 
        update_rollout()
    refresh_charts()

    return fig, speed, fig_unc, stop, stop_dist, lead, dagger_shift, rd_is_lined, fig_lane_width


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
