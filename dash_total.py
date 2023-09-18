from dash import dcc, html, Dash
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
import dash_player

from common.arguments import parse_args
from plotly.subplots import make_subplots

### configuration ###
# --viz-subject EUDs4Front.mp4

args = parse_args()
vid = args.viz_subject
ground_truth_world = np.float32(np.load('inference/' + vid[:-4] + '/' + vid[:-4] + '_world.npy')) / 100  # cm to meters

detections = ['openpose', 'mp3d', 'detectron', 'mpcoco', ]  # mp3d 'detectron','mediapipe'
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()
fig5 = go.Figure()
dfs = {}
# Define the color map
color_map = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
             "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#C5B0D5", "#D2691E",
             "#A9A9A9", "#FFC0CB", "#FFEBCD", "#F0E68C", "#98FB98", "#ADD8E6"]


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, }  # "easing": "linear"},
    }


# Create figure
parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
data_allframes_gt = []
for frame in range(ground_truth_world.shape[0]):
    legend = False
    data_frame = []
    x_data = ground_truth_world[frame, :, 0]
    y_data = ground_truth_world[frame, :, 1]
    z_data = ground_truth_world[frame, :, 2]
    for j, j_parent in enumerate(parents):
        if j == len(parents) - 1:
            legend = True
        if j_parent == -1:
            continue
        data_frame.append(go.Scatter3d(
            x=np.array([x_data[j], x_data[j_parent]]),
            y=np.array([y_data[j], y_data[j_parent]]),
            z=np.array([z_data[j], z_data[j_parent]]),
            mode='lines+markers',
            line=dict(width=1, color='green'),
            marker=dict(size=3, color='blue'),
            name='GT',
            showlegend=legend
        ))
    data_allframes_gt.append(data_frame)

parents = np.array([-1, 0, 0, 0, 3, 4, 5, 4, 7, 8, 4, 10, 11])

line_color = 'black'
line_width = 1
dimension = ['X', 'Y', 'Z']
mean_error_text = ''
allframes_allpreds = {}
for ind, d in enumerate(detections):
    predictions = np.float32(
        np.load('inference/' + vid[:-4] + '/' + d + '_predictions/' + vid[:-4] + '_pred3d_world.npy'))
    df1 = pd.read_csv('inference/' + vid[:-4] + '/' + d + '_predictions/' + vid[:-4] + '_error3d_xyz.csv')
    df2 = pd.read_csv('inference/' + vid[:-4] + '/' + d + '_predictions/' + vid[:-4] + '_error3d.csv')
    checklist = df2.columns[[i for i in range(6)]]
    df1 = df1.round(1)
    df2 = df2.round(1)
    dfs[d] = df1
    mean_error_kps_xyz = df1.mean(axis=0)
    mean_error_kps = df2.mean(axis=0)
    mean_error_frame = df2.mean(axis=1)
    total_mean_error = mean_error_frame.mean()
    # print (mean_error_text)
    error_x = 0
    error_y = 0
    error_z = 0

    for i in range(0, len(mean_error_kps_xyz), 3):
        error_x += mean_error_kps_xyz[i]
        error_y += mean_error_kps_xyz[i + 1]
        error_z += mean_error_kps_xyz[i + 2]

    mean_error_x = error_x / len(mean_error_kps)
    mean_error_y = error_y / len(mean_error_kps)
    mean_error_z = error_z / len(mean_error_kps)
    mean_error_text += d + ' : ' + str(round(total_mean_error)) + ' mm ( x:' + str(int(mean_error_x)) + ' | y:' + str(
        int(mean_error_y)) + ' | z:' + str(int(mean_error_z)) + ' )' + '  \n'

    fig1.add_trace(go.Bar(x=mean_error_kps_xyz.index, y=mean_error_kps_xyz.round(0).values,
                          text=mean_error_kps_xyz.round(0).values, textposition='outside',
                          hovertemplate='%{text}', name=d,
                          marker=dict(color=color_map[ind + 1], line=dict(color=line_color, width=line_width))))
    fig2.add_trace(go.Bar(x=mean_error_kps.index, y=mean_error_kps.round(0).values, text=mean_error_kps.round(0).values,
                          textposition='outside',
                          hovertemplate='%{text}', name=d,
                          marker=dict(color=color_map[ind + 1], line=dict(color=line_color, width=line_width))))
    fig3.add_trace(
        go.Scatter(x=mean_error_frame.index, y=mean_error_frame.round(0).values, name=d, mode='markers+lines',
                   marker=dict(color=color_map[ind + 1])))

    traces = []
    for col in df2.columns:
        traces.append(go.Scatter(x=df2.index, y=df2[col], mode='markers+lines', name=col + '_' + d,
                                 marker=dict(color=color_map[ind + 1])))
    fig4.add_traces(traces)

    traces = []
    for col in df1.columns:
        traces.append(go.Scatter(x=df1.index, y=df1[col], mode='markers+lines', name=col + '_' + d,
                                 marker=dict(color=color_map[ind + 1])))
    fig5.add_traces(traces)

    # if d != 'mp3d':
    predictions = np.delete(predictions, [6, 5, 3, 2], axis=1)  # delete foot 3d joints
    data_allframes_pred = []
    for frame in range(predictions.shape[0]):
        legend = False
        data_frame = []
        x_data = predictions[frame, :, 0]
        y_data = predictions[frame, :, 1]
        z_data = predictions[frame, :, 2]
        for j, j_parent in enumerate(parents):
            if j == len(parents) - 1:
                legend = True
            if j_parent == -1:
                continue
            data_frame.append(go.Scatter3d(
                x=np.array([x_data[j], x_data[j_parent]]),
                y=np.array([y_data[j], y_data[j_parent]]),
                z=np.array([z_data[j], z_data[j_parent]]),
                mode='lines+markers',
                line=dict(width=1, color=color_map[ind + 1]),
                marker=dict(size=3, color=color_map[ind + 1]),
                name=d.upper(),
                showlegend=legend
            ))
        data_allframes_pred.append(data_frame)
    allframes_allpreds[d] = data_allframes_pred

fig1.update_layout(barmode='group', xaxis_tickangle=-60)
fig1.update_layout(hovermode='x')
fig1.update_layout(yaxis_title="Mean error in mm")
fig1.update_layout(title={'text': "Mean error per component of joint", 'x': 0.5, 'y': 0.95})
fig1.update_layout(height=600)

fig2.update_layout(barmode='group', xaxis_tickangle=-60)
fig2.update_layout(hovermode='x')
fig2.update_layout(height=600)
fig2.update_layout(yaxis_title="Mean error in mm")
fig2.update_layout(title={'text': "Mean error per joint", 'x': 0.5, 'y': 0.95})

fig3.update_layout(hovermode="x unified")
fig3.update_layout(xaxis_title="Frame")
fig3.update_layout(yaxis_title="Mean error in mm")
fig3.update_layout(title={'text': "Mean error per frame", 'x': 0.5, 'y': 0.95})

# Initialize the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = vid + ' - Comparison'
app.server.static_folder = 'static'

custom_css = {
    'line-height': '3.0',
    # 'margin-right': '3px',
    # 'margin-left': '10px'
}

video_url = 'static/'+vid

app.layout = html.Div([
    dbc.Row([
        dbc.Col(dcc.Markdown(mean_error_text, style={'color': 'red', 'fontSize': 18, 'text-align': 'center'}), width=4),
        dbc.Col(dash_player.DashPlayer(
            id="player",
            url=video_url,
            controls=True,
            width="100%",
            height="500px",
        )),

    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='line-chart3',
                figure=fig3,
            ), width=6),
        dbc.Col(
            dcc.Graph(
                id='3d_plot',
                figure={}
            ), width=5, ),
        dbc.Col(
            dcc.Checklist(
                id='line-selector1',
                options=[{'label': col, 'value': col} for col in detections],
                value=['openpose'],
                # labelStyle={'display': 'inline-block'},
                # style={'margin-top': '11px'},
                # inputStyle=custom_css,
                # labelClassName='my-custom-label'
            ), width=1, align="center"
        ),
    ]),
    dbc.Row([

        dbc.Col(
            dcc.Graph(
                id='line-chart2',
                figure=fig2,
            ), width=5),
        dbc.Col(
            dcc.Graph(
                id='line-chart1',
                figure=fig1,
            ), width=7)
    ]),

    dbc.Row([
        dbc.Col(
            html.Div([
                html.H6('Select a joint:', style={'font-weight': 'bold'}),
                dcc.RadioItems(
                    id='line-selector',
                    options=[{'label': col, 'value': col} for col in checklist],
                    value='leftwrist',
                    # labelStyle={'display': 'inline-block'},
                    # style={'margin-top': '11px'},
                    # inputStyle=custom_css,
                    # labelClassName='my-custom-label'
                )]), width=1, align="center"
        ),
        dbc.Col(
            dcc.Graph(
                id='line-chart4',
                figure=fig4,
            ), width=6),
        dbc.Col(
            dcc.Graph(
                id='line-chart6',
                figure={},
            ), width=5),
    ]),
    dbc.Row([
        dbc.Col(

            dcc.RadioItems(
                id='line-selector3',
                options=[{'label': col, 'value': col} for col in dimension],
                value='X',
                # labelStyle={'display': 'inline-block'},
                # style={'margin-top': '11px'},
                # inputStyle=custom_css,
                # labelClassName='my-custom-label'
            ), width=1, align="center"
        ),
        dbc.Col(
            dcc.Graph(
                id='line-chart5',
                figure=fig5,
            ), width=9),
    ]),
    dbc.Row([
        dbc.Col([
            html.H6("Change the value in the text box!"),
            html.Div([
                "Frame: ",
                dcc.Input(id='my-input', value=0, type='number', min=0, max=len(dfs['openpose'].index)-1, step=1)])
                          #max=len(dfs['openpose'].index),
            ,
            dcc.Graph(
                id='box-plot6',
                figure={},
            )])]),
])


# Define the callback function that updates the chart based on the selected lines


@app.callback(
    Output('box-plot6', 'figure'),
    [Input('my-input', 'value')])
def update_chart(frame):
    fig = go.Figure()
    # fig = px.bar(df.iloc[frame,:], y=df.iloc[frame,:].round(0).values,text_auto=True, color=df.iloc[0,:].index)
    for ind, dfk in enumerate(dfs.keys()):
        df = dfs[dfk]
        fig.add_trace(go.Bar(x=df.iloc[frame, :].index, y=df.iloc[frame, :].round(0).values,
                             text=df.iloc[frame, :].round(0).values, textposition='outside',
                             hovertemplate='%{text}', name=dfk,
                             marker=dict(color=color_map[ind + 1], line=dict(color=line_color, width=line_width))))

    fig.update_layout(xaxis_title="")
    fig.update_layout(yaxis_title="error in mm")
    fig.update_layout(title={'text': " Error per component of joint on frame : " + str(frame), 'x': 0.5, 'y': 0.95})
    return fig


# Define the callback function that updates the chart based on the selected lines
@app.callback(
    Output('line-chart4', 'figure'),
    [Input('line-selector', 'value')])
def update_chart(selected_lines):
    fig = go.Figure()
    for t in fig4.data:
        if t.name.startswith(selected_lines):
            fig.add_traces(t)
    fig.update_layout(xaxis_title="Frame")
    fig.update_layout(yaxis_title="Error in mm")
    fig.update_layout(title={'text': "Joint error per frame", 'x': 0.5, 'y': 0.95})
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig


@app.callback(
    Output('line-chart5', 'figure'),
    [Input('line-selector', 'value'), Input('line-selector3', 'value')])
def update_chart(selected_lines1, selected_lines2):
    fig = go.Figure()
    for t in fig5.data:
        if t.name.startswith(selected_lines1 + selected_lines2):
            # if t.name.startswith(selected_lines1):
            fig.add_traces(t)
    fig.update_layout(xaxis_title="Frame")
    fig.update_layout(yaxis_title="Error in mm")
    fig.update_layout(title={'text': "Component error per frame", 'x': 0.5, 'y': 0.95})
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig


@app.callback(
    Output('line-chart6', 'figure'),
    [Input('line-selector', 'value')])
def update_chart(selected_lines):
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("X-dimension", "Y-dimension", "Z-dimension"),
                        shared_xaxes=True,
                        vertical_spacing=0.1, x_title='Frame', y_title='Error in mm')
    for t in fig5.data:
        if t.name.startswith(selected_lines + 'X'):
            fig.add_trace(t, row=1, col=1)
        elif t.name.startswith(selected_lines + 'Y'):
            fig.add_trace(t, row=2, col=1)
        elif t.name.startswith(selected_lines + 'Z'):
            fig.add_trace(t, row=3, col=1)

    # fig.update_layout(xaxis_title="Frame")
    # fig.update_layout(yaxis_title="Error in mm")
    fig.update_layout(title={'text': "Component error per frame", 'x': 0.5, 'y': 0.95})
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified", showlegend=False)
    return fig


@app.callback(
    Output('3d_plot', 'figure'),
    [Input('line-selector1', 'value')])
def update_chart(selected_lines):
    input_lists = [data_allframes_gt] + [allframes_allpreds[detection] for detection in selected_lines]
    data_allframes = []
    rotation_angle = 2
    all_cameras = []
    for f in range(predictions.shape[0]):
        frame_data = []
        azimuth = f * rotation_angle

        # Calculate the new camera view based on the azimuth angle
        camera = dict(
            eye=dict(
                x=np.cos(np.deg2rad(azimuth)) * 2,
                y=np.sin(np.deg2rad(azimuth)) * 2,
                z=1.2
            )
        )
        all_cameras.append(camera)
        for in_list in input_lists:
            frame_data.extend(in_list[f])
        data_allframes.append(frame_data)

    fig = go.Figure(data=data_allframes[0], layout=go.Layout(
        autosize=False,
        width=750,
        height=750, ))

    frames = [go.Frame(data=data_allframes[k], name=f'frame{k}', layout=dict(scene_camera=all_cameras[k])) for k in
              range(predictions.shape[0])]

    fig.frames = frames

    fig.update_layout(
        scene_camera=dict(eye=dict(x=np.cos(np.deg2rad(azimuth)) * 2,
                                   y=np.sin(np.deg2rad(azimuth)) * 2,
                                   z=0)),
        updatemenus=[{"buttons": [
            {
                "args": [None, frame_args(0)],
                "label": "&#9654;",  # play symbol
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "&#9724;",  # pause symbol
                "method": "animate",
            }],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }],
        sliders=[{

            "currentvalue": {"prefix": "Frame: "},
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }]
    )
    fig.update_layout(scene_aspectmode='auto', uirevision=True)
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='localhost', port=8050)
