from dash import dcc, html, Dash
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
from common.arguments import parse_args
args = parse_args()
vid = args.viz_subject

df = pd.read_csv('inference/'+vid[:-4]+'/'+ args.keypoints +  '_predictions/'+vid[:-4]+'_error3d_xyz.csv')
df = df.round(1)

df2 = pd.read_csv('inference/'+vid[:-4]+'/'+ args.keypoints +  '_predictions/'+vid[:-4]+'_error3d.csv')
df2 = df2.round(1)

mean_error_kps_xyz = df.mean(axis=0)
mean_error_kps = df2.mean(axis=0)
mean_error_frame = df2.mean(axis=1)
total_mean_error = (mean_error_frame).mean()
print(total_mean_error)
error_x = 0
error_y = 0
error_z = 0

for i in range(0,len(mean_error_kps_xyz),3):
    error_x += mean_error_kps_xyz[i]
    error_y += mean_error_kps_xyz[i + 1]
    error_z += mean_error_kps_xyz[i + 2]

mean_error_x = error_x/len(mean_error_kps)
mean_error_y = error_y/len(mean_error_kps)
mean_error_z = error_z/len(mean_error_kps)
print(mean_error_x, mean_error_y, mean_error_z)

# Initialize the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = args.keypoints.upper()
app.server.static_folder = 'static'

custom_css = {
    'line-height': '2.5',
    # 'margin-right': '3px',
    # 'margin-left': '10px'
}
predictions = np.float32(np.load('inference/'+vid[:-4]+'/'+ args.keypoints+ '_predictions/'+vid[:-4]+'_pred3d_world.npy'))
ground_truth_world = np.float32(np.load('inference/'+vid[:-4]+'/'+vid[:-4]+'_world.npy'))/100 #cm to meters


# Define the color map
color_map = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
             "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#C5B0D5", "#D2691E",
             "#A9A9A9", "#FFC0CB", "#FFEBCD", "#F0E68C", "#98FB98", "#ADD8E6"]

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration,}# "easing": "linear"},
        }

# Create figure
parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
data_allframes_gt = []
###############
for frame in range(predictions.shape[0]):
    legend = False
    data_frame = []
    x_data = ground_truth_world[frame,:,0]
    y_data = ground_truth_world[frame,:,1]
    z_data = ground_truth_world[frame,:,2]
    for j, j_parent in enumerate(parents):
        if j==len(parents)-1:
            legend = True
        if j_parent == -1:
            continue
        data_frame.append(go.Scatter3d(
            x=np.array([x_data[j], x_data[j_parent]]),
            y=np.array([y_data[j], y_data[j_parent]]),
            z=np.array([z_data[j], z_data[j_parent]]),
            mode='lines+markers',
            line=dict(width=1, color='green'),
            marker = dict(size=3, color='blue'),
            name='GT',
            showlegend=legend
        ))
    data_allframes_gt.append(data_frame)

parents = np.array([-1, 0, 0, 0, 3, 4, 5, 4, 7, 8, 4, 10, 11])

## comment the following line for mp3d case  ##
predictions = np.delete(predictions, [6, 5, 3, 2], axis=1)  # delete foot 3d joints

############
data_allframes_pred = []
for frame in range(predictions.shape[0]):
    legend = False
    data_frame = []
    x_data = predictions[frame, :, 0]
    y_data = predictions[frame, :, 1]
    z_data = predictions[frame, :, 2]
    for j, j_parent in enumerate(parents):
        if j==len(parents)-1:
            legend = True
        if j_parent == -1:
            continue
        data_frame.append(go.Scatter3d(
            x=np.array([x_data[j], x_data[j_parent]]),
            y=np.array([y_data[j], y_data[j_parent]]),
            z=np.array([z_data[j], z_data[j_parent]]),
            mode='lines+markers',
            line=dict(width=1, color='red'),
            marker = dict(size=3, color='black'),
            name=args.keypoints.upper(),
            showlegend=legend
        ))
    data_allframes_pred.append(data_frame)
data_allframes = [l1 + l2 for l1, l2 in zip(data_allframes_gt, data_allframes_pred)]
fig3d = go.Figure(data=data_allframes[0],layout = go.Layout(
    autosize=False,
    width=750,
    height=750,))

frames = [go.Frame(data=data_allframes[k], name=f'frame{k}') for k in range(predictions.shape[0])]
fig3d.frames = frames

fig3d.update_layout(
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
                                "label":  str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig3d.frames)
                        ],
                    }]
                 )

fig3d.update_layout(scene_aspectmode='auto', )


# Define the layout of the app

fig1 = px.bar(mean_error_kps, y=mean_error_kps.round(0).values,text_auto=True, color=mean_error_kps.index)
fig1.update_layout(xaxis_title="")
fig1.update_layout(yaxis_title="Mean error in mm")
fig1.update_layout(title={ 'text': "Mean error per joint", 'x':0.5, 'y':0.95  })

mean_error_kps_xyz
fig2 = px.bar(mean_error_kps_xyz, x=mean_error_kps_xyz.index, y=mean_error_kps_xyz.round(0).values,text_auto=True ,
              color=mean_error_kps_xyz.index, color_discrete_sequence=color_map)
fig2.update_layout(xaxis_title="")
fig2.update_layout(yaxis_title="Mean error in mm")
fig2.update_layout(title={ 'text': "Mean error per component of joint", 'x':0.5, 'y':0.95  })


fig3 = px.line(mean_error_frame, x=mean_error_frame.index, y=mean_error_frame.round(0).values)
fig3.update_layout(xaxis_title="Frame")
fig3.update_layout(yaxis_title="Mean error in mm")
fig3.update_layout(title={ 'text': "Mean error per frame", 'x':0.5, 'y':0.95  })

# fig4 = px.line(df2, x=df2.index, y=df2.columns[0:2])
fig4 = go.Figure()
traces = []
for col in df2.columns:
    traces.append(go.Scatter(x=df2.index, y=df2[col], mode='lines', name=col))
fig4.add_traces(traces)
fig4.update_layout(xaxis_title="Frame")
fig4.update_layout(yaxis_title="Error in mm")
fig4.update_layout(title={ 'text': "Joint error per frame", 'x':0.5, 'y':0.95  })
fig4.update_traces(mode="markers+lines", hovertemplate=None)
fig4.update_layout(hovermode="x unified")

fig5 = go.Figure(layout = go.Layout(
    autosize=False,
    width=1100,
    height=600,))
traces = []
for color, col in enumerate  (df.columns):
    traces.append(go.Scatter(x=df.index, y=df[col], mode='markers+lines', name=col, line=dict(color=color_map[color])))
fig5.add_traces(traces)
fig5.update_layout(xaxis_title="Frame")
fig5.update_layout(yaxis_title="Error in mm")
fig5.update_layout(title={ 'text': "Component error per frame", 'x':0.5, 'y':0.95  })
fig5.update_layout(hovermode="x unified")


mpjpe_str = args.keypoints.upper() + ': ' + str(int(total_mean_error))+'mm (mpjpe) ( x:'+ str(int(mean_error_x))+ ' | y:',str(int(mean_error_y))+ ' | z:' + str(int(mean_error_z)) +' )'

app.layout = html.Div([
    dbc.Row(
        dbc.Col(html.Div(mpjpe_str, style={'color': 'red', 'fontSize': 24, 'text-align':'center'}))),
    dbc.Row(
        dbc.Col(
            html.Video(src='static/output_'+vid[:-4]+'_'+args.keypoints+'.mp4', autoPlay=False, loop=False, controls=True)) ,justify="center", align="center" #className="h-25"
    ),
    dbc.Row([
dbc.Col(
            dcc.Graph(
                id='3d_plot',
                figure=fig3d
            ), width=5,),
        dbc.Col(

            dcc.Graph(
                id='line-chart',
                figure=fig5,
            ), width=6),


    ], className="g-0",
    ),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='mean_kps',
                figure=fig1
            ), width=6
        ),
        dbc.Col(
            dcc.Graph(
                id='mean_kps_xyz',
                figure=fig2
            ), width=6
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='mean_error_frame',
                figure=fig3
            )
        ),
        dbc.Col(
            dcc.Graph(
                id='dict_keypoints_error',
                figure=fig4
            )
        )
    ]),

])




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,host='localhost', port=8042)
