from dash import dcc, html, Dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
from common.arguments import parse_args
args = parse_args()
vid = args.viz_subject

df = pd.read_csv('inference/'+vid[:-4]+'/'+ args.keypoints +  '_predictions/'+vid[:-4]+'_error2d_xy.csv')
df = df.round(1)

df2 = pd.read_csv('inference/'+vid[:-4]+'/'+ args.keypoints +  '_predictions/'+vid[:-4]+'_error2d.csv')
df2 = df2.round(1)

mean_error_kps_xy = df.mean(axis=0)
mean_error_kps = df2.mean(axis=0)
mean_error_frame = df2.mean(axis=1)
total_mean_error = (mean_error_frame).mean()
print(total_mean_error)
error_x = 0
error_y = 0

for i in range(0,len(mean_error_kps_xy),3):
    error_x += mean_error_kps_xy[i]
    error_y += mean_error_kps_xy[i + 1]

mean_error_x = error_x/len(mean_error_kps)
mean_error_y = error_y/len(mean_error_kps)

# Initialize the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = args.keypoints.upper()
app.server.static_folder = 'static'

custom_css = {
    'line-height': '2.5',
    # 'margin-right': '3px',
    # 'margin-left': '10px'
}


# Define the color map
color_map = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
             "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#C5B0D5", "#D2691E",
             "#A9A9A9", "#FFC0CB", "#FFEBCD", "#F0E68C", "#98FB98", "#ADD8E6"]

# Define the layout of the app

fig1 = px.bar(mean_error_kps, y=mean_error_kps.round(0).values,text_auto=True, color=mean_error_kps.index)
fig1.update_layout(xaxis_title="")
fig1.update_layout(yaxis_title="Mean error in pixels")
fig1.update_layout(title={ 'text': "Mean error per joint", 'x':0.5, 'y':0.95  })

mean_error_kps_xy
fig2 = px.bar(mean_error_kps_xy, x=mean_error_kps_xy.index, y=mean_error_kps_xy.round(0).values,text_auto=True ,
              color=mean_error_kps_xy.index, color_discrete_sequence=color_map)
fig2.update_layout(xaxis_title="")
fig2.update_layout(yaxis_title="Mean error in pixels")
fig2.update_layout(title={ 'text': "Mean error per component of joint", 'x':0.5, 'y':0.95  })


fig3 = px.line(mean_error_frame, x=mean_error_frame.index, y=mean_error_frame.round(0).values)
fig3.update_layout(xaxis_title="Frame")
fig3.update_layout(yaxis_title="Mean error in pixels")
fig3.update_layout(title={ 'text': "Mean error per frame", 'x':0.5, 'y':0.95  })

# fig4 = px.line(df2, x=df2.index, y=df2.columns[0:2])
fig4 = go.Figure()
traces = []
for col in df2.columns:
    traces.append(go.Scatter(x=df2.index, y=df2[col], mode='lines', name=col))
fig4.add_traces(traces)
fig4.update_layout(xaxis_title="Frame")
fig4.update_layout(yaxis_title="Error in pixels")
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
fig5.update_layout(yaxis_title="Error in pixels")
fig5.update_layout(title={ 'text': "Component error per frame", 'x':0.5, 'y':0.95  })
fig5.update_layout(hovermode="x unified")


mpjpe_str = args.keypoints.upper() + ': ' + str(int(total_mean_error))+'pixels ( x:'+ str(int(mean_error_x))+ ' | y:',str(int(mean_error_y))+ ' )'

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
                id='dict_keypoints_error',
                figure=fig4
            ),width=6)
        ,
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
                id='mean_kps_xy',
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

    ]),

])




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,host='localhost', port=8040)

