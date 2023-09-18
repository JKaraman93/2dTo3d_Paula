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
from plotly.subplots import make_subplots
vid = args.viz_subject



detections = ['detectron', 'openpose', 'mediapipe',] 
fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()
fig5 = go.Figure()

# Define the color map
color_map = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
             "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#C5B0D5", "#D2691E",
             "#A9A9A9", "#FFC0CB", "#FFEBCD", "#F0E68C", "#98FB98", "#ADD8E6"]





line_color = 'black'
line_width = 1
dimension = ['X','Y']
mean_error_text = ''
allframes_allpreds = {}
for ind, d in enumerate (detections) :
    df1 = pd.read_csv('inference/'+vid[:-4]+'/' + d + '_predictions/'+vid[:-4]+'_error2d_xy.csv')
    df2 = pd.read_csv('inference/'+vid[:-4]+'/' + d + '_predictions/'+vid[:-4]+'_error2d.csv')
    checklist = df2.columns[[i for i in range(6)]]
    df1 = df1.round(1)
    df2 = df2.round(1)
    mean_error_kps_xy = df1.mean(axis=0)
    mean_error_kps = df2.mean(axis=0)
    mean_error_frame = df2.mean(axis=1)
    total_mean_error = mean_error_frame.mean()
    #print (mean_error_text)
    error_x = 0
    error_y = 0

    for i in range(0, len(mean_error_kps_xy), 3):
        error_x += mean_error_kps_xy[i]
        error_y += mean_error_kps_xy[i + 1]

    mean_error_x = error_x / len(mean_error_kps)
    mean_error_y = error_y / len(mean_error_kps)
    mean_error_text += d +' : ' + str(round(total_mean_error))+' mm ( x:' + str(int(mean_error_x)) + ' | y:'+str(int(mean_error_y))+ ' )'+'  \n'
    print(total_mean_error)
    print ('dim-x : ',mean_error_x )
    print('dim-y : ', mean_error_y)

    fig1.add_trace(go.Bar(x=mean_error_kps_xy.index, y=mean_error_kps_xy.round(0).values, text=mean_error_kps_xy.round(0).values, textposition='outside',
                 hovertemplate='%{text}',  name=d,  marker=dict(color=color_map[ind+1],  line=dict(color=line_color, width=line_width))))
    fig2.add_trace(go.Bar(x=mean_error_kps.index, y=mean_error_kps.round(0).values, text=mean_error_kps.round(0).values, textposition='outside',
                 hovertemplate='%{text}',  name=d,  marker=dict(color=color_map[ind+1],  line=dict(color=line_color, width=line_width))))
    fig3.add_trace(go.Scatter(x=mean_error_frame.index, y=mean_error_frame.round(0).values, name=d, mode='markers+lines', marker=dict(color=color_map[ind+1])))

    traces = []
    for col in df2.columns:
        traces.append(go.Scatter(x=df2.index, y=df2[col], mode='markers+lines', name=col+'_'+d ,marker=dict(color=color_map[ind+1])))
    fig4.add_traces(traces)

    traces = []
    for col in df1.columns:
        traces.append(go.Scatter(x=df1.index, y=df1[col], mode='markers+lines', name=col+'_'+d ,marker=dict(color=color_map[ind+1])))
    fig5.add_traces(traces)



fig1.update_layout(barmode='group', xaxis_tickangle=-60)
fig1.update_layout(hovermode='x')
fig1.update_layout(yaxis_title="Mean error in pixels")
fig1.update_layout(title={ 'text': "Mean error per component of joint", 'x':0.5, 'y':0.95  })
fig1.update_layout(height=600)

fig2.update_layout(barmode='group', xaxis_tickangle=-60)
fig2.update_layout(hovermode='x')
fig2.update_layout(height=600)
fig2.update_layout(yaxis_title="Mean error in pixels")
fig2.update_layout(title={ 'text': "Mean error per joint", 'x':0.5, 'y':0.95  })

fig3.update_layout(hovermode="x unified")
fig3.update_layout(xaxis_title="Frame")
fig3.update_layout(yaxis_title="Mean error in pixels")
fig3.update_layout(title={ 'text': "Mean error per frame", 'x':0.5, 'y':0.95  })

# Initialize the Dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Comparison'

custom_css = {
    'line-height': '3.0',
    # 'margin-right': '3px',
    # 'margin-left': '10px'
}





app.layout = html.Div([
dbc.Row(
        dbc.Col(dcc.Markdown(mean_error_text,style={'color': 'red', 'fontSize': 18, 'text-align':'center'}))),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='line-chart3',
                figure=fig3,
            ), width=6),
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
        html.H6('Select a joint:',style={'font-weight': 'bold'}),
            dcc.RadioItems(
                id='line-selector',
                options=[{'label': col, 'value': col} for col in checklist],
                value='leftwrist',
                # labelStyle={'display': 'inline-block'},
                #style={'margin-top': '11px'},
                #inputStyle=custom_css,
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
])

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
    fig.update_layout(yaxis_title="Error in pixels")
    fig.update_layout(title={'text': "Joint error per frame", 'x': 0.5, 'y': 0.95})
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig

@app.callback(
    Output('line-chart5', 'figure'),
    [Input('line-selector', 'value'),Input('line-selector3', 'value')])
def update_chart(selected_lines1,selected_lines2):
    fig = go.Figure()
    for t in fig5.data:
        if t.name.startswith(selected_lines1+selected_lines2):
        #if t.name.startswith(selected_lines1):
            fig.add_traces(t)
    fig.update_layout(xaxis_title="Frame")
    fig.update_layout(yaxis_title="Error in pixels")
    fig.update_layout(title={'text': "Component error per frame", 'x': 0.5, 'y': 0.95})
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig

@app.callback(
    Output('line-chart6', 'figure'),
    [Input('line-selector', 'value')])
def update_chart(selected_lines):
    fig = make_subplots(rows=2, cols=1,
    subplot_titles=("X-dimension", "Y-dimension"),
    shared_xaxes = True,
    vertical_spacing = 0.1, x_title='Frame', y_title='Error in pixels')
    for t in fig5.data:
        if t.name.startswith(selected_lines+'X'):
            fig.add_trace(t,row=1, col=1)
        elif t.name.startswith(selected_lines+'Y'):
            fig.add_trace(t,row=2, col=1)

    #fig.update_layout(xaxis_title="Frame")
    #fig.update_layout(yaxis_title="Error in pixels")
    fig.update_layout(title={'text': "Component error per frame", 'x': 0.5, 'y': 0.95})
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified",showlegend=False)
    return fig



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,host='localhost', port=8050)
