import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input,Output,State
import dash_table
import dash_daq as daq

external_stylesheets=[dbc.themes.BOOTSTRAP]

app=dash.Dash(__name__, external_stylesheets=external_stylesheets,meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=0.7, maximum-scale=1.2,minimum-scale=0.5'}])


SIDEBAR_STYLE={
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#000000",
    'display': 'inline-block'
}
CONTENT_STYLE = {
    "margin-left": "13rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "height":"50%"
}
TOP_CARD_STYLE={
    "margin-left": "13rem",
    "padding": "2rem 1rem",
    "height":"10%"
}


TOP_CARD=dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**INPUTS**")),
        dbc.CardBody(
            [
            html.Button('+ ADD PARAMETER',id='AP_button',n_clicks=0,className='ip_buttons'),
            html.Br(),
            html.Br(),
            dbc.Row([
            dbc.Col([
            html.Label("SELECT DISTRIBUTION",className="ip-labels"),
            dcc.Dropdown(
                     options=[
                         {'label': 'Gaussian', 'value': 'Gaussian'},
                         {'label': 'Analytical', 'value': 'Analytical'},
                         {'label': 'Random', 'value': 'Random'}
                     ],
                     value='Gaussian'
                     , className="m-1",id='drop-1',
                    optionHeight=25,
                style={
                    "width":"90%"
                }
                 ),],
            width=2),
            dbc.Col([
            html.Label("SELECT RANGE",className="ip-labels"),
            dcc.RangeSlider(
                id='rc-slider-track ',
                min=0,
                max=30,
                value=[0, 15],
                marks={
                    0: {'label':'0','style':{'color': '#800080'}},
                    30:{'label':'30','style': {'color': '#800080'}}
                },
                allowCross=False,
                 ),
            ],width=2),

            ]),
        html.Br(),
        dbc.Row([
        dbc.Col([
        dcc.Input(id="input1", type="text", placeholder="Input Function...",className='ip_field'),
           ],
        width=2
        ),
        dbc.Col([
        html.Button('Compute',id='Comp_button',n_clicks=0,className='ip_buttons'),
           ],
        width=3),
        ])]
        )
    ],style=TOP_CARD_STYLE
    ,color="#000000",
    inverse=True
)





SIDEBAR=html.Div([
    html.Label("MODE",className='Side-text'),
    dcc.Link(html.Button('ANALYTICAL MODEL ',className="AT-button"),href='#',className='link'),
    dcc.Link(html.Button('OFFLINE MODEL',className="AT-button"),href='#',className='link')
], style=SIDEBAR_STYLE)

app.layout=html.Div([
SIDEBAR,
TOP_CARD
])



if __name__=="__main__":
    app.run_server(debug=True)