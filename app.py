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
import equadratures as eq
import equadratures.distributions as db
import requests
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go


external_stylesheets=[dbc.themes.BOOTSTRAP]

app=dash.Dash(__name__, external_stylesheets=external_stylesheets,meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=0.2, maximum-scale=1.2,minimum-scale=0.5'}])


TOP_BAR_STYLE={
    "background-color": "#edf2f8",
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

MEAN_VAR_DIST={
    "Gaussian":db.gaussian,
    "Uniform":db.uniform
}
SHAPE_PARAM_DIST={
    "Lognormal":db.lognormal,
}
LOWER_UPPER_DIST={
    "Chebyshev":db.chebyshev
}
LOW_UP_SHA_SHB={
    "Beta":db.beta
}

TOP_CARD=dbc.Card(
    [
        dbc.CardBody(
            [
        dbc.Row([
            dbc.Col([html.Button('+ ADD PARAMETER',id='AP_button',n_clicks=0,className='ip_buttons')],
                    width={"size": 3}),]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Row([html.Label("INPUT DISTRIBUTION")],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
                dbc.Row([
                    dcc.Dropdown(
                        options=[
                            {'label': 'Gaussian', 'value': 'Gaussian'},
                            {'label': 'Uniform', 'value': 'Uniform'},
                            {'label': 'ChebyShev', 'value': 'Chebyshev'},
                            {'label': 'Chi', 'value': 'chi'},
                            {'label': 'Cauchy', 'value': 'cauchy'},
                            {'label': 'LogNormal', 'value': 'Lognormal'},
                            {'label': 'Beta', 'value': 'Beta'}
                        ],
                        placeholder='Select a distribution'
                        , className="m-1",id='drop-1',
                        optionHeight=25,
                        style={
                            "width":"150px",
                        }
                    )
                ]),
            ], lg=2, width=4),
            dbc.Col([
                dbc.Row([html.Label('INPUT STATISTICAL MOMENTS')],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
                dbc.Row([
                    dbc.Col([html.Div([
                            dcc.Input(id="params", type="text",value='',placeholder='',className='ip_field',style={'width': '150px'}),
                        ],style={}),],width={}),
                    dbc.Col([html.Div([
                            dcc.Input(id="params_2", type="text",value='',placeholder='',className='ip_field',style={'width': '150px'}),
                        ],style={},id='wrap_input'),],width={}),
                    dbc.Col([html.Div([
                            dcc.Input(id="params_3", type="text", value='', placeholder='', className='ip_field',style={'width': '150px'}),
                        ],style={}, id='wrap_input_2'),],width={}),
                    dbc.Col([
                        html.Div([
                            dcc.Input(id="params_4", type="text", value='', placeholder='', className='ip_field',style={'width': '150px'}),
                        ],style={}, id='wrap_input_3'),],width={}),
                ]),
            ], lg=5, xs=3, width=8),
            dbc.Col([
                dbc.Row([html.Label('INPUT MIN/MAX VALUE')],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
                dbc.Row([
                    dbc.Col([
                        dcc.Input(id="max_val", type="number", value='', placeholder='Enter maximum value...', className='ip_field',style={'width': '150px'}),
                    ]),
                    dbc.Col([
                        dcc.Input(id="min_val",type="number",value='',placeholder="Enter minimum value...", className='ip_field',style={'width': '150px'})
                    ])
                ], justify='start')
            ], lg=3, xs=2, width=6)

        ],
        no_gutters=True,
        justify='start'),
        html.Br(),
        dbc.Row([
        dbc.Col([
        dcc.Input(id="input1", type="text", placeholder="Input Function...",className='ip_field',style={'width': '150px'}),
           ],
        width=4),
        ]),

        dbc.Row(
            dbc.Col(
                html.Div(id='param_add', children=[])))

            ]
        )
    ]
    ,color="#FFFFFF",
    inverse=True,
    style={"width": "90%",
           "left":"2rem",
           "top":"1rem"},
)


# TOP_TABLE=dbc.Container(
#     fig=go.Figure(data=[go.Table(header=dict(values=['Parameter','Distribution','Mean','Variance','Max','Min']),
#                                  cells=dict(values=[]))])
# )




SIDEBAR=dbc.Navbar([
    dbc.Row([
    dbc.Col(html.Img(src='assets/eq.png',height='50px'),width={'order':'first'}),
    dbc.Col(html.Label("EQUADRATURES",className='Side-text'),width={'offset':0}),
    dbc.Col(dcc.Link(html.Button('ANALYTICAL MODEL',className="AT-button"),href='#',className='link')),
    dbc.Col(dcc.Link(html.Button('OFFLINE MODEL',className="AT-button"),href='#',className='link'),width={}),
   ],
        no_gutters=True,
        align='center'
    )
],

color="#FAFAFA")



app.layout=html.Div([
SIDEBAR,
TOP_CARD
])


@app.callback(
    Output('params','placeholder'),
    Output('params_2','placeholder'),
    Output('params_3','placeholder'),
    Output('params_4','placeholder'),
    Output('wrap_input','style'),
    Output('wrap_input_2','style'),
    Output('wrap_input_3','style'),
    [Input('drop-1','value')],
    prevent_initial_callback=True
)
def UpdateInputField(value):
    show={"display":"block"}
    hide={"display":"none"}
    if value is None:
        return ['Statistical Measures based on Distribution','Statistical Measures based on Distribution',' ',' ',hide,hide,hide]
    if value in MEAN_VAR_DIST.keys():
        return 'Enter Mean...','Enter Variance...',' ',' ',show,hide,hide
    if value in SHAPE_PARAM_DIST.keys():
        return 'Enter Shape...',' ','','',hide,hide,hide
    if value in LOWER_UPPER_DIST.keys():
        return 'Enter Lower Value...','Enter Upper Value...','','',show,hide,hide
    if value in LOW_UP_SHA_SHB.keys():
        return 'Enter Mean...','Enter Variance...','Enter Shape A...','Enter Shape B...',show,show,show


@app.callback(
    Output('AP_button','disabled'),
    [Input('AP_button','n_clicks')]
)
def check_param(n_clicks):
    if n_clicks>4:
        return True
    else:
        return False

@app.callback(
    Output('param_add','children'),
    [Input('AP_button','n_clicks'),
     State('param_add','children')]
)
def addInputs(n_clicks,children):
    add_card=dbc.Row([
        dbc.Col([
            dbc.Row([html.Label("INPUT DISTRIBUTION")],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
        dbc.Row([
            dcc.Dropdown(
                     options=[
                         {'label': 'Gaussian', 'value': 'Gaussian'},
                         {'label': 'Uniform', 'value': 'Uniform'},
                         {'label': 'ChebyShev', 'value': 'Chebyshev'},
                        {'label': 'Chi', 'value': 'chi'},
                        {'label': 'Cauchy', 'value': 'cauchy'},
                        {'label': 'LogNormal', 'value': 'Lognormal'},
                         {'label': 'Beta', 'value': 'Beta'}
                     ],
                     placeholder='Select a distribution'
                     , className="m-1",id={
                    'type':'drop-1',
                    'index':n_clicks
                },
                    optionHeight=25,
                style={
                    "width":"150px",
                }
                 )]),],
            width={'order':'first','offset':0}),
            dbc.Col([
                dbc.Row([html.Label('INPUT STATISTICAL MOMENTS')],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
                dbc.Row([
                    dbc.Col([dcc.Input(id={'type':'params','index':n_clicks}, type="text", value='', placeholder='', className='ip_field',style={'width': '150px'}),
                   ],width={'offset':1}),
            dbc.Col([html.Div([
            dcc.Input(id={'type':'params_2','index':n_clicks}, type="text",value='',placeholder='',className='ip_field',style={'width': '150px'}),
            ],style={},id={'type':'wrap_input','index':n_clicks}),],width={}),
            dbc.Col([html.Div([
                dcc.Input(id={'type':'params_3','index':n_clicks}, type="text", value='', placeholder='', className='ip_field',style={'width': '150px'}),
                    ],style={},id={'type':'wrap_input_2','index':n_clicks}),],width={}),
            dbc.Col([
                html.Div([
                    dcc.Input(id={'type':'params_4','index':n_clicks}, type="text", value='', placeholder='', className='ip_field',style={'width': '150px'}),
                ], style={}, id={'type':'wrap_input_3','index':n_clicks}), ]),]),],width={'offset':1}),
        dbc.Col([
            dbc.Row([html.Label('INPUT MIN/MAX VALUE')],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
            dbc.Row([
            dbc.Col([
            dcc.Input(id={'type':'max_val','index':n_clicks}, type="number", value='', placeholder='Enter maximum value...', className='ip_field',style={'width': '150px'}),
        ],width={'order':'first','size':5}),
        dbc.Col([
        dcc.Input(id={'type':'min_val','index':n_clicks},type="number",value='',placeholder="Enter minimum value...", className='ip_field',style={'width': '150px'})
                ],width={'order':2,'size':3})])
            ],width={"offset":1})

        ],
        no_gutters=True,
        justify='start')
    children.append(add_card)
    return children



if __name__=="__main__":
    app.run_server(debug=True)
