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
    "Gaussian":'mean','variance'
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
                    width={"size": 3}),
            dbc.Col([dcc.Input(id="input1", type="text", placeholder="Input Function...",className='ip_field',style={'width': '150px'}),
           ],width=4),
            ]),

        html.Br(),
        dbc.Row(
            dbc.Col(
                html.Div(id='param_add', children=[])))

            ]
        )
    ]
    , className='top_card',
    color="#FFFFFF",
    inverse=True,
    style={"width": "1300px",
           'height': '350px',
           "left": "2rem",
           "top": "1rem",
           },
)


TOP_GRAPH=html.Div(id='plot_pdf')




NAV=dbc.Navbar([
    dbc.Container([
    dbc.Row([
    dbc.Col(html.Img(src='assets/eq.png',height='50px'),width={'order':'first'}),
    dbc.Col(html.Label("EQUADRATURES",className='Side-text'),width={'offset':0}),
        ],
        align='center',
        no_gutters=True
    ),
    dbc.Nav([
    dbc.NavItem(dbc.NavLink('ANALYTICAL MODEL',href='#',className='link')),
    dbc.NavItem(dbc.NavLink('OFFLINE MODEL',href='#',className='link')),
   ])

],
),
    ],
className='nav',
color="#FAFAFA")


app.layout=html.Div([
NAV,
TOP_CARD,
TOP_GRAPH
])





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
    if n_clicks!=0:

        add_card=dbc.Row([
            dbc.Col([
                dbc.FormGroup([
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
                    )
                ]),
             ])],width=3,lg=2),
            dbc.Col([
                dbc.FormGroup([
                dbc.Row([html.Label('INPUT STATISTICAL MOMENTS')],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
                dbc.Row([
                    dbc.Col([
                            dbc.Input(bs_size="sm",id={'type':'params','index':n_clicks}, type="number",value='',placeholder='',className='ip_field',style={'width': '100px'}),
                        ],width={}),
                    dbc.Col([
                            dbc.Input(bs_size="sm",id={'type':'params_2','index':n_clicks}, type="number",value='',placeholder='...',className='ip_field',style={'width': '100px'}),
                        ],width={}),
                    dbc.Col([
                            dbc.Input(bs_size="sm",id={'type':'params_3','index':n_clicks}, type="number", value='', placeholder='...', className='ip_field',style={'width': '100px'}),
                        ],width={}),
                    dbc.Col([

                            dbc.Input(bs_size="sm",id={'type':'params_4','index':n_clicks}, type="number", value='', placeholder='...', className='ip_field',style={'width': '100px'}),
                        ],width={}),
                ]),
            ]),],lg=4, xs=3, width=4),
            dbc.Col([
                dbc.Row([html.Label('INPUT MIN/MAX VALUE')],style={"color":"#000000","font-size":"0.9rem","font-family":"Times New Roman, Times, serif"}),
                dbc.Row([
                    dbc.Col([
                        dbc.Input(bs_size="sm",id={'type':'max_val','index':n_clicks}, type="number", value='', placeholder='Maximum value...', className='ip_field',style={'width': '100px'}),
                    ]),
                    dbc.Col([
                        dbc.Input(bs_size="sm",id={'type':'min_val','index':n_clicks},type="number",value='',placeholder="Minimum value...", className='ip_field',style={'width': '100px'})
                    ]),
                    dbc.Col([
                        dbc.Checklist(
                            options = [
            {"label": "Pdf_{}".format(n_clicks), "value": "val_{}".format(n_clicks)},
                            ],
                            switch=True,
                            id={
                                "type":"radio_pdf",
                                "index":"n_clicks"
                            }
                        )
                    ])

                ], justify='start',no_gutters=True)
            ], lg=3, xs=2, width=3),
        ],
        no_gutters=True,
        justify='start')
    else:
        add_card=dbc.Row()
    children.append(add_card)
    return children



@app.callback(
    Output({'type': 'params', 'index': dash.dependencies.MATCH},'placeholder'),
    Output({'type': 'params_2', 'index': dash.dependencies.MATCH},'placeholder'),
    Output({'type': 'params_3', 'index': dash.dependencies.MATCH},'placeholder'),
    Output({'type': 'params_4', 'index': dash.dependencies.MATCH},'placeholder'),
    Output({'type': 'params_2', 'index': dash.dependencies.MATCH},'disabled'),
    Output({'type': 'params_3', 'index': dash.dependencies.MATCH},'disabled'),
    Output({'type': 'params_4', 'index': dash.dependencies.MATCH},'disabled'),
    [Input({'type':'drop-1','index':dash.dependencies.MATCH},'value')],
    prevent_initial_callback=True,
)
def UpdateInputField(value):
    show=False
    hide=True
    if value is None:
        return ['Statistical Measures based on Distribution',' ',' ',' ',hide,hide,hide]
    if value in MEAN_VAR_DIST.keys():
        return 'Mean...','Variance...',' ',' ',show,hide,hide
    if value in SHAPE_PARAM_DIST.keys():
        return 'Shape...',' ','','',hide,hide,hide
    if value in LOWER_UPPER_DIST.keys():
        return 'Lower Value...','Upper Value...','','',show,hide,hide
    if value in LOW_UP_SHA_SHB.keys():
        return 'Mean...','Variance...','Shape A...','Shape B...',show,show,show



def CreateParam(distribution,shape_parameter_A,shape_parameter_B,shape_A,shape_B,min,max):
    param_obj=eq.Parameter(distribution=distribution,shape_parameter_A=shape_parameter_A,shape_parameter_B=shape_parameter_B,
                           lower=min,upper=max,order=3)
    s_values,pdf=param_obj.get_pdf()
    return param_obj,s_values,pdf

@app.callback(
    Output({'type': 'plot_pdf', 'index': dash.dependencies.MATCH}, 'children'),
    [Input({'type': 'radio_pdf', 'index': dash.dependencies.MATCH}, 'value')],
    [State({'type': 'params', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'params_2', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'params_3', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'params_4', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'drop-1', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'max_val', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'min_val', 'index': dash.dependencies.MATCH}, 'value'),
     State({'type': 'AP_button', 'index': dash.dependencies.MATCH}, 'n_clicks')
     ],
    prevent_initial_call=True
)
def PlotPdf(pdf_val,param1_val,params2_val,params3_val,params4_val,drop1_val,max_val,min_val,n_clicks):
    fig=go.Figure()
    if pdf_val == 'val_{}'.format(n_clicks):
        if params4_val and params3_val is None:
            param,s_values,pdf=CreateParam(distribution=drop1_val,shape_parameter_A=param1_val,shape_parameter_B=params2_val,
                                       shape_A=None,shape_B=None,min=min_val,max=max_val)

            fig.add_trace(go.Scatter(x=s_values,y=pdf,line = dict(color='rgba(0,0,0,0)'),fill='tonexty')),
            return fig
        else:
            param, s_values, pdf = CreateParam(distribution=drop1_val, shape_parameter_A=param1_val,
                                           shape_parameter_B=params2_val,
                                           shape_A=None, shape_B=None, min=min_val, max=max_val)

            fig.add_trace(go.Scatter(x=s_values, y=pdf, line=dict(color='rgba(0,0,0,0)'), fill='tonexty')),
            return fig
    else:
        fig.add_trace(x=[],y=[])
        return fig




if __name__=="__main__":
    app.run_server(debug=True)
