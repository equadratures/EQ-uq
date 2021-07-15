import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_table
import dash_daq as daq
import equadratures as eq
import equadratures.distributions as db
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import json
import jsonpickle
import ast
from equadratures import *
import numexpr as ne
import scipy as sp
import time
from app import app
from navbar import navbar
from utils import convert_latex
import requests
import timeit


###################################################################
# Style Elements
###################################################################

TOP_BAR_STYLE = {
    "background-color": "#edf2f8",
}
CONTENT_STYLE = {
    "margin-left": "13rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "height": "50%"
}
TOP_CARD_STYLE = {
    "margin-left": "13rem",
    "padding": "2rem 1rem",
    "height": "10%"
}
###################################################################
# Distributions Data (@To add more)
###################################################################

MEAN_VAR_DIST = {
    "Gaussian": db.gaussian,
}
LOWER_UP_UNI_DIST = {
    "Uniform": db.uniform
}
SHAPE_PARAM_DIST = {
    "Lognormal": db.lognormal,
}
LOWER_UPPER_DIST = {
    "Chebyshev": db.chebyshev
}
LOW_UP_SHA_SHB = {
    "Beta": db.beta
}

###################################################################
# Analytical Introduction Text
###################################################################

TOP_TEXT=r'''
## ANALYTICAL MODEL

This model is used for uncertainty quantification for user defined parameters and polynomial. The sobol indices provides insights about senstivity analysis
'''
top_text=dcc.Markdown(convert_latex(TOP_TEXT),style={"margin-left": "2rem"})

toptext = dbc.Container([TOP_TEXT])

###################################################################
# Parameter Definition Card
###################################################################

TOP_CARD = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**Parameter Definition**",style={"color": "#000000"})),
        dbc.CardBody(
            [
                dbc.Row([
                    dbc.Col(
                        [dbc.Button('+ ADD PARAMETER', id='AP_button', n_clicks=0, color="primary", className="py-0")],
                        width={"size": 3}),
                        dbc.Tooltip(
                            dcc.Markdown([
                                 "**Note:** Maximum 5 parameters can be used for analytical model"]),
                                 target="AP_button",
                                 placement='right'
                             ),
                    dbc.Col([dcc.Input(id="input_func", type="text", placeholder="Input Function...",
                                       className='ip_field', debounce=True
                                       , style={'width': '150px'}),
                             dbc.Col([
                                 dbc.Spinner(html.Div(id='loading'), color="primary", show_initially=False,
                                                 spinner_style={'top': "-5rem"})
                                         ]),
                             dbc.Tooltip(
                                 "The variables for input function should be of form x1,x2...",
                                 target="input_func",
                                 placement='right'
                             ),
                             ], width=4),
                ]),
                html.Br(),
                html.Br(),
                dcc.Store(id='ndims'),
                dbc.Row(
                    dbc.Col(
                        html.Div(id='param_add', children=[])))

            ],
        className='top_card',
        )
    ],
    id='top_card',
    color="#FFFFFF",
    inverse=True,
    style={"width": "96%",
           # 'height': '740px',
           "left": "2rem",
           "top": "0rem",
           },
)

###################################################################
# PDF Plot
###################################################################

PDF_PLOT=dbc.Container([dcc.Graph(id='plot_pdf', style={'width': '100%',})])


PDF_GRAPH = dbc.Card([
    dbc.CardHeader(dcc.Markdown("**Probability Density Function**",style={"color": "#000000"})),
    dbc.CardBody([

    dbc.Row([
    PDF_PLOT
        ])
    ])

], style={'top': "0.5rem", 'left': '2rem','width':'620px'})





###################################################################
# Card for setting basis, levels, q and compute cardinality
###################################################################

BASIS_CARD = dbc.Card([
    dbc.CardHeader([dcc.Markdown("**Basis Selection**",style={"color": "#000000"})]),
    dbc.CardBody([
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                options=[
                    {'label': 'Univariate', 'value': 'univariate'},
                    {'label': 'Total-order', 'value': 'total-order'},
                    {'label': 'Tensor-grid', 'value': 'tensor-grid'},
                    {'label': 'Sparse-grid', 'value': 'sparse-grid'},
                    {'label': 'Hyperbolic-basis', 'value': 'hyperbolic-basis'},
                    {'label': 'Euclidean-degree', 'value': 'euclidean-degree'}
                ],
                placeholder='Select Basis',
                className="m-1", id='drop_basis',
                optionHeight=45,
                style={
                    "width": "165px",

                }
            ),

        ], width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Input(bs_size="sm", id='q_val', type="number", value=np.nan, placeholder='q',
                              className='ip_field',
                              disabled=True, style={'width': '100px'}),
                ], width=3),
                dbc.Col([
                    dbc.Input(bs_size="sm", id='levels', type='number', value=np.nan, placeholder='Level',
                              className='ip_field',
                              disabled=True, style={'width': '100px'})
                ], width=3),
            ], no_gutters=True,
                justify="start")
        ], width=9)
    ],
        no_gutters=True,
        justify='start'
    ),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Exponential', 'value': 'exponential'},
                ],
                placeholder='Growth Rule',
                className="m-1", id='basis_growth_rule',
                optionHeight=45,
                disabled=True,
                style={
                    "width": "800px",
                    "display": "flex"
                }
            ),

        ])
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([dbc.Button('Cardinality Check', id='CC_button', n_clicks=0, className='ip_buttons',color='primary',
                            disabled=True)]),
        ]),
    dbc.Row([
        dbc.Col(dbc.Alert(id='compute-warning',color='danger',is_open=False),width='auto')
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Input(bs_size="sm", id='op_box', type="number", value='', placeholder='Cardinality...', className='ip_field',
                           disabled=True, style={'width': '100px'})], width='auto'),
        dcc.Store(id='ParamObjects'),
        dcc.Store(id='PolyObject'),
        dbc.Col(dcc.Graph(id='plot_basis',
                          style={'display': 'inline-block', 'width': '450px', 'margin-top': '-80px', 'height': '250px',
                                 'margin-left': '90px'}), width=8),

    ], no_gutters=True,
        justify='start')
        ])
], style={"top": "0.5rem", "width": "96%", "height": "525px"})


###################################################################
# UQ CARD:-
# Outputs: Mean, Variance, R2, Sobol Indices, Polyfit
###################################################################

params = dict(
                gridcolor="white",
                showbackground=False,
                linecolor='black',
                tickcolor='black',
                ticks='outside',
                zerolinecolor="white")

layout = dict(margin={'t': 0, 'r': 0, 'l': 0, 'b': 0, 'pad': 10}, autosize=True,
        scene=dict(
            aspectmode='cube',
            xaxis=params,
            yaxis=params,
            zaxis=params,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
polyfig3D = go.Figure(layout=layout)
polyfig3D.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
polyfig3D.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')

polyfig3D.add_trace(go.Surface(x=[], y=[], z=[], showscale=False, opacity=0.5,
                    colorscale=[[0, 'rgb(178,34,34)'], [1, 'rgb(0,0,0)']]))

polyfig3D.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers',
                   marker=dict(size=10, color="rgb(144, 238, 144)", opacity=0.6,
                               line=dict(color='rgb(0,0,0)', width=1))))

COMPUTE_CARD = dbc.Card([
    dbc.CardHeader([dcc.Markdown("**Compute Uncertainty**",style={"color": "#000000"})]),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            options=[
                                {'label': 'Least-squares', 'value': 'least-squares'},
                                {'label': 'Numerical-integration', 'value': 'numerical-integration'},
                            ],
                            placeholder='Solver Method',
                            value='numerical-integration',
                            className="m-1", id='solver_method',
                            optionHeight=45,
                            style={
                                "width": "200px",
                            }
                        ),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(['Compute Uncertainty'], id='CU_button', n_clicks=0, className='ip_buttons',color='primary',
                                   disabled=True)
                    ]),
                    dbc.Tooltip(['Cardinality Check and Input Function are necessary for computing uncertainty'
                                 ],id='cu_tooltip',target="CU_button",
                                 placement='right'),
                    dcc.Store(id='ModelSet'),
                    dcc.Store(id='True_vals')

                ]),
                html.Br(),
                html.Br(),
                dbc.Row([
                            dbc.Col([
                                dbc.FormGroup([
                                dbc.Label("MEAN",html_for='mean'),
                            dbc.Row([
                                dbc.Col([
                                dbc.Input(bs_size="sm", id='mean', type='number', value=np.nan, placeholder='Mean...',
                                          className='ip_field',
                                          disabled=True, style={'width': '100px'})
                                    ])
                            ],style={'align':'centre'})
                                    ])
                            ]),
                            dbc.Col([
                        dbc.FormGroup([
                                dbc.Label("VARIANCE",html_for='variance'),
                            dbc.Row([
                            dbc.Col([
                            dbc.Input(bs_size="sm", id='variance', type='number', value=np.nan,
                                          placeholder='Variance..,',
                                          className='ip_field',
                                          disabled=True, style={'width': '100px'})
                                ])
                                ])
                            ])
                            ]),
                            dbc.Col([
                        dbc.FormGroup([
                                dbc.Label("R2 Score",html_for='r2_score'),
                            dbc.Col([
                            dbc.Row([
                                dbc.Input(bs_size="sm", id='r2_score', type='number', value=np.nan,
                                          placeholder='R2 Score..,',
                                          className='ip_field',
                                          disabled=True, style={'width': '100px'})
                                ])
                            ])
                            ])
                            ])




                    ]),
                html.Br(),
                dbc.Row([
                        dbc.Col([
                    dbc.FormGroup([
                        dbc.Label("SENSTIVITY ANALYSIS",html_for='sobol_order'),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Order 1', 'value': 1},
                                        {'label': 'Order 2', 'value': 2},
                                        {'label': 'Order 3', 'value': 3},

                                    ],
                                    placeholder='Interaction Order',
                                    className="m-1", id='sobol_order',
                                    optionHeight=45,
                                    style={
                                        "width": "200px",
                                    }
                                ),
                            ]), ]),
                        ]),
                        html.Br(),

                        dbc.Row([
                            dbc.FormGroup([
                            dbc.Col([
                                    dcc.Graph(id='Sobol_plot', style={'width': '400px', 'top': '-14rem','height':'300px',
                                                                  'left':'20px'})
                            ])
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Store(id='Sobol_Indices'),
                                dcc.Store(id='ndims')
                            ])
                        ])

                    ])
                ])
            ], width=5),

            dbc.Col([
                dbc.Row([
                dcc.Graph(id='plot_poly_3D', style={'width': '600px'}, figure=polyfig3D),
                    ]),
dbc.Row([
dbc.Col([
    dbc.Row([
                dbc.Col(["1D"],width='auto',style={'margin-left':'180px'}),
                dbc.Col(daq.ToggleSwitch(id='toggle',value=False,disabled=True),width='auto'),
                dbc.Col(["2D"],width='auto'),

            ])
            ]),
])
            ], width=6),




        ]),
        dbc.Row([
        ])
    ])
], style={"top": "0.7rem", "margin-left": "3.5rem","height": "680px",'width':'94.5%'})

# PLOYFIT_CARD=dbc.Card(
#     [
#         dbc.CardHeader([dcc.Markdown("**Compute Uncertainty**",style={"color": "#000000"})]),
#         dbc.CardBody([
#             dbc.Col([
#                 dbc.Row([
#                 dcc.Graph(id='plot_poly_3D', style={'width': '600px'}),
#                     ]),
# dbc.Row([
# dbc.Col([
#     dbc.Row([
#                 dbc.Col(["1D"],width='auto',style={'margin-left':'180px'}),
#                 dbc.Col(daq.ToggleSwitch(id='toggle',value=False,disabled=True),width='auto'),
#                 dbc.Col(["2D"],width='auto'),
#
#             ])
#             ]),
# ])
#             ], width=6),
#
#         ])
#     ],
#     style={"top":'7.7rem','margin-left':'4.5rem','height':'680px'}
# )


###################################################################
# Analytical Layout
###################################################################

layout = html.Div([
    navbar,
    top_text,
    dbc.Row([
        dbc.Col(TOP_CARD, width=12),
    ]),
    dbc.Row([
        dbc.Col(PDF_GRAPH, width=5),
        dbc.Col(BASIS_CARD, width=7)
    ],

        no_gutters=False
    ),
    dbc.Row([
        COMPUTE_CARD,

    ])
])

###################################################################
# Callback for disabling AP button after 5 clicks
###################################################################

@app.callback(
    Output('AP_button', 'disabled'),
    [Input('AP_button', 'n_clicks')]
)
def check_param(n_clicks):
    if n_clicks > 4:
        return True
    else:
        return False

###################################################################
# Callback for adding parameter to param definition card
###################################################################


@app.callback(
    Output('param_add', 'children'),
    Output('loading', 'children'),
    [Input('AP_button', 'n_clicks'),
     State('param_add', 'children')]
)
def addInputs(n_clicks, children):
    if n_clicks != 0:

        add_card = dbc.Row([
            dbc.Col([
                dbc.Form([
                        dbc.Label('INPUT DISTRIBUTION', html_for='drop-1',style={"color": "#000000"}),
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
                            placeholder='Select a distribution',
                            value='Uniform',
                            className="m-1", id={
                                'type': 'drop-1',
                                'index': n_clicks
                            },
                            clearable=False,
                            optionHeight=20,
                            style={
                                "width": "150px",
                            }
                        )

                ])], width=3, lg=2),
            dbc.Col([
            dbc.Row([
            dbc.Col([
                dbc.Form([
                    dbc.Label('INPUT STATISTICAL MOMENTS',html_for='params',
                                       style={"color": "#000000","font-family": "Raleway"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(bs_size="sm", id={'type': 'params', 'index': n_clicks}, type="number",
                                      value=np.nan, placeholder='',
                                      debounce=True, className='ip_field', style={'width': '100px'}),
                        ], width=4),
                        dbc.Col([
                            dbc.Input(bs_size="sm", id={'type': 'params_2', 'index': n_clicks}, type="number",
                                      value=np.nan, placeholder='...',
                                      debounce=True, className='ip_field', style={'width': '100px'}),
                        ], width=4),
                    ])
                ]), ], lg=4, xs=5, width=5),
            dbc.Col([
                dbc.Form([

                    dbc.Label('INPUT MIN/MAX/ORDER VALUE',
                              html_for='min_val',style={"color": "#000000","font-family": 'Raleway'}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(bs_size="sm", id={'type': 'min_val', 'index': n_clicks}, type="number",
                                      value=np.nan, placeholder='Minimum value...',
                                      debounce=True, className='ip_field', style={'width': '100px'}),
                        ], width='auto'),
                        dbc.Col([
                            dbc.Input(bs_size="sm", id={'type': 'max_val', 'index': n_clicks}, type="number",
                                      value=np.nan, placeholder="Maximum value...",
                                      debounce=True, className='ip_field', style={'width': '100px'})
                        ], width='auto'),
                        dbc.Col([
                            dbc.Input(bs_size="sm", id={'type': 'order', 'index': n_clicks}, type="number",
                                      value=np.nan,
                                      placeholder="Order",
                                      debounce=True, className='ip_field', style={'width': '100px'})
                        ], width='auto'),
                        dbc.Col([
                            dbc.Checklist(
                                options=[
                                    {"label": "x{}".format(n_clicks), "value": "val_{}".format(n_clicks)},
                                ],
                                switch=True,
                                value=[0],
                                id={
                                    "type": "radio_pdf",
                                    "index": n_clicks
                                }
                            )
                        ], width='auto'),

                    ], justify='start')
                ])], lg=5, xs=4, width='auto'),
            ],
                no_gutters=True)
            ],width=9, lg=9),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
        ],
            no_gutters=True,
            justify='start')
        wait = time.sleep(1)
    else:
        add_card = dbc.Row()
    children.append(add_card)
    wait = None
    return children, wait

###################################################################
# Callback for disabling Cardinality Check button
###################################################################



@app.callback(
    Output('CC_button','disabled'),
    [
    Input('AP_button','n_clicks')
        ]
)
def CheckifAPClicked(n_clicks):
    if n_clicks>0:
        return False
    else:
        return True

###################################################################
# Callback for disabling Compute Uncertainty button
###################################################################

@app.callback(
    Output('CU_button','disabled'),
    [
        Input('CC_button','n_clicks'),
        Input('input_func','value')
    ]
)
def CheckifCCClickd(n_clicks,input_val):
    if n_clicks>0 and input_val is not None:
        return False
    else:
        return True

###################################################################
# Callback to map input boxes to distributions
###################################################################


@app.callback(
    Output({'type': 'params', 'index': dash.dependencies.MATCH}, 'placeholder'),
    Output({'type': 'params_2', 'index': dash.dependencies.MATCH}, 'placeholder'),
    Output({'type': 'min_val', 'index': dash.dependencies.MATCH}, 'placeholder'),
    Output({'type': 'max_val', 'index': dash.dependencies.MATCH}, 'placeholder'),
    Output({'type': 'params', 'index': dash.dependencies.MATCH}, 'disabled'),
    Output({'type': 'params_2', 'index': dash.dependencies.MATCH}, 'disabled'),
    Output({'type': 'min_val', 'index': dash.dependencies.MATCH}, 'disabled'),
    Output({'type': 'max_val', 'index': dash.dependencies.MATCH}, 'disabled'),
    [Input({'type': 'drop-1', 'index': dash.dependencies.MATCH}, 'value')],
    prevent_initial_callback=True,
)
def UpdateInputField(value):
    show = False
    hide = True
    if value is None:
        return ['Statistical Measures based on Distribution', '...', '...', '...', hide, hide, hide]
    if value in MEAN_VAR_DIST.keys():
        return 'Mean...', 'Variance...', ' ', ' ', show, show, hide, hide
    if value in LOWER_UP_UNI_DIST.keys():
        return '', '', 'Min Value...', 'Max Value...', hide, hide, show, show
    if value in SHAPE_PARAM_DIST.keys():
        return 'Shape...', ' ', '', '', show, hide, hide, hide
    if value in LOWER_UPPER_DIST.keys():
        return 'Lower Value...', 'Upper Value...', 'Min Value...', 'Max Value...', show, show, show, show
    if value in LOW_UP_SHA_SHB.keys():
        return 'Shape A...', 'Shape B...', 'Min Value...', 'Max Value...', show, show, show, show



###################################################################
# Callback to create EQ Param Objects
###################################################################


@app.callback(
    Output('ParamObjects', 'data'),
    Output('ndims','data'),
    [
        Input('AP_button', 'n_clicks'),
        Input({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'order', 'index': dash.dependencies.ALL}, 'value'),

    ],
    prevent_intial_call=True
)
def ParamListUpload(n_clicks, shape_parameter_A, shape_parameter_B, distribution, max, min, order):
    i = len(distribution)
    param_list = []
    Show=False
    Block=True
    if i > 0:
        for j in range(i):
            if distribution[j] in MEAN_VAR_DIST.keys():
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                     , shape_parameter_B=shape_parameter_B[j], lower=min[j], upper=max[j],
                                     order=order[j])

            elif distribution[j] in LOW_UP_SHA_SHB.keys():
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                     , shape_parameter_B=shape_parameter_B[j], lower=min[j], upper=max[j],
                                     order=order[j])


            elif distribution[j] in SHAPE_PARAM_DIST.keys():
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j],
                                     order=order[j])

            elif distribution[j] in LOWER_UP_UNI_DIST.keys():
                param = eq.Parameter(distribution=distribution[j], lower=min[j], upper=max[j], order=order[j])


            elif distribution[j] in LOWER_UPPER_DIST.keys():
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j],
                                     shape_parameter_B=shape_parameter_B[j], lower=min[j], upper=max[j], order=order[j])

            param_list.append(param)
    return jsonpickle.encode(param_list),len(param_list)


###################################################################
# Function to compute s_values and pdf
###################################################################

def CreateParam(distribution, shape_parameter_A, shape_parameter_B, min, max, order):
    param_obj = eq.Parameter(distribution=distribution, shape_parameter_A=shape_parameter_A,
                             shape_parameter_B=shape_parameter_B,
                             lower=min, upper=max, order=order)
    s_values, pdf = param_obj.get_pdf()
    return param_obj, s_values, pdf


###################################################################
# Callback to plot pdf
###################################################################


@app.callback(
    Output('plot_pdf', 'figure'),
    Input({'type': 'radio_pdf', 'index': dash.dependencies.ALL}, 'value'),
    [State({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'order', 'index': dash.dependencies.ALL}, 'value'),

     ],
    prevent_initial_call=True
)
def PlotPdf(pdf_val, param1_val, params2_val, drop1_val, max_val, min_val, order):
    layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
              'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}

    fig = go.Figure(layout=layout)
    fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
    fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')

    ctx = dash.callback_context
    id = ctx.triggered[0]['prop_id'].split('.')[0]
    idx = ast.literal_eval(id)['index']

    elem = [0, 'val_{}'.format(idx)]
    check = elem in pdf_val
    if check:
        i = pdf_val.index(elem)
        if param1_val and params2_val is None:
            param, s_values, pdf = CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                                               shape_parameter_B=params2_val[i],
                                               min=min_val[i], max=max_val[i], order=order[i])

            fig.add_trace(go.Scatter(x=s_values, y=pdf, line=dict(color='rgb(0,176,246)'), fill='tonexty', mode='lines',
                                     name='Polyfit', line_width=4, line_color='black')),
        else:
            param, s_values, pdf = CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                                               shape_parameter_B=params2_val[i], min=min_val[i], max=max_val[i],
                                               order=order[i])

            fig.add_trace(go.Scatter(x=s_values, y=pdf, line=dict(color='rgb(0,176,246)'), fill='tonexty')),
    return fig



###################################################################
# Callback to handle toggle switch in param definition card
###################################################################


@app.callback(
    Output({'type': 'radio_pdf', 'index': dash.dependencies.ALL}, 'value'),
    Input({'type': 'radio_pdf', 'index': dash.dependencies.ALL}, 'value'),
    prevent_initial_call=True
)
def setToggles(pdf_val):
    ctx = dash.callback_context
    id = ctx.triggered[0]['prop_id'].split('.')[0]
    idx = ast.literal_eval(id)['index']

    elem = [0, 'val_{}'.format(idx)]
    check = elem in pdf_val
    ret_vals = pdf_val
    if check:
        i = pdf_val.index(elem)
        ret_vals[i] = elem

        for j in range(len(ret_vals)):
            if j != i:
                ret_vals[j] = [0]

        test = [[0] if j != i else elem for j, x in enumerate(pdf_val)]
    return ret_vals


###################################################################
# Callback to disable basis card input boxes based on basis selection
###################################################################



@app.callback(
    Output('q_val', 'disabled'),
    Output('levels', 'disabled'),
    Output('basis_growth_rule', 'disabled'),
    [Input('drop_basis', 'value')],
    prevent_initial_call=True
)
def BasisShow(value):
    show = False
    hide = True
    if value is not None:
        if value == 'sparse-grid':
            return hide, show, show
        elif value == 'hyperbolic-basis':
            return show, hide, hide
        else:
            return hide, hide, hide
    else:
        return hide, hide, hide


def Set_Basis(basis_val, order, level, q_val, growth_rule):
    basis_set = Basis('{}'.format(basis_val), orders=order, level=level, q=q_val, growth_rule=growth_rule)
    return basis_set


def Set_Polynomial(parameters, basis, method):
    myPoly = eq.Poly(parameters=parameters, basis=basis, method=method)
    return myPoly

###################################################################
# Callback for automatic selection of solver method based on basis selection
###################################################################

@app.callback(
    Output('solver_method', 'value'),
    [Input('drop_basis', 'value')],
    prevent_initial_call=True
)
def SetMethod(drop_basis):
    if drop_basis == 'total-order':
        return 'least-squares'
    else:
        return 'numerical-integration'



###################################################################
# Callback for computing cardinality and output warning if necessary
###################################################################

@app.callback(
    [Output('op_box', 'value'),
     Output('PolyObject', 'data'),
     Output('compute-warning','is_open'),
     Output('compute-warning','children')
     ],
    [
        Input('CC_button', 'n_clicks'),
        Input('ParamObjects', 'data'),
        Input('ndims','data')],
    [
        State('AP_button', 'n_clicks'),
        State('drop_basis', 'value'),
        State('q_val', 'value'),
        State('levels', 'value'),
        State('basis_growth_rule', 'value'),
        State('solver_method', 'value')
    ],
    prevent_initial_call=True
)
def OutputCardinality(n_clicks, param_obj,ndims,params_click, basis_select, q_val, levels, growth_rule, solver_method):
    if n_clicks != 0:
        if basis_select is None:
            return 'Error...',None,True,'No basis value selected'
        elif basis_select=='sparse-grid' and (levels or growth_rule is None):
            return 'ERROR...',None,True,'Enter the required values'
        else:
            param_data = jsonpickle.decode(param_obj)
            basis_ord=[]
            for elem in param_data:
                basis_ord.append(elem.order)
            mybasis = Set_Basis(basis_val=basis_select, order=basis_ord, level=levels, q_val=q_val, growth_rule=growth_rule)
            myPoly = eq.Poly(parameters=param_data, basis=mybasis, method=solver_method)


            return [mybasis.get_cardinality(), jsonpickle.encode(myPoly),False,None]
    else:
        raise PreventUpdate



###################################################################
# Plotting Function: To plot basis 1D/2D/3D
###################################################################


@app.callback(
    Output('plot_basis', 'figure'),
    [
        Input('PolyObject', 'data'),
        Input('AP_button', 'n_clicks'),
        Input('ndims','data')
    ]
)
def PlotBasis(poly, n_clicks,ndims):
    if poly is not None:
        myPoly = jsonpickle.decode(poly)
        DOE = myPoly.get_points()
        layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                   'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True,
                  "xaxis":{"title": r'x/C'}, "yaxis": {"title": r'y/C'}}

        fig = go.Figure(layout=layout)
        fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
        fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside',
                         zerolinecolor='lightgrey')
        if ndims == 1:
            fig.add_trace(go.Scatter(x=DOE, y=DOE, mode='markers',marker=dict(size=5, color="rgb(144, 238, 144)", opacity=0.6,
                                               line=dict(color='rgb(0,0,0)', width=1))))
            return fig
        elif ndims == 2:
            fig.add_trace(go.Scatter(x=DOE[:, 0], y=DOE[:, 1],mode='markers',marker=dict(size=5, color="rgb(144, 238, 144)", opacity=0.6,
                                               line=dict(color='rgb(0,0,0)', width=1))))
            return fig
        elif ndims>=3:
            fig.update_layout(dict(margin={'t': 0, 'r': 0, 'l': 0, 'b': 0, 'pad': 10}, autosize=True,
                      scene=dict(
                          aspectmode='cube',
                          xaxis=dict(
                              title='X1',
                              gridcolor="white",
                              showbackground=False,
                              linecolor='black',
                              tickcolor='black',
                              ticks='outside',
                              zerolinecolor="white", ),
                          yaxis=dict(
                              title='X2',
                              gridcolor="white",
                              showbackground=False,
                              linecolor='black',
                              tickcolor='black',
                              ticks='outside',
                              zerolinecolor="white"),
                          zaxis=dict(
                              title='X3',
                              backgroundcolor="rgb(230, 230,200)",
                              gridcolor="white",
                              showbackground=False,
                              linecolor='black',
                              tickcolor='black',
                              ticks='outside',
                              zerolinecolor="white", ),
                      ),
                      ))
            fig.add_trace(go.Scatter3d(x=DOE[:, 0], y=DOE[:, 1], z=DOE[:, 2], mode='markers',
                                       marker=dict(size=10, color="rgb(144, 238, 144)", opacity=0.6,
                                                   line=dict(color='rgb(0,0,0)', width=1))))
            return fig

    else:
        raise PreventUpdate

###################################################################
# Callback to set Poly object, calculate mean, variance, r2_score and compute Sobol_Indices
###################################################################

@app.callback(
    [Output('ModelSet', 'data'),
     Output('mean', 'value'),
     Output('variance', 'value'),
     Output('r2_score', 'value'),
     Output('True_vals', 'data'),
     Output('Sobol_plot','figure'),
     ],
    [
        Input('PolyObject', 'data'),
        Input('input_func', 'value'),
        Input('CU_button', 'n_clicks'),
        Input('AP_button', 'n_clicks'),
        Input('sobol_order', 'value'),
        ]
)
def SetModel(poly, expr, compute_button, n_clicks, order):
    if compute_button != 0:
        myPoly = jsonpickle.decode(poly)
        x = [r"x{} = op[{}]".format(j, j - 1) for j in range(1, n_clicks + 1)]

        def f(op):
            for i in range(n_clicks):
                exec(x[i])
            return ne.evaluate(expr)
        myPoly.set_model(f)
        values = myPoly.get_mean_and_variance()
        mean = values[0]
        variance = values[1]
        DOE = myPoly.get_points()
        y_true = []
        for i in range(len(DOE)):
            y_true.append(f(DOE[i]))
        y_true = np.array(y_true)
        y_true = y_true.reshape(-1, 1)
        y_pred = myPoly.get_polyfit(DOE).squeeze()
        y_pred = y_pred.reshape(-1, 1)
        r2_score = eq.datasets.score(y_true, y_pred, metric='r2')
        layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                              'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
        fig=go.Figure(layout=layout)
        fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
        fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside',
                         zerolinecolor='lightgrey')
        if order is not None:
            ndims=myPoly.dimensions
            sobol_indices=myPoly.get_sobol_indices(order=order)
            layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                              'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
            fig=go.Figure(layout=layout)
            if order==1:
                fig.update_yaxes(title=r'$S_{i}$')
                labels = [r'$X_%d$' % i for i in range(int(ndims))]
                to_plot = [sobol_indices[(i,)] for i in range(int(ndims))]
            elif order==2:
                fig.update_yaxes(title=r'$S_{ij}$')
                labels = [r'$S_{%d%d}$' % (i, j) for i in range(int(ndims)) for j in range(i + 1, int(ndims))]
                to_plot = [sobol_indices[(i, j)] for i in range(int(ndims)) for j in range(i + 1, int(ndims))]
            elif order==3:
                fig.update_yaxes(title=r'$S_{ijk}$')
                labels = [r'$S_{%d%d%d}$' % (i, j, k) for i in range(int(ndims)) for j in range(i + 1, int(ndims)) for k in
                                  range(j + 1, int(ndims))]
                to_plot = [sobol_indices[(i, j, k)] for i in range(int(ndims)) for j in range(i + 1, int(ndims)) for k in
                                   range(j + 1, int(ndims))]
            fig.update_xaxes(nticks=len(sobol_indices),tickvals=labels,tickangle=45)
            data=go.Bar(
            x=np.arange(len(sobol_indices)),
            y=to_plot,marker_color='LightSkyBlue',marker_line_width=2,marker_line_color='black')
            fig = go.Figure(layout=layout,data=data)
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle=-30)

        return jsonpickle.encode(myPoly), mean, variance, r2_score, jsonpickle.encode(y_true),fig ###
    else:
        raise PreventUpdate


###################################################################
# Disabling toggle for 1D/2D for polyfit plotting function
###################################################################

@app.callback(
    Output('toggle','disabled'),
    Input('CU_button','n_clicks')
)
def ToggleCheck(n_clicks):
    if n_clicks>0:
        return False
    else:
        return True

# @app.callback(
#     Output('cu_tooltip','children'),
#     [Input('CC_button','n_clicks'),
#      Input('input_func','value'),
#      Input('cu_tooltip','children')]
# )
# def TooltipVal(n_clicks,input_func,children):
#     if n_clicks==0:
#         print(n_clicks)
#         add_val=dcc.Markdown(["Cardinality Check is required for computing uncertainty"])
#     elif input_func is None:
#         add_val=dcc.Markdown(["Please Enter the input function to continue"])
#     else:
#         add_val=None
#     print(add_val)
#     children = children.append(add_val)
#     return children


###################################################################
# Plotting Function: Polyfit plot
###################################################################

@app.callback(
    Output('plot_poly_3D', 'figure'),
    [
        Input('ModelSet', 'data'),
        Input('CU_button', 'n_clicks'),
        Input('True_vals', 'data'),
        Input('AP_button','n_clicks'),
        Input('toggle','value'),
        Input('ndims','data')
    ],
    State('plot_poly_3D', 'figure'),
    prevent_initial_call=True
)
def Plot_poly_3D(ModelSet, n_clicks, true_vals, param_num, dims,ndims,fig):
    if ModelSet is not None:
        if dims:
            if param_num==2:
                layout = dict(margin={'t': 0, 'r': 0, 'l': 0, 'b': 0, 'pad': 0}, autosize=True,
                        scene=dict(
                          aspectmode='cube',
                          xaxis=dict(
                              title='X1'),
                          yaxis=dict(
                              title='X2'),
                          zaxis=dict(
                              title=r'f(x)'),
                      ),
                      )
                myPoly = jsonpickle.decode(ModelSet)
                y_true = jsonpickle.decode(true_vals)
                myPolyFit = myPoly.get_polyfit
                DOE = myPoly.get_points()
                N = 20
                s1_samples = np.linspace(DOE[0, 0], DOE[-1, 0], N)
                s2_samples = np.linspace(DOE[0, 1], DOE[-1, 1], N)
                [S1, S2] = np.meshgrid(s1_samples, s2_samples)
                S1_vec = np.reshape(S1, (N * N, 1))
                S2_vec = np.reshape(S2, (N * N, 1))
                samples = np.hstack([S1_vec, S2_vec])
                PolyDiscreet = myPolyFit(samples)
                PolyDiscreet = np.reshape(PolyDiscreet, (N, N))

                fig = go.Figure(fig)
                fig.update_layout(layout, scene_camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)))
                fig.data = fig.data[0:2]
                fig.plotly_restyle({'x': S1, 'y': S2, 'z': PolyDiscreet}, 0)
                fig.plotly_restyle({'x': DOE[:, 0], 'y': DOE[:, 1], 'z': y_true.squeeze()}, 1)

                return fig
            else:
                raise PreventUpdate
        else:
            layout = {"xaxis": {"title": r'X1'}, "yaxis": {"title": r'X2'},
                      'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 60},
                      'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
            myPoly = jsonpickle.decode(ModelSet)
            y_true = jsonpickle.decode(true_vals)
            myPolyFit = myPoly.get_polyfit
            DOE = myPoly.get_points()

            fig = go.Figure(fig)
            fig.update_layout(layout)
            fig.plotly_restyle({'x': [[]], 'y': [[]], 'z': [[]]}, 0)
            fig.plotly_restyle({'x': [[]], 'y': [[]], 'z': [[]]}, 1)
            if len(fig.data) == 3:
                fig.plotly_restyle({'x': DOE[:,0], 'y': y_true.flatten()}, 2)
            else:
                fig.add_trace(go.Scatter(x=DOE[:,0], y=y_true.flatten(), mode='markers', name='Training samples',
                                        marker=dict(color='rgb(135,206,250)', size=15, opacity=0.5,
                                                    line=dict(color='rgb(0,0,0)', width=1))))

            return fig

    else:
        raise PreventUpdate
