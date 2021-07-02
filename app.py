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
import json
import jsonpickle
import ast
from equadratures import *
import numexpr as ne
import scipy as sp



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
}
LOWER_UP_UNI_DIST={
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
            dbc.Col([dcc.Input(id="input_func", type="text", placeholder="Input Function...",className='ip_field',debounce=True
                               ,style={'width': '150px'}),
            dbc.Tooltip(
                "The variables for input function should be of form x1,x2...",
                target="input_func",
                placement='right'
                    ),
           ],width=4),
            ]),

        html.Br(),
        dbc.Row(
            dbc.Col(
                html.Div(id='param_add', children=[])))

            ]
        )
    ]
    ,className='top_card',
    color="#FFFFFF",
    inverse=True,
    style={"width": "96%",
           'height': '380px',
           "left":"2rem",
           "top":"5rem",
           },
)



PDF_GRAPH= dbc.Card([
    dcc.Graph(id='plot_pdf', style={'width': '100%', 'display': 'inline-block', 'margin-top': '10px'})
], style={'display': 'inline','top':"5rem",'left':'2rem'})

BASIS_CARD=dbc.Card([
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
            className="m-1",id='drop_basis',
                        optionHeight=45,
                        style={
                            "width":"165px",

                        }
                    ),

    ],width=3),
    dbc.Col([
        dbc.Row([
    dbc.Col([
        dbc.Input(bs_size="sm", id='q_val', type="number", value=np.nan, placeholder='q',className='ip_field',
                  disabled=True,style={'width': '100px'}),
    ], width=3),
    dbc.Col([
        dbc.Input(bs_size="sm",id='levels',type='number',value=np.nan,placeholder='Level',className='ip_field',
                  disabled=True,style={'width':'100px'})
    ], width=3),
            ], no_gutters=True,
        justify="start")
],width=9)
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
            className="m-1",id='basis_growth_rule',
                        optionHeight=45,
                        disabled=True,
                        style={
                            "width":"800px",
                            "display":"flex"
                        }
                    ),

        ])
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([html.Button('Cardinality Check', id='CC_button', n_clicks=0, className='ip_buttons')]),]
            ),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Label('Cardinality'),width='auto'),
        dbc.Col([dbc.Input(bs_size="sm", id='op_box', type="number", value='', placeholder='', className='ip_field',
                  disabled=True,style={'width': '100px'})],width='auto'),
       dcc.Store(id='ParamObjects'),
       dcc.Store(id='PolyObject'),
       dbc.Col(dcc.Graph(id='plot_basis',style={'display':'inline-block','width': '500px','margin-top':'-80px' ,'height':'300px',
                                                  'margin-left':'20px'}),width=8),

    ],no_gutters=True,
    justify='start')
],style={"top": "5.5rem","margin-left":"0.5rem","width":"96%","height":"460px"})



COMPUTE_CARD=dbc.Card([
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
            className="m-1",id='solver_method',
                        optionHeight=45,
                        style={
                            "width":"200px",
                        }
                    ),
        ])
        ]),
dbc.Row([
    dbc.Col([
        html.Button(['Compute Uncertainty'], id='CU_button', n_clicks=0, className='ip_buttons')
    ]),
    dcc.Store(id='ModelSet'),
    dcc.Store(id='True_vals')

        ]),
            html.Br(),
            html.Br(),
dbc.Row([
    dbc.FormGroup([
        dbc.Row([
    dbc.Col([
        html.Label("MEAN")
    ]),
    dbc.Col([
        html.Label("VARIANCE")
    ]),
    dbc.Col([
        html.Label("R2 Score")
    ])
            ]),
        dbc.Row([
            dbc.Col([
                dbc.Input(bs_size="sm", id='mean', type='number', value=np.nan, placeholder='Mean...',
                          className='ip_field',
                          disabled=True, style={'width': '100px'})
            ]),
            dbc.Col([
                dbc.Input(bs_size="sm", id='variance', type='number', value=np.nan, placeholder='Variance..,',
                          className='ip_field',
                          disabled=True, style={'width': '100px'})
            ]),
            dbc.Col([
                dbc.Input(bs_size="sm", id='r2_score', type='number', value=np.nan, placeholder='R2 Score..,',
                          className='ip_field',
                          disabled=True, style={'width': '100px'})
            ])
        ])
        ])
]),
            html.Br(),
            dbc.Row([
                dbc.FormGroup([
                dbc.Row([
                    dbc.Col(html.Label('SENSITIVITY ANALYSIS'))
                ]),
                    html.Br(),
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
                        ]),]),

                dbc.Row([
                    dbc.Col([
                        html.Div(id='sobol_values')
                    ])
                ])

            ])
            ])
        ],width=5),
    dbc.Col([
        dcc.Graph(id = 'plot_poly_3D', style={'width':'600px'})
    ],width=6)
])
    ])
],style={"top": "5.5rem","margin-left":"2.8rem","width":"94.5%","height":"460px"})



navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src='https://equadratures.org/_static/logo_new.png', height="40px")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://equadratures.org/",
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Introduction", href="/Intro",active='exact',className='px-3')),
                    dbc.NavItem(dbc.NavLink("Analytical Model", href="/", active='exact',className='px-3')),
                    dbc.NavItem(dbc.NavLink("Offline Model", href="/offline",active='exact',className='px-3'))
                ], className="ml-auto", navbar=True, fill=True
            ),
        ], fluid=True
    ),
    color="white",
    dark=False,
    className="mb-0",
    fixed='top'
)


app.layout=html.Div([
navbar,
dbc.Row([
dbc.Col(TOP_CARD,width=12),
]),
dbc.Row([
    dbc.Col(PDF_GRAPH,width=5),
    dbc.Col(BASIS_CARD,width=7)
],

    no_gutters=False
),
dbc.Row([
    COMPUTE_CARD
])
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
                dbc.Row([html.Label("INPUT DISTRIBUTION")],style={"color":"#000000","font-size":"0.9rem","font-family":"Raleway"}),
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
                        placeholder='Select a distribution',
                        value='Uniform',
                        className="m-1",id={
                            'type':'drop-1',
                            'index':n_clicks
                        },
                        clearable=False,
                        optionHeight=20,
                        style={
                            "width":"150px",
                        }
                    )
                ]),
             ])],width=3,lg=2),
            dbc.Col([
                dbc.FormGroup([
                dbc.Row([html.Label('INPUT STATISTICAL MOMENTS')],style={"color":"#000000","font-size":"0.9rem","font-family":"Raleway"}),
                dbc.Row([
                    dbc.Col([
                            dbc.Input(bs_size="sm",id={'type':'params','index':n_clicks}, type="number",value=np.nan,placeholder='',
                                        debounce=True,className='ip_field',style={'width': '100px'}),
                        ],width=3),
                    dbc.Col([
                            dbc.Input(bs_size="sm",id={'type':'params_2','index':n_clicks}, type="number",value=np.nan,placeholder='...',
                                      debounce=True,className='ip_field',style={'width': '100px'}),
                        ],width=3),
                ]),
            ]),],lg=4, xs=3, width=3),
            dbc.Col([
                dbc.Row([html.Label('INPUT MIN/MAX/ORDER VALUE')],style={"color":"#000000","font-size":"0.9rem","font-family":'Raleway'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Input(bs_size="sm",id={'type':'min_val','index':n_clicks}, type="number", value=np.nan, placeholder='Minimum value...',
                                  debounce=True,className='ip_field',style={'width': '100px'}),
                    ],width='auto'),
                    dbc.Col([
                        dbc.Input(bs_size="sm",id={'type':'max_val','index':n_clicks},type="number",value=np.nan,placeholder="Maximum value...",
                                  debounce=True,className='ip_field',style={'width': '100px'})
                    ],width='auto'),
                    dbc.Col([
                        dbc.Input(bs_size="sm", id={'type': 'order', 'index': n_clicks}, type="number", value=np.nan,
                                  placeholder="Order",
                                  debounce=True, className='ip_field', style={'width': '100px'})
                    ],width='auto'),
                    dbc.Col([
                        dbc.Checklist(
                            options=[
                                {"label": " ", "value": "val_{}".format(n_clicks)},
                            ],
                            switch=True,
                            value=[0],
                            id={
                                "type": "radio_pdf",
                                "index": n_clicks
                            }
                        )
                    ],width='auto'),
                    # dbc.Col([
                    #     html.Label("X{}".format(n_clicks),style={'color':'black'})
                    # ],width=1)

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
    Output({'type': 'min_val', 'index': dash.dependencies.MATCH},'placeholder'),
    Output({'type': 'max_val', 'index': dash.dependencies.MATCH},'placeholder'),
    Output({'type': 'params', 'index': dash.dependencies.MATCH},'disabled'),
    Output({'type': 'params_2', 'index': dash.dependencies.MATCH},'disabled'),
    Output({'type': 'min_val', 'index': dash.dependencies.MATCH},'disabled'),
    Output({'type': 'max_val', 'index': dash.dependencies.MATCH},'disabled'),
    [Input({'type':'drop-1','index':dash.dependencies.MATCH},'value')],
    prevent_initial_callback=True,
)
def UpdateInputField(value):
    show=False
    hide=True
    if value is None:
        return ['Statistical Measures based on Distribution','...','...','...',hide,hide,hide]
    if value in MEAN_VAR_DIST.keys():
        return 'Mean...','Variance...',' ',' ',show,show,hide,hide
    if value in LOWER_UP_UNI_DIST.keys():
        return '','','Min Value...','Max Value...',hide,hide,show,show
    if value in SHAPE_PARAM_DIST.keys():
        return 'Shape...',' ','','',show,hide,hide,hide
    if value in LOWER_UPPER_DIST.keys():
        return 'Lower Value...','Upper Value...','Min Value...','Max Value...',show,show,show,show
    if value in LOW_UP_SHA_SHB.keys():
        return 'Shape A...','Shape B...','Min Value...','Max Value...',show,show,show,show

@app.callback(
    Output('ParamObjects','data'),
    [
        Input('AP_button','n_clicks'),
        Input({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'order', 'index': dash.dependencies.ALL}, 'value'),

    ],
    prevent_intial_call=True
)

def ParamListUpload(n_clicks,shape_parameter_A,shape_parameter_B,distribution,max,min,order):
    i=len(distribution)
    param_list=[]
    if i>0:
        for j in range(i):
            if distribution[j] in MEAN_VAR_DIST.keys():
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                     , shape_parameter_B=shape_parameter_B[j], lower=min[j], upper=max[j], order=order[j])
            elif distribution[j] in LOW_UP_SHA_SHB.keys():
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                 , shape_parameter_B=shape_parameter_B[j],lower=min[j], upper=max[j], order=order[j])
            elif distribution[j] in SHAPE_PARAM_DIST.keys():
                param = eq.Parameter(distribution=distribution[j],shape_parameter_A=shape_parameter_A[j],order=order[j])
            elif distribution[j] in LOWER_UP_UNI_DIST.keys():
                param= eq.Parameter(distribution=distribution[j],lower=min[j],upper=max[j],order=order[j])
            elif distribution[j] in LOWER_UPPER_DIST.keys():
                param=eq.Parameter(distribution=distribution[j],shape_parameter_A=shape_parameter_A[j],
                                   shape_parameter_B=shape_parameter_B[j],lower=min[j],upper=max[j],order=order[j])
            param_list.append(param)
    return jsonpickle.encode(param_list)


LOWER_UPPER_DIST={
    "Chebyshev":db.chebyshev
}


def CreateParam(distribution,shape_parameter_A,shape_parameter_B,min,max,order):
    param_obj=eq.Parameter(distribution=distribution,shape_parameter_A=shape_parameter_A,shape_parameter_B=shape_parameter_B,
                           lower=min,upper=max,order=order)
    s_values,pdf=param_obj.get_pdf()
    return param_obj,s_values,pdf

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
def PlotPdf(pdf_val,param1_val,params2_val,drop1_val,max_val,min_val,order):
    layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
              'autosize': True}
    fig=go.Figure(layout=layout)
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
        ),
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )
    ctx = dash.callback_context
    id = ctx.triggered[0]['prop_id'].split('.')[0]
    idx = ast.literal_eval(id)['index']

    elem = [0, 'val_{}'.format(idx)]
    check = elem in pdf_val
    if check:
        i = pdf_val.index(elem)
        if param1_val and params2_val is None:
            param,s_values,pdf=CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                                           shape_parameter_B=params2_val[i],
                                           min=min_val[i], max=max_val[i],order=order[i])

            fig.add_trace(go.Scatter(x=s_values,y=pdf,line = dict(color='rgb(0,176,246)'),fill='tonexty',mode='lines',name='NACA0012',line_width=4,line_color='black')),
        else:
            param, s_values, pdf = CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                                           shape_parameter_B=params2_val[i],min=min_val[i], max=max_val[i],order=order[i])

            fig.add_trace(go.Scatter(x=s_values, y=pdf, line=dict(color='rgb(0,176,246)'), fill='tonexty')),
    return fig

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

        test = [[0] if j != i else elem for j,x in enumerate(pdf_val)]
    return ret_vals

@app.callback(
    Output('q_val','disabled'),
    Output('levels','disabled'),
    Output('basis_growth_rule','disabled'),
    [Input('drop_basis','value')],
    prevent_initial_call=True
)
def BasisShow(value):
    show=False
    hide=True
    if value is not None:
        if value=='sparse-grid':
            return  hide,show,show
        elif value=='hyperbolic-basis':
            return  show,hide,hide
        else:
            return hide,hide,hide
    else:
        return hide,hide,hide

def Set_Basis(basis_val,order,level,q_val,growth_rule):

    basis_set=Basis('{}'.format(basis_val),orders=order,level=level,q=q_val,growth_rule=growth_rule)
    return basis_set

def Set_Polynomial(parameters,basis,method):
    myPoly=eq.Poly(parameters=parameters,basis=basis,method=method)
    return myPoly


@app.callback(
    [Output('op_box','value'),
    Output('PolyObject','data')],
    [
    Input('CC_button', 'n_clicks'),
    Input('ParamObjects','data')],
    [
     State('AP_button','n_clicks'),
     State('drop_basis','value'),
     State('q_val','value'),
     State('levels','value'),
     State('basis_growth_rule','value'),
     State('solver_method','value')
     ],
    prevent_initial_call=True
)
def OutputCardinality(n_clicks,param_obj,params_click,basis_select,q_val,levels,growth_rule,solver_method):
    if n_clicks!=0:
        param_data=jsonpickle.decode(param_obj)
        basis_ord=[]
        for elem in param_data:
            basis_ord.append(elem.order)
        mybasis=Set_Basis(basis_val=basis_select,order=basis_ord,level=levels,q_val=q_val,growth_rule=growth_rule)
        # print(mybasis.basis_type)
        # print(solver_method)
        myPoly=eq.Poly(parameters=param_data,basis=mybasis,method=solver_method)

        return [mybasis.get_cardinality(),jsonpickle.encode(myPoly)]
    else:
        raise PreventUpdate


@app.callback(
    Output('solver_method','value'),
    [Input('drop_basis','value')],
    prevent_initial_call=True
)
def SetMethod(drop_basis):
    if drop_basis=='total-order':
        return 'least-squares'
    else:
        return 'numerical-integration'





@app.callback(
    Output('plot_basis','figure'),
    [
    Input('PolyObject','data'),
    Input('AP_button','n_clicks'),
    ]
)
def PlotBasis(poly,n_clicks):
    if poly is not None:
        myPoly=jsonpickle.decode(poly)
        DOE=myPoly.get_points()
        if n_clicks==1:
            fig=px.scatter(DOE)
            return fig
        elif n_clicks==2:
            fig=px.scatter(x=DOE[:,0], y=DOE[:,1])
            return fig
        # fig = px.scatter(x=DOE[:,0], y=DOE[:,1])

    else:
        raise PreventUpdate


@app.callback(
    [Output('ModelSet','data'),
    Output('mean','value'),
    Output('variance','value'),
    # Output('sobol_values','children')
    Output('r2_score','value'),
    Output('True_vals','data')
     ],
    [
     Input('PolyObject','data'),
     Input('input_func','value'),
     Input('CU_button','n_clicks'),
     Input('AP_button','n_clicks'),
     Input('sobol_order','value')]
)
def SetModel(poly,expr,compute_button,n_clicks,sobol_order):
    if compute_button!=0:
        myPoly = jsonpickle.decode(poly)
        x = [r"x{} = op[{}]".format(j, j - 1) for j in range(1, n_clicks + 1)]
        def f(op):
            for i in range(n_clicks):
                exec(x[i])
            return ne.evaluate(expr)

        myPoly.set_model(f)
        values=myPoly.get_mean_and_variance()
        mean=values[0]
        variance=values[1]
        sobol_vals = myPoly.get_sobol_indices(order=sobol_order)
        print(myPoly)
        print(sobol_vals)
        DOE=myPoly.get_points()
        y_true = []
        for i in range(len(DOE)):
            y_true.append(f(DOE[i]))
        y_true = np.array(y_true)
        y_true=y_true.reshape(-1,1)
        y_pred=myPoly.get_polyfit(DOE).squeeze()
        y_pred=y_pred.reshape(-1,1)
        r2_score=eq.datasets.score(y_true,y_pred,metric='r2')
        return jsonpickle.encode(myPoly),mean,variance,r2_score,jsonpickle.encode(y_true)  ###
    else:
        raise PreventUpdate

@app.callback(
    Output('sobol_values','children'),
    [
    Input('sobol_order','value'),
    Input('ModelSet','data'),
    Input('CU_button','n_clicks')
    ],
    prevent_initial_call=True
)
def SobolTest(order,Model,n_clicks):
    if n_clicks>0 and order is not None:
        myPoly=jsonpickle.decode(Model)
        vals=myPoly.get_sobol_indices(order=order)

        print('vals',vals)
        print('order',type(order))
        return '{}'.format(vals)
    else:
        raise PreventUpdate

# @app.callback(
#     Output('sobol_values','children'),
#     [Input('sobol_order','value'),
#      Input('AP_button','n_clicks'),
#      State('sobol_values','children')]
# )
# def SobolIndices(sobol_order,n_clicks,children):
#     if sobol_order==1:
#         add_card=dbc.Row([
#             dbc.Col([
#                             dbc.Input(bs_size="sm",id={'type':'sobol','index':n_clicks}, type="number",value=np.nan,
#                                       placeholder='X{}'.format(n_clicks),
#                                         debounce=True,className='ip_field',style={'width': '100px'}),
#                         ],width=3)
#         ])
#         children.append(add_card)
#         return children
#     elif sobol_order==2:










@app.callback(
    Output('plot_poly_3D','figure'),
    [
    Input('ModelSet','data'),
    Input('CU_button','n_clicks'),
    Input('True_vals','data')
    ],
    prevent_initial_call=True
)
def Plot_poly_3D(ModelSet,n_clicks,true_vals):
    if n_clicks>0:
        layout = dict(margin={'t': 0, 'r': 0, 'l': 0, 'b': 0, 'pad': 10}, autosize=True,
                     scene=dict(
                         aspectmode='cube',
                         xaxis=dict(
                             title='Samples from DOE',
                             gridcolor="white",
                             showbackground=False,
                             linecolor='black',
                             tickcolor='black',
                             ticks='outside',
                             zerolinecolor="white", ),
                         yaxis=dict(
                             title='Samples from DOE',
                             gridcolor="white",
                             showbackground=False,
                             linecolor='black',
                             tickcolor='black',
                             ticks='outside',
                             zerolinecolor="white"),
                         zaxis=dict(
                             title='Polyfit',
                             backgroundcolor="rgb(230, 230,200)",
                             gridcolor="white",
                             showbackground=False,
                             linecolor='black',
                             tickcolor='black',
                             ticks='outside',
                             zerolinecolor="white", ),
                     ),
                     )
        myPoly=jsonpickle.decode(ModelSet)
        y_true=jsonpickle.decode(true_vals)
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
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Surface(x=S1, y=S2, z=PolyDiscreet,showscale=False,opacity=0.5,
                                 colorscale = [[0,'rgb(178,34,34)'],[1,'rgb(0,0,0)']]))

        fig.add_trace(go.Scatter3d(x=DOE[:,0],y=DOE[:,1],z=y_true.squeeze(),mode='markers',
                      marker=dict(size=10,color="rgb(144, 238, 144)",opacity=0.6,line=dict(color='rgb(0,0,0)',width=1))))

        return fig
    else:
        raise PreventUpdate




if __name__=="__main__":
    app.run_server(debug=True)