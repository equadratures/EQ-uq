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
    ,className='top_card',
    color="#FFFFFF",
    inverse=True,
    style={"width": "96%",
           'height': '370px',
           "left":"2rem",
           "top":"5rem",
           },
)



# TOP_TABLE=dbc.Container(
#     fig=go.Figure(data=[go.Table(header=dict(values=['Parameter','Distribution','Mean','Variance','Max','Min']),
#            TOP_CARD=dbc.Card(
#     [
#
# )

PDF_GRAPH= dbc.Card([
    dcc.Graph(id='plot_pdf', style={'width': '100%', 'display': 'inline-block', 'margin-top': '10px'})
], style={'display': 'inline','top':"5rem",'left':'2rem'})

BASIS_CARD=dbc.Card([
html.Br(),
dbc.Row([
    dbc.Col([
        dcc.Dropdown(
            options=[
                {'label': 'Univariate', 'value': 'Univariate'},
                {'label': 'Total-order', 'value': 'Total-order'},
                {'label': 'Tensor-grid', 'value': 'Tensor-grid'},
                {'label': 'Sparse-grid', 'value': 'Sparse-grid'},
                {'label': 'Hyperbolic-basis', 'value': 'Hyperbolic-basis'},
                {'label': 'Euclidean-degree', 'value': 'Euclidean-degree'}
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
        dbc.Input(bs_size="sm", id='basis_order', type="number", value='', placeholder='Order', className='ip_field',
                  style={'width': '100px'}),
    ], width=3),
    dbc.Col([
        dbc.Input(bs_size="sm", id='q_val', type="number", value='', placeholder='q',className='ip_field',
                  disabled=True,style={'width': '100px'}),
    ], width=3),
    dbc.Col([
        dbc.Input(bs_size="sm",id='levels',type='number',value='',placeholder='Level',className='ip_field',
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
                {'label': 'Linear', 'value': 'Linear'},
                {'label': 'Exponential', 'value': 'Exponential'},
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
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
            options=[
                {'label': 'Compressed-sensing', 'value': 'compressed-sensing'},
                {'label': 'Least-squares', 'value': 'least-squares'},
                {'label': 'Minimum-norm', 'value': 'minimum-norm'},
                {'label': 'Numerical-integration', 'value': 'numerical-integration'},
                {'label': 'Least-squares-with-gradients', 'value': 'least-squares-with-gradients'},
                {'label': 'Least-absolute-residual', 'value': 'least-absolute-residual'},
                {'label': 'Huber', 'value': 'huber'},
                {'label': 'Elastic-net', 'value': 'elastic-net'},
                {'label': 'Relevance-vector-machine', 'value': 'relevance-vector-machine'},
            ],
            placeholder='Solver Method',
            className="m-1",id='solver_method',
                        optionHeight=45,
                        style={
                            "width":"200px",
                        }
                    ),
        ])
        ]),
    html.Br(),
    dbc.Row([html.Button('Cardinality Check', id='CC_button', n_clicks=0, className='ip_buttons')],
            ),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Label('Cardinality'),width='auto'),
        dbc.Col(dbc.Input(bs_size="sm", id='op_box', type="number", value='', placeholder='', className='ip_field',
                  disabled=True,style={'width': '100px'}),width='auto'),
        dbc.Col(html.Div(children=[],id='test'),width='auto'),
    ],no_gutters=True,
    justify='start'),


],style={"top": "5.5rem","margin-left":"0.5rem","width":"96%","height":"460px"})






COMPUTE_CARD=dbc.Card([
    dbc.Button(['COMPUTE UNCERTAINTY'])
],style={"width":'50%','display':'inline-block'})




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
)
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
                dbc.Row([html.Label('INPUT STATISTICAL MOMENTS')],style={"color":"#000000","font-size":"0.9rem","font-family":"Raleway"}),
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
                dbc.Row([html.Label('INPUT MIN/MAX VALUE')],style={"color":"#000000","font-size":"0.9rem","font-family":'Raleway'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Input(bs_size="sm",id={'type':'max_val','index':n_clicks}, type="number", value='', placeholder='Maximum value...', className='ip_field',style={'width': '100px'}),
                    ]),
                    dbc.Col([
                        dbc.Input(bs_size="sm",id={'type':'min_val','index':n_clicks},type="number",value='',placeholder="Minimum value...", className='ip_field',style={'width': '100px'})
                    ]),
                    dbc.Col([
                        dbc.Checklist(
                            options=[
                                {"label": "Pdf_{}".format(n_clicks), "value": "val_{}".format(n_clicks)},
                            ],
                            switch=True,
                            value=[0],
                            id={
                                "type": "radio_pdf",
                                "index": n_clicks
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
        return ['Statistical Measures based on Distribution','...','...','...',hide,hide,hide]
    if value in MEAN_VAR_DIST.keys():
        return 'Mean...','Variance...',' ',' ',show,hide,hide
    if value in SHAPE_PARAM_DIST.keys():
        return 'Shape...',' ','','',hide,hide,hide
    if value in LOWER_UPPER_DIST.keys():
        return 'Lower Value...','Upper Value...','','',show,hide,hide
    if value in LOW_UP_SHA_SHB.keys():
        return 'Mean...','Variance...','Shape A...','Shape B...',show,show,show

@app.callback(
    Output('test','children'),
    [
        Input('AP_button','n_clicks'),
        Input({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'params_3', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'params_4', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
        Input('test','children')
    ],
    prevent_intial_call=True
)

def ParamListUpload(n_clicks,shape_parameter_A,shape_parameter_B,shape_A,shape_B,distribution,min,max,children):
    i=len(distribution)
    param_list=[]
    if i>0:
        for j in range(i):
            if j==0:

                param = eq.Parameter(distribution=distribution[0], shape_parameter_A=shape_parameter_A[0]
                                     , shape_parameter_B=shape_parameter_B[0], lower=min[0], upper=max[0], order=3)
            else:
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                 , shape_parameter_B=shape_parameter_B[j], lower=min[j], upper=max[j], order=3)
            param_list.append(param)

    return jsonpickle.encode(param_list)




def CreateParam(distribution,shape_parameter_A,shape_parameter_B,shape_A,shape_B,min,max):
    param_obj=eq.Parameter(distribution=distribution,shape_parameter_A=shape_parameter_A,shape_parameter_B=shape_parameter_B,
                           lower=min,upper=max,order=3)
    s_values,pdf=param_obj.get_pdf()
    return param_obj,s_values,pdf

@app.callback(
    Output('plot_pdf', 'figure'),
    Input({'type': 'radio_pdf', 'index': dash.dependencies.ALL}, 'value'),
    [State({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_3', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_4', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
     ],
    prevent_initial_call=True
)
def PlotPdf(pdf_val,param1_val,params2_val,params3_val,params4_val,drop1_val,max_val,min_val):
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
        if params4_val and params3_val is None:
            param,s_values,pdf=CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                                           shape_parameter_B=params2_val[i],shape_A=None, shape_B=None,
                                           min=min_val[i], max=max_val[i])

            fig.add_trace(go.Scatter(x=s_values,y=pdf,line = dict(color='rgb(0,176,246)'),fill='tonexty',mode='lines',name='NACA0012',line_width=4,line_color='black')),
        else:
            param, s_values, pdf = CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                                           shape_parameter_B=params2_val[i],
                                           shape_A=None, shape_B=None, min=min_val[i], max=max_val[i])

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
        if value=='Sparse-grid':
            return  hide,show,show
        elif value=='Hyperbolic-basis':
            return  show,hide,hide
        else:
            return hide,hide,hide
    else:
        return hide,hide,hide

def Set_Basis(basis_val):
    basis_set=Basis('{}'.format(basis_val))
    return basis_set
def Set_Polynomial(parameters,basis,method):
    polynomial=eq.Poly(parameters=parameters,basis=basis,method='{}'.format(method))
    return polynomial
def Cardinality(polynomial):
    return polynomial.basis.get_cardinality()

@app.callback(
    Output('op_box','value'),
    Input('CC_button', 'n_clicks'),
    [State({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_3', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'params_4', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
     State('AP_button','n_clicks'),
     State('drop_basis','value'),
     State('basis_order','value'),
     State('q_val','value'),
     State('levels','value'),
     State('solver_method','value')
     ],
    prevent_initial_call=True
)
def OutputCardinality(n_clicks,param_1,param_2,param_3,param_4,distribution,max_val,min_val,params_click,basis_select,basis_order,q_val,levels,solver_method):
    if n_clicks!='0':
        param_list=[]
        distribution_list=[]
        for i in range(params_click):
            param, s_values, pdf = CreateParam(distribution=distribution[i], shape_parameter_A=param_1[i],
                                                 shape_parameter_B=param_2[i],
                                                    shape_A=None, shape_B=None, min=min_val[i], max=max_val[i])
            param_list.append(param)
            distribution_list.append(distribution[i])
        mybasis=Set_Basis(basis_val=basis_select)
        mypoly=Set_Polynomial(parameters=param_list,basis=mybasis,method=solver_method)
        return mypoly.basis.get_cardinality()
    else:
        return 'Hi'


    # elem = [0, 'val_{}'.format(idx)]
    # check = elem in pdf_val
    # if check:
    #     i = pdf_val.index(elem)
    #     if params4_val and params3_val is None:
    #         param, s_values, pdf = CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
    #                                            shape_parameter_B=params2_val[i],
    #                                            shape_A=None, shape_B=None, min=min_val[i], max=max_val[i])


if __name__=="__main__":
    app.run_server(debug=True)