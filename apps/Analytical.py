import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, State, Trigger
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import numpy as np
import equadratures as eq
import equadratures.distributions as db
import ast
import numexpr as ne
from utils import convert_latex

from app import app

###################################################################
# Distributions Data (@To add more)
###################################################################

MEAN_VAR_DIST = ["gaussian"]
LOWER_UPPER_DIST = ["uniform"]
SHAPE_PARAM_DIST = ["lognormal"]
ALL_4 = ["beta", "truncated-gaussian"]

###################################################################
# Collapsable more info card
###################################################################
info_text = r'''
Instructions go here...
'''

info = html.Div(
    [
    dbc.Button("More Information",color="primary",id="data-info-open",className="py-0"),
    dbc.Modal(
        [
            dbc.ModalHeader(dcc.Markdown('**More Information**')),
            dbc.ModalBody(dcc.Markdown(convert_latex(info_text),dangerously_allow_html=True)),
            dbc.ModalFooter(dbc.Button("Close", id="data-info-close", className="py-0", color='primary')),
        ],
        id="data-info",
        scrollable=True,size='lg'
    ),
    ]
)

###################################################################
# Parameter Definition Card
###################################################################

TOP_CARD = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**Parameter Definition**",style={"color": "#000000"})),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button('Add Parameter', id='AP_button', 
                                n_clicks=0, color="primary", className="py-0"), 
                            width='auto'),
                        dbc.Col(
                            dbc.Spinner(html.Div(id='param_added'),color='primary'),
                            width=1),
                        dbc.Col(
                            dcc.Input(id="input_func", type="text", placeholder="Input Function...",
                                className='ip_field', debounce=True), 
                            width=3),
                        dbc.Col(dbc.Alert(id='input-warning',color='danger',is_open=False), width=3),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        html.Div(id='param_add', children=[])
                    )
                )
            ],
        className='top_card',
        )
    ],
    id='top_card',
)

###################################################################
# PDF Plot
###################################################################
PDF_PLOT = dcc.Graph(id='plot_pdf', style={'height':'50vh','width': 'inherit'})

PDF_GRAPH = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**Probability Density Function**")),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(PDF_PLOT,width=12))
            ]
        )
    ], style={'height':'60vh'}
)

###################################################################
# Card for setting basis, levels, q and compute cardinality
###################################################################
basis_dropdown = dcc.Dropdown(
    options=[
        # {'label': 'Univariate', 'value': 'univariate'},
        {'label': 'Total-order', 'value': 'total-order'},
        {'label': 'Tensor-grid', 'value': 'tensor-grid'},
        {'label': 'Sparse-grid', 'value': 'sparse-grid'},
        {'label': 'Hyperbolic-basis', 'value': 'hyperbolic-basis'},
        {'label': 'Euclidean-degree', 'value': 'euclidean-degree'}
    ],
    placeholder='Select Basis', className="m-1", id='drop_basis',
    value='tensor-grid', clearable=False,
)

growth_dropdown = dcc.Dropdown(
    options=[
        {'label': 'Linear', 'value': 'linear'},
        {'label': 'Exponential', 'value': 'exponential'},
    ],
    placeholder='Growth Rule', clearable=False,
    className="m-1", id='basis_growth_rule',
    disabled=True,
)

BASIS_CARD = dbc.Card(
    [
    dbc.CardHeader(dcc.Markdown("**Basis Selection**")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(basis_dropdown, width=3),
                    dbc.Col(
                        dbc.Input(bs_size="sm", id='q_val', type="number", value=np.nan, placeholder='q',
                        className='ip_field', disabled=True), 
                    width=2),
                    dbc.Col(
                        dbc.Input(bs_size="sm", id='levels', type='number', value=np.nan, placeholder='Level',
                        className='ip_field', disabled=True),
                    width=2),
                    dbc.Col(growth_dropdown,width=3)
                ], justify="start"
            ),
            dbc.Row(
                [
                dbc.Col(
                    dbc.Button('Set basis', id='basis_button', n_clicks=0, className='ip_buttons',color='primary',disabled=True),
                width=2),
                dbc.Col(
                    dbc.Input(bs_size="sm", id='op_box', type="number", value='', placeholder='Cardinality...', className='ip_field',disabled=True), 
                width=3),
                dbc.Col(dbc.Alert(id='compute-warning',color='danger',is_open=False),width='auto')
                ]
            ),
            dbc.Row(dbc.Col(
                dcc.Graph(id='plot_basis',style={'width': 'inherit', 'height': '40vh', 'margin-top':'5px'}),
            width=8),align='start',justify='center'),
        ]
    )
    ], style={"height": "60vh"}
)

###################################################################
# Results card
###################################################################
params = dict(
    gridcolor="white",
    showbackground=False,
    linecolor='black',
    tickcolor='black',
    ticks='outside',
    zerolinecolor="white"
)

xparams = params.copy()
xparams['title'] = 'x1'
yparams = params.copy()
yparams['title'] = 'x2'
zparams = params.copy()
zparams['title'] = 'f(x)'

layout = dict(margin={'t': 0, 'r': 0, 'l': 0, 'b': 0, 'pad': 10}, autosize=True,
    scene=dict(
        aspectmode='cube',
        xaxis=xparams,
        yaxis=yparams,
        zaxis=zparams,
    ),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
)

polyfig3D = go.Figure(layout=layout)
polyfig3D.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
polyfig3D.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')

polyfig3D.add_trace(go.Surface(x=[], y=[], z=[], showscale=False, opacity=0.5,
                    colorscale=[[0, 'rgb(178,34,34)'], [1, 'rgb(0,0,0)']]))

polyfig3D.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers',
                   marker=dict(size=10, color="rgb(144, 238, 144)", opacity=0.6,
                               line=dict(color='rgb(0,0,0)', width=1))))

method_dropdown = dcc.Dropdown(
    options=[
        {'label': 'Least-squares', 'value': 'least-squares'},
        {'label': 'Numerical-integration', 'value': 'numerical-integration'},
    ],
    placeholder='Solver method', clearable=False,
    value='numerical-integration',
    className="m-1", id='solver_method',
)


mean_form = dbc.FormGroup(
    [
        dbc.Label("Mean",html_for='mean'),
        dbc.Row(dbc.Col(
            dbc.Input(bs_size="sm", id='mean', type='number', value=np.nan, placeholder='Mean...',
                className='ip_field', disabled=True)
        ), style={'align':'center'})
    ]
)

var_form = dbc.FormGroup(
    [
        dbc.Label("Variance",html_for='variance'),
        dbc.Row(dbc.Col(
            dbc.Input(bs_size="sm", id='variance', type='number', value=np.nan,
                placeholder='Variance..,', className='ip_field', disabled=True)
        ), style={'align': 'center'})
    ]
)

r2_form = dbc.FormGroup(
    [
        dbc.Label("R2 score",html_for='r2_score'),
        dbc.Row(dbc.Col(
            dbc.Input(bs_size="sm", id='r2_score', type='number', value=np.nan,
                placeholder='R2 Score..,', className='ip_field', disabled=True)
        ), style={'align':'center'})
    ]
)

sobol_form = dbc.FormGroup(
    [
        dbc.Label("Senstivity Indices",html_for='sobol_order'),
        dbc.Row(dbc.Col(
            dcc.Dropdown(
                options=[
                    {'label': 'Order 1', 'value': 1},
                    {'label': 'Order 2', 'value': 2},
                    {'label': 'Order 3', 'value': 3},

                ],
                placeholder='Order 1', value=1,
                className="m-1", id='sobol_order',
                disabled=True, clearable=False,
            ),
        ))
    ]
)

sobol_plot = dcc.Graph(id='Sobol_plot', style={'width': 'inherit', 'height':'40vh'})

left_side = [
    dbc.Row(dbc.Col(method_dropdown,width=6)),
    dbc.Row(dbc.Col(
        dbc.Button('Compute Polynomial', id='CU_button', n_clicks=0, className='ip_buttons',color='primary',disabled=True)
    )),
    dbc.Row(
        [
            dbc.Col(mean_form),
            dbc.Col(var_form),
            dbc.Col(r2_form),
        ]
    ),
    dbc.Row(dbc.Col(sobol_form,width=6)),
    dbc.Row(dbc.Col(sobol_plot,width=8))
]

right_side = dbc.Spinner(
    [
        dcc.Graph(id='plot_poly_3D', style={'width': 'inherit','height':'60vh'}, figure=polyfig3D)
    ], color='primary',type='grow',show_initially=False
)

COMPUTE_CARD = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**Compute Polynomial**")),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(left_side,width=6),
                        dbc.Col(right_side,width=6)
                    ]
                )
            ]
        )
    ], style={"height": "80vh"}
)

tooltips = html.Div(
    [
        dbc.Tooltip("Maximum of 5 parameters",target="AP_button"),
        dbc.Tooltip("The variables should be of the form x1,x2...",target="input_func"),
        dbc.Tooltip('Set basis and Input Function first',target="CU_button"),
    ]
)

###################################################################
# Overal app layout
###################################################################

layout = dbc.Container(
    [
        html.H2("Uncertainty quantification of an analytical model"),
        dbc.Row(
            [
                dbc.Col(dcc.Markdown('Define an analytical model, and its uncertain input parameters. Then, use polynomial chaos to compute output uncertainties and sensitivities.'),width='auto'),
                dbc.Col(info,width='auto')
            ], align='center', style={'margin-bottom':'10px'}
        ),
        dbc.Row(dbc.Col(TOP_CARD, width=12),style={'margin-bottom':'10px'}),
        dbc.Row(
            [
                dbc.Col(PDF_GRAPH, width=5),
                dbc.Col(BASIS_CARD, width=7)
            ],
        ),
        dbc.Row(dbc.Col(COMPUTE_CARD,width=12),
            style={'margin-top':'10px'}),

        # Small bits of data (store clientside)
        dcc.Store(id='ndims'),
    
        # Big bits of data (store serverside)
        dcc.Store(id='ParamsObject'),
        dcc.Store(id='PolyObject'),
        dcc.Store(id='BasisObject'),
        tooltips
    ], fluid=True
)

###################################################################
# Callback for disabling AP button after 5 clicks
###################################################################

@app.callback(
    Output('AP_button', 'disabled'),
    [Input('AP_button', 'n_clicks'),
     Input('basis_button','n_clicks')]
)
def check_param(n_clicks,cn_clicks):
    if n_clicks > 4 or cn_clicks>0:
        return True
    else:
        return False

###################################################################
# Callback for adding parameter to param definition card
###################################################################


@app.callback(
    Output('param_add', 'children'),
    Output('param_added','children'),
    [Input('AP_button', 'n_clicks'),
     State('param_add', 'children')]
)
def addInputs(n_clicks, children):
    dist_dropdown = dcc.Dropdown(
        options=[
            {'label': 'Uniform', 'value': 'uniform'},
            {'label': 'Gaussian', 'value': 'gaussian'},
            {'label': 'Truncated Gaussian', 'value': 'truncated-gaussian'},
#            {'label': 'Chi', 'value': 'chi'},
#            {'label': 'Cauchy', 'value': 'cauchy'},
            {'label': 'LogNormal', 'value': 'lognormal'},
            {'label': 'Beta', 'value': 'beta'}
        ],
        placeholder='Select a distribution', value='uniform', clearable=False,
        className="m-1", id={'type': 'drop-1', 'index': n_clicks},
    )

    dist_form = dbc.Form(
        [
            dbc.Label('Distribution', html_for='drop-1'),
            dist_dropdown,   
        ]
    ) 

    params_form = dbc.Form(
        [
            dbc.Label('Statistical moments/shape parameters',html_for='params'),
            dbc.Row(
                [
                dbc.Col(
                    dbc.Input(bs_size="sm", id={'type': 'params', 'index': n_clicks}, type="number",
                              value=np.nan, placeholder='',
                              debounce=True, className='ip_field'),
                    width=6),
                dbc.Col(
                    dbc.Input(bs_size="sm", id={'type': 'params_2', 'index': n_clicks}, type="number",
                              value=np.nan, placeholder='',
                              debounce=True, className='ip_field'),
                    width=6),
                ], justify='start', align='start'
            )
        ]    
    )

    min_max_form = dbc.Form(
        [
            dbc.Label('Support', html_for='min_val'),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Input(bs_size="sm", id={'type': 'min_val', 'index': n_clicks}, type="number",
                                  value=np.nan, placeholder='Minimum value...',
                                  debounce=True, className='ip_field'),
                    ], width=6),
                    dbc.Col([
                        dbc.Input(bs_size="sm", id={'type': 'max_val', 'index': n_clicks}, type="number",
                                  value=np.nan, placeholder="Maximum value...",
                                  debounce=True, className='ip_field')
                    ], width=6),
                ], justify='start', align='start'
            )
        ]
    )

    order_form = dbc.Form(
        [
            dbc.Label('Order'),
            dbc.Input(bs_size="sm", id={'type': 'order', 'index': n_clicks}, type="number",
                              value=np.nan,
                              placeholder="Order",
                              debounce=True, className='ip_field')
        ]
    )

    toggle_form = dbc.Form(
        [
            dbc.Label('Plot PDF'),
            dbc.Checklist(
                options=[{"value": "val_{}".format(n_clicks)}],
                switch=True, value=[0], id={"type": "radio_pdf","index": n_clicks}
            )
        ]
    )

    # Assemble layout
    if n_clicks > 0:
        add_card = dbc.Row(
            [
                dbc.Col(dcc.Markdown(convert_latex(r'$x_%d$' %n_clicks),dangerously_allow_html=True), width=1),
                dbc.Col(dist_form, width=2),
                dbc.Col(params_form, width=3),
                dbc.Col(min_max_form, width=3),
                dbc.Col(order_form, width=1),
                dbc.Col(toggle_form,width=1)
            ], align='start'
        )
    else:
        add_card = dbc.Row()
    children.append(add_card)

    return children,None

###################################################################
# Callback for disabling Cardinality Check button
###################################################################
@app.callback(
    Output('basis_button','disabled'),
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
        Input('basis_button','n_clicks'),
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
        return ['...', '...', '...', '...', hide, hide, hide]
    elif value in MEAN_VAR_DIST:
        return 'Mean...', 'Variance...', ' ', ' ', show, show, hide, hide
    elif value in LOWER_UPPER_DIST:
        return '', '', 'Lower bound...', 'Upper bound...', hide, hide, show, show
    elif value in SHAPE_PARAM_DIST:
        return 'Shape parameter...', ' ', '', '', show, hide, hide, hide
    elif value in ALL_4:
        return 'Shape param. A...', 'Shape param. B...', 'Lower bound...', 'Upper bound...', show, show, show, show


###################################################################
# Callback to create EQ Param Objects
###################################################################
@app.callback(
    ServersideOutput('ParamsObject', 'data'),
    Output('ndims','data'),
    [
        Input({'type': 'params', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'params_2', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'drop-1', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'max_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'min_val', 'index': dash.dependencies.ALL}, 'value'),
        Input({'type': 'order', 'index': dash.dependencies.ALL}, 'value'),
    ],
    prevent_intial_call=True
)
def ParamListUpload(shape_parameter_A, shape_parameter_B, distribution, max_val, min_val, order):
    i = len(distribution)
    param_list = []
    Show=False
    Block=True
    if i > 0:
        for j in range(i):
            if distribution[j] in MEAN_VAR_DIST:
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                     , shape_parameter_B=shape_parameter_B[j], lower=min_val[j], upper=max_val[j],
                                     order=order[j])

            elif distribution[j] in ALL_4:
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                     , shape_parameter_B=shape_parameter_B[j], lower=min_val[j], upper=max_val[j],
                                     order=order[j])


            elif distribution[j] in SHAPE_PARAM_DIST:
                param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j],
                                     order=order[j])

            elif distribution[j] in LOWER_UPPER_DIST:
                param = eq.Parameter(distribution=distribution[j], lower=min_val[j], upper=max_val[j], order=order[j])

            param_list.append(param)
    return param_list,len(param_list)


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
# Misc Callbacks
###################################################################
# More info collapsable
@app.callback(
    Output("data-info", "is_open"),
    [Input("data-info-open", "n_clicks"), Input("data-info-close", "n_clicks")],
    [State("data-info", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

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
              'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True,
              "xaxis":{"title": r'$x$'}, "yaxis": {"title": 'PDF'}}


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
                    shape_parameter_B=params2_val[i], min=min_val[i], max=max_val[i], order=order[i])

            fig.add_trace(go.Scatter(x=s_values, y=pdf, line=dict(color='rgb(0,176,246)'), fill='tonexty', mode='lines',
                name='Polyfit', line_width=4, line_color='black')),
        else:
            param, s_values, pdf = CreateParam(distribution=drop1_val[i], shape_parameter_A=param1_val[i],
                    shape_parameter_B=params2_val[i], min=min_val[i], max=max_val[i],order=order[i])
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
    basis_set = eq.Basis('{}'.format(basis_val), orders=order, level=level, q=q_val, growth_rule=growth_rule)
    return basis_set


def Set_Polynomial(parameters, basis, method):
    mypoly = eq.Poly(parameters=parameters, basis=basis, method=method)
    return mypoly

###################################################################
# Callback for automatic selection of solver method based on basis selection
###################################################################
@app.callback(
    Output('solver_method', 'value'),
    Input('drop_basis', 'value'),
    prevent_initial_call=True
)
def SetMethod(drop_basis):
    if drop_basis == 'total-order':
        return 'least-squares'
    else:
        return 'numerical-integration'

###################################################################
# Callback for setting basis
###################################################################
@app.callback(
    Output('op_box', 'value'),
    ServersideOutput('BasisObject', 'data'),
    Output('compute-warning','is_open'),
    Output('compute-warning','children'),
    Trigger('basis_button', 'n_clicks'),
    Input('ParamsObject', 'data'),
    State('drop_basis', 'value'),
    State('q_val', 'value'),
    State('levels', 'value'),
    State('basis_growth_rule', 'value'),
    prevent_initial_call=True
)
def SetBasis(param_obj, basis_select, q_val, levels, growth_rule):
    # Compute subspace (if button has just been pressed)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'basis_button' in changed_id:
        if basis_select is None:
            return 'Error...',None,True,'No basis value selected'
        elif basis_select=='sparse-grid' and (levels or growth_rule) is None:
                return 'ERROR...',None,True,'Enter the required values'
        else:
            basis_ord=[]
            for elem in param_obj:
                basis_ord.append(elem.order)
            mybasis = Set_Basis(basis_val=basis_select, order=basis_ord, level=levels, q_val=q_val, growth_rule=growth_rule)

            return mybasis.get_cardinality(), mybasis, False, None
    else:
        raise PreventUpdate

###################################################################
# Plotting Function: To plot basis 1D/2D/3D
###################################################################
@app.callback(
    Output('plot_basis', 'figure'),
    Input('ParamsObject', 'data'),
    Input('BasisObject', 'data'),
    Input('solver_method', 'value'),
    Input('ndims','data')
)
def PlotBasis(params, mybasis, method, ndims):
    if mybasis is not None:
        # Fit a poly just to get points (this isn't used elsewhere)
        mypoly = Set_Polynomial(params, mybasis, method)
        DOE = mypoly.get_points()
        layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                   'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True,
                  "xaxis":{"title": r'$x_1$'}, "yaxis": {"title": r'$x_2$'}}

        fig = go.Figure(layout=layout)
        fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
        fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
        if ndims == 1:
            fig.add_trace(go.Scatter(x=DOE[:,0], y=np.zeros_like(DOE[:,0]), mode='markers',marker=dict(size=8, color="rgb(144, 238, 144)", opacity=1,
                                               line=dict(color='rgb(0,0,0)', width=1))))
            fig.update_yaxes(visible=False)
            return fig
        elif ndims == 2:
            fig.add_trace(go.Scatter(x=DOE[:, 0], y=DOE[:, 1],mode='markers',marker=dict(size=8, color="rgb(144, 238, 144)", opacity=0.6,
                                               line=dict(color='rgb(0,0,0)', width=1))))
            return fig
        elif ndims>=3:
            fig.update_layout(dict(margin={'t': 0, 'r': 0, 'l': 0, 'b': 0, 'pad': 10}, autosize=True,
                      scene=dict(
                          aspectmode='cube',
                          xaxis=dict(
                              title=r'$x_1$',
                              gridcolor="white",
                              showbackground=False,
                              linecolor='black',
                              tickcolor='black',
                              ticks='outside',
                              zerolinecolor="white", ),
                          yaxis=dict(
                              title=r'$x_2$',
                              gridcolor="white",
                              showbackground=False,
                              linecolor='black',
                              tickcolor='black',
                              ticks='outside',
                              zerolinecolor="white"),
                          zaxis=dict(
                              title=r'x_3',
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
                                       marker=dict(size=8, color="rgb(144, 238, 144)", opacity=0.6,
                                                   line=dict(color='rgb(0,0,0)', width=1))))
            return fig

    else:
        raise PreventUpdate

###################################################################
# Callback to set Poly object, calculate mean, variance, r2_score
###################################################################
@app.callback(
    ServersideOutput('PolyObject', 'data'),
    Output('mean', 'value'),
    Output('variance', 'value'),
    Output('r2_score', 'value'),
    Output('input-warning','is_open'),
    Output('input-warning','children'),
    Trigger('CU_button', 'n_clicks'),
    Input('ParamsObject', 'data'),
    Input('BasisObject', 'data'),
    Input('solver_method', 'value'),
    State('input_func', 'value'),
    State('ndims', 'data'),
    prevent_initial_call=True
)
def SetModel(params,mybasis,method,expr,ndims):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button' in changed_id:
        # Create poly 
        mypoly = Set_Polynomial(params, mybasis, method)

        # Parse input function
        x = [r"x{} = op[{}]".format(j, j - 1) for j in range(1, ndims + 1)]
        def f(op):
            for i in range(ndims):
                exec(x[i])
            return ne.evaluate(expr)

        # Evaluate model
        try:
            mypoly.set_model(f)
        except KeyError or ValueError:
            return None,None,None,None,True,"Incorrect variable naming",True

        # Get mean and variance
        mean, var = mypoly.get_mean_and_variance()
        DOE = mypoly.get_points()

        # Get R2 score
        y_true = mypoly._model_evaluations
        y_pred = mypoly.get_polyfit(DOE).squeeze()
        y_pred = y_pred.reshape(-1, 1)
        r2_score = eq.datasets.score(y_true, y_pred, metric='r2')

        return mypoly, mean, var, r2_score, False,None ###
    else:
        raise PreventUpdate

###################################################################
# Callback to plot Sobol' indices 
###################################################################
@app.callback(
    Output('Sobol_plot', 'figure'),
    Output('sobol_order','disabled'),
    Input('PolyObject', 'data'),
    Input('sobol_order', 'value'),
    Trigger('CU_button', 'n_clicks'),
    Input('ndims','data'),
    State('Sobol_plot', 'figure'),
    prevent_initial_call=True

)
def Plot_Sobol(mypoly, order, ndims, fig):
    layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                          'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
    fig=go.Figure(layout=layout)
    fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
    fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
    if ndims == 1:
        disabled = True
    else:
        disabled = False
    
        if mypoly is not None:
            sobol_indices=mypoly.get_sobol_indices(order=order)
            layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                              'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
            fig=go.Figure(layout=layout)
            if order==1:
                fig.update_yaxes(title=r'$S_{i}$')
                labels = [r'$X_%d$' % i for i in range((ndims))]
                to_plot = [sobol_indices[(i,)] for i in range((ndims))]
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

    return fig, disabled

###################################################################
# Plotting Function: Polyfit plot
###################################################################
@app.callback(
    Output('plot_poly_3D', 'figure'),
    Output('plot_poly_3D','style'),
    Input('PolyObject', 'data'),
    Trigger('CU_button', 'n_clicks'),
    Input('ndims','data'),
    State('plot_poly_3D', 'figure'),
    prevent_initial_call=True
)
def Plot_poly_3D(mypoly, ndims,fig):
    hide={'display':'None'}
    default={'width':'600px'}
    if (mypoly is not None):
        y_true = mypoly._model_evaluations.ravel()
        if ndims==2:
            DOE = mypoly.get_points()
            N = 20
            s1_samples = np.linspace(DOE[0, 0], DOE[-1, 0], N)
            s2_samples = np.linspace(DOE[0, 1], DOE[-1, 1], N)
            [S1, S2] = np.meshgrid(s1_samples, s2_samples)
            S1_vec = np.reshape(S1, (N * N, 1))
            S2_vec = np.reshape(S2, (N * N, 1))
            samples = np.hstack([S1_vec, S2_vec])
            PolyDiscreet = mypoly.get_polyfit(samples)
            PolyDiscreet = np.reshape(PolyDiscreet, (N, N))

            fig = go.Figure(fig)
            fig.data = fig.data[0:2]
            fig.plotly_restyle({'x': S1, 'y': S2, 'z': PolyDiscreet}, 0)
            fig.plotly_restyle({'x': DOE[:, 0], 'y': DOE[:, 1], 'z': y_true}, 1)
            return fig,default
        elif ndims==1:
            layout = {"xaxis": {"title": r'$x_1$'}, "yaxis": {"title": r'$f(x_1)$'},
                      'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 60},
                      'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
            DOE = mypoly.get_points()
            N = 20
            s1_samples = np.linspace(DOE[0, 0], DOE[-1, -1], N)
            [S1] = np.meshgrid(s1_samples)
            S1_vec = np.reshape(S1, (N , 1))
            samples = np.hstack([S1_vec])
            PolyDiscreet = mypoly.get_polyfit(samples)
            PolyDiscreet = np.reshape(PolyDiscreet, (N))
            fig = go.Figure(fig)
            fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
            fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
            fig.update_layout(layout)

            fig.plotly_restyle({'x': [[]], 'y': [[]], 'z': [[]]}, 0)
            fig.plotly_restyle({'x': [[]], 'y': [[]], 'z': [[]]}, 1)
            if len(fig.data) == 4:
                fig.plotly_restyle({'x': DOE[:,0], 'y': y_true}, 2)
                fig.plotly_restyle({'x': S1      , 'y': PolyDiscreet}, 3)
            else:
                fig.add_trace(go.Scatter(x=DOE[:,0], y=y_true, mode='markers', name='Training samples',
                                        marker=dict(color='rgb(135,206,250)', size=15, opacity=0.5,
                                                    line=dict(color='rgb(0,0,0)', width=1))))
                fig.add_trace(go.Scatter(x=S1,y=PolyDiscreet,mode='lines',name='Polynomial approx.',line_color='rgb(178,34,34)'))

            return fig,default
        else:
            return {},hide

    else:
        return fig,hide
