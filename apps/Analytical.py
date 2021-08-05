import io

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, State, Trigger
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import base64
import dash_table
from dash_table.Format import Format, Scheme, Trim

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


model_selection=dcc.Dropdown(
    options=[
        {'label': 'Analytical Model', 'value': 'analytical'},
        {'label': 'Offline Model', 'value': 'offline'},
        ],
    className="m-1", id='model_select',
    value='analytical', clearable=False)


###################################################################
# Collapsable more info card
###################################################################
info_text = r'''
This app uses Equadratures to compute unceratinty in the user-defined data. In this model user 
can define parameters, select basis function and create a polynomial.

#### Instructions
1. Click **add parameter** button in parameter definition card to add parameters. Choose the type of distribution and on basis of **selected distribution**, input the required fields.
2. To visualize the defined parameters **probability density function** press the toggle button to obtain the plot in Probability density function card
3. Select the **basis** type from basis selection card and input required fields based on the basis selected (For example sparse-grid requires q-val, order and growth as input)
4. Use **Set Basis** button to compute the **cardinality** and get insights regarding the basis function chosen in the basis selection card.
5. Set the solver method for Polynomial and enter the **input function** in parameter definition card for computing **statistical moments**, use sobol dropdown to gain insights regarding **sensitivity analysis**

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

DOE_download= html.Div([
    dbc.Button("Download DOE",id='download_button',style={'display':'None'},color='primary'),
    dcc.Download(id='download_DOE_data')
])

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
                    dbc.Button('Set basis', id='basis_button', n_clicks=0, className='ip_buttons',color='primary',disabled=False),
                width=2),
                dbc.Col(
                    dbc.Input(bs_size="sm", id='op_box', type="number", value='', placeholder='Cardinality...', className='ip_field',disabled=True), 
                width=3),
                dbc.Col(dbc.Alert(id='compute-warning',color='danger',is_open=False),width='auto'),
                dbc.Col(DOE_download,width=3)
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


Upload_region=html.Div([
    dcc.Upload(
        id='upload_data',
        children=html.Div([
            'Upload model evalutions ',
            html.A('Select Files',style={'color':'blue'},id='filename_append'),

        ]),
    ),
])
dataset_info = html.Div(
    [
    dbc.Button("View Dataset",color="primary",id="dataset-info-open",className="py-0",disabled=True),
    dbc.Modal(
        [
            dbc.ModalHeader(dcc.Markdown('',id='dataset_filename')),
            dbc.ModalBody([dash_table.DataTable(data=[],columns=[],id='upload_data_table',
                style_table={'overflowX': 'auto','overflowY':'auto','height':'35vh'},
                editable=True,fill_width=True,page_size=20)],id='dataset_data'),
            dbc.ModalFooter(dbc.Button("Close", id="dataset-info-close", className="py-0", color='primary')),
        ],
        id="dataset-info",
        scrollable=True,size='lg'
    ),
    ],
id='dataset-div'
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

sobol_plot = dcc.Graph(id='Sobol_plot', style={'width': 'inherit', 'height':'35vh'})

left_side = [
    dbc.Row([dbc.Col(method_dropdown,width=6),
            dbc.Col(
                dbc.Spinner([Upload_region],show_initially=False,color='primary')
            )
            ]),
    dbc.Row([dbc.Col(
        dbc.Button('Compute Polynomial', id='CU_button', n_clicks=0, className='ip_buttons',color='primary',disabled=True)
    ),
    dbc.Col(dataset_info,width='auto')
    ]),
    dbc.Row([dbc.Col(dbc.Alert(id='poly-warning',color='danger',is_open=False), width=3)]),
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
        dcc.Graph(id='plot_poly_3D', style={'width': 'inherit','height':'60vh'}, figure=polyfig3D),
        dbc.Alert(id='plot_poly_info',color='primary',is_open=False)
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
        # dbc.Tooltip('Set basis and Input Function first',target="CU_button"),
    ]
)

###################################################################
# Overal app layout
###################################################################

layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(model_selection,width=4)
        ]),
        html.H2("Uncertainty quantification of an analytical model",id='main_text'),
        dbc.Row(
            [
                dbc.Col(dcc.Markdown('Define an analytical model, and its uncertain input parameters. Then, use polynomial chaos to compute output uncertainties and sensitivities.',id='info_text'),width='auto'),
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
        dcc.Store(id='DOE'),
        dcc.Store(id='UploadedDF'),
        tooltips
    ], fluid=True
)

###################################################################
# Callback for disabling AP button after 5 clicks
###################################################################

@app.callback(
    Output('AP_button', 'disabled'),
    Input('AP_button', 'n_clicks'),
    Input('basis_button','n_clicks')
)
def check_param(n_clicks,ndims):
    if n_clicks > 4:
        return True
    else:
        return False

@app.callback(
    Output('input_func','style'),
    Output('upload_data','style'),
    Output('dataset-div','style'),
    Input('model_select','value')
)
def InputRequired(model):
    if model=='analytical':
        return None,{'display':'None'},{'display':'None'}
    else:
        style = {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
        return {'display':'None'},style,None





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
                              value=np.nan,min=0,
                              placeholder="Order",
                              debounce=True, className='ip_field')
        ]
    )

    toggle_form = dbc.Form(
        [
            dbc.Label('Plot PDF'),
            dbc.Checklist(
                options=[{"value": "val_{}".format(n_clicks),'disabled':True}],
                switch=True, value=[0], id={"type": "radio_pdf","index": n_clicks},
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




@app.callback(
    Output('main_text','children'),
    Output('info_text','children'),
    Input('model_select','value')
)
def MainText(model):
    if model=='analytical':
        return 'Uncertainty quantification of an analytical model','Define an analytical model, and its uncertain input parameters. Then, use polynomial chaos to compute output uncertainties and sensitivities.'
    else:
        return 'Uncertainty quantification of an offline model','Define an offline model, and its uncertain input parameters.Download the DOE points and then, use polynomial chaos to compute output uncertainties and sensitivities at your simulation results.'



###################################################################
# Callback for disabling Cardinality Check button
###################################################################
@app.callback(
    Output('basis_button','disabled'),
    [
    Input('AP_button','n_clicks'),
        ]
)
def CheckifAPClicked(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'AP_button' in changed_id:
        return False
    else:
        return True


@app.callback(
    Output('download_button','disabled'),
    Input('BasisObject','data'),
)
def ShowDownload(basis):
    if basis is not None:
        return False
    else:
        return True

@app.callback(
    Output('download_DOE_data','data'),
    Output('download_button','style'),
    Input('download_button','n_clicks'),
    Input('model_select','value'),
    Input('ParamsObject', 'data'),
    Input('BasisObject', 'data'),
    Input('solver_method', 'value'),
    prevent_initial_call = True
)
def DOEdownload(n_clicks,model,params,basis,method):
    if model=='offline':
        if basis is not None:
            mypoly = Set_Polynomial(params, basis, method)
            DOE = mypoly.get_points()
            DOE=pd.DataFrame(DOE)
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'download_button' in changed_id:
                return dcc.send_data_frame(DOE.to_csv, "DOE.csv"), None
            else:
                raise PreventUpdate
        else:
            return None,None
    if model=='analytical':
        return None,{'display':'None'}
    else:
        raise PreventUpdate

def ParseData(content,filename):
    content_type,content_string=content.split(',')
    try:
        if 'csv' in filename:
            decoded = base64.b64decode(content_string)
            df=np.genfromtxt(io.StringIO(decoded.decode('utf-8')),delimiter=',')
            data=[]
            for i in range(1,len(df)):
                data.append(df[i][-1])
            data=np.array(data).reshape(-1,1)
            return data
        elif 'npy' in filename:
            r = base64.b64decode(content_string)
            data=np.load(io.BytesIO(r)).reshape(-1,1)
            return data
    except Exception:
        return None

    else:
        raise PreventUpdate

@app.callback(
    ServersideOutput('UploadedDF','data'),
    Output('filename_append','children'),
    Output('dataset-info-open','disabled'),
    Input('model_select','value'),
    Input('upload_data','filename'),
    Input('upload_data','contents'),
    Input('DOE','data')
)
def ParsedData(model,filename,content,DOE):
    if model=='offline':
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'upload_data' in changed_id:
            df=ParseData(content,filename)
            children=[filename]
            if df.shape[0]==DOE.shape[0]:
                return df,children,False
            else:
                return None,'Error',True
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

@app.callback(
    Output("dataset-info", "is_open"),
    [Input("dataset-info-open", "n_clicks"), Input("dataset-info-close", "n_clicks")],
    [State("dataset-info", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('dataset_filename','children'),
    Output('upload_data_table','data'),
    Output('upload_data_table','columns'),
    Input('filename_append','children'),
    Input("dataset-info", "is_open"),
    Input("UploadedDF",'data'),
    Input("DOE",'data')
)
def DatasetInfo(filename,is_open,df,DOE):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'dataset-info' in changed_id:
        if is_open:
            data=[]
            vals=np.column_stack((df,DOE))
            for i in range(vals.shape[0]):
                val_dict = {}
                for j in range(vals.shape[1]):
                    if j==0:
                        val_dict['model_evaluations'] = vals[i][j]
                    else:
                        val_dict['DOE_{}'.format(j)] = vals[i][j]
                    if j==vals.shape[1]-1:
                        data.append(val_dict)
            print(data)
            columns = [
                {'name': i, 'id': i, 'deletable': False, 'type': 'numeric', 'format': Format(precision=4)}
                for i in data[0].keys()]
            return filename,data,columns
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate



###################################################################
# Callback for disabling Compute Uncertainty button
###################################################################
@app.callback(
    Output('CU_button','disabled'),
    [
        Input('basis_button','n_clicks'),
        Input('input_func','value'),
        Input('AP_button','n_clicks'),
        Input('model_select','value'),
        Input('UploadedDF','data')
    ],
)
def CheckifCCClickd(n_clicks,input_val,ap,model,uploaded):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'AP_button' in changed_id:
        return True
    else:
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'basis_button' or 'input_func' in changed_id:
            if model=='analytical':
                if n_clicks>0 and input_val is not None:
                    return False
                else:
                    return True
            else:
                if n_clicks>0 and uploaded is not None:
                    return False
                else:
                    return True
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


# @app.callback(
#     Output({'type':'radio_pdf','index': dash.dependencies.ALL},'disabled'),
#     Input('AP_button','n_clicks'),
#     prevent_intial_call=True
# )
# def Toggle(n_clicks):
#     changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
#     print(changed_id)
#     if 'basis_button' in changed_id:
#         return [{'disabled':False}]
#     else:
#         val={'disabled':True}
#         return [val]*n_clicks



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
        Input('basis_button','n_clicks'),
    ],
    prevent_intial_call=True
)
def ParamListUpload(shape_parameter_A, shape_parameter_B, distribution, max_val, min_val, order,basis_click):
    i = len(distribution)
    param_list = []
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'basis_button' in changed_id:
        if i > 0:
            for j in range(i):
                if distribution[j] in MEAN_VAR_DIST:
                    if (shape_parameter_A[j] and shape_parameter_B[j] and order[j]) is None:
                        return None,None
                    if order[j]<0:
                        return None,None
                    else:
                        param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                             , shape_parameter_B=shape_parameter_B[j], lower=min_val[j],
                                             upper=max_val[j],
                                             order=order[j])

                elif distribution[j] in ALL_4:
                    if (shape_parameter_A[j] and shape_parameter_B[j] and min_val[j] and max_val[j] and order[j]) is None:
                        return None,None
                    elif min_val[j]>max_val[j]:
                        return None,None
                    else:
                        param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j]
                                     , shape_parameter_B=shape_parameter_B[j], lower=min_val[j], upper=max_val[j],
                                     order=order[j])

                elif distribution[j] in SHAPE_PARAM_DIST:
                    if (shape_parameter_A[j] and order[j]) is None:
                        return None,None
                    else:
                        param = eq.Parameter(distribution=distribution[j], shape_parameter_A=shape_parameter_A[j],
                                     order=order[j])

                elif distribution[j] in LOWER_UPPER_DIST:
                    if (min_val[j] and max_val[j] and order[j]) is None:
                        return None,None
                    else:
                        param = eq.Parameter(distribution=distribution[j], lower=min_val[j], upper=max_val[j], order=order[j])

                param_list.append(param)
        return param_list,len(param_list)
    else:
        raise PreventUpdate




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
    Input('ParamsObject', 'data'),
    Input('basis_button','n_clicks'),
    State('drop_basis', 'value'),
    State('q_val', 'value'),
    State('levels', 'value'),
    State('basis_growth_rule', 'value'),
    prevent_initial_call=True
)
def SetBasis(param_obj,n_clicks,basis_select, q_val, levels, growth_rule):
    # Compute subspace (if button has just been pressed)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'basis_button' in changed_id:
        if param_obj is not None:
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
            return 'ERROR...',None,True,'Incorrect parameter values'
    else:
        raise PreventUpdate

###################################################################
# Plotting Function: To plot basis 1D/2D/3D
###################################################################
@app.callback(
    Output('plot_basis', 'figure'),
    ServersideOutput('DOE','data'),
    Input('ParamsObject', 'data'),
    Input('BasisObject', 'data'),
    Input('solver_method', 'value'),
    Input('ndims','data')
)
def PlotBasis(params, mybasis, method, ndims):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'ParamsObject' in changed_id:
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
                return fig,DOE
            elif ndims == 2:
                fig.add_trace(go.Scatter(x=DOE[:, 0], y=DOE[:, 1],mode='markers',marker=dict(size=8, color="rgb(144, 238, 144)", opacity=0.6,
                line=dict(color='rgb(0,0,0)', width=1))))
                return fig,DOE
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
                          title=r'$x_3$',
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
                marker=dict(size=8, color="rgb(144, 238, 144)", opacity=0.6, line=dict(color='rgb(0,0,0)', width=1))))
                return fig,DOE
            else:
                raise PreventUpdate

        else:
            raise PreventUpdate
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
    Output('poly-warning','is_open'),
    Output('poly-warning','children'),
    Trigger('CU_button', 'n_clicks'),
    Input('ParamsObject', 'data'),
    Input('BasisObject', 'data'),
    Input('solver_method', 'value'),
    Input('model_select','value'),
    Input('UploadedDF','data'),
    State('input_func', 'value'),
    State('ndims', 'data'),
    prevent_initial_call=True
)
def SetModel(params,mybasis,method,model,data,expr,ndims):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button' in changed_id:
        mypoly = Set_Polynomial(params, mybasis, method)
        if model=='analytical':
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
                return None,None,None,True,"Incorrect variable naming",True,False,None

        # Get mean and variance
            mean, var = mypoly.get_mean_and_variance()
            DOE = mypoly.get_points()

        # Get R2 score
            y_true = mypoly._model_evaluations
            y_pred = mypoly.get_polyfit(DOE).squeeze()
            y_pred = y_pred.reshape(-1, 1)
            r2_score = eq.datasets.score(y_true, y_pred, metric='r2')
            return mypoly, mean, var, r2_score, False,None,False,None ###
        else:
            try:
                mypoly.set_model(data)
            except KeyError:
                return None,None,None,None,False,True,"Incorrect Model evaluations"
            # except AssertionError:
            #     return None,None,None,True,None,False,True,"Incorrect Data uploaded"
            # except ValueError:
            #     return None, None, None, None, False, True, "Incorrect Model evaluations"
            # except IndexError:
            #     return None, None, None, None, False, True, "Incorrect Model evaluations"
            #

            mean, var = mypoly.get_mean_and_variance()
            DOE=mypoly.get_points()

            y_true=data.squeeze()
            y_pred = mypoly.get_polyfit(DOE).squeeze()
            y_pred = y_pred.reshape(-1, 1)
            r2_score = eq.datasets.score(y_true, y_pred, metric='r2')
            return mypoly, mean, var, r2_score, False, None,False,None ###
    else:
        raise PreventUpdate

###################################################################
# Callback to plot Sobol' indices 
###################################################################

@app.callback(
    Output('sobol_order','options'),
    Input('CU_button','n_clicks'),
    State('ndims','data'),
    State('sobol_order','options')
,
prevent_intial_call=True
)
def SobolCheck(n_clicks,ndims,options):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button' in changed_id:
        opt=options
        if ndims==1:
            return options
        elif ndims==2:
            opt[0]['disabled']=False
            opt[1]['disabled']=False
            opt[2]['disabled']=True
            return opt
        elif ndims>=3:
            opt[0]['disabled'] = False
            opt[1]['disabled'] = False
            opt[2]['disabled'] = False
            return opt
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate



@app.callback(
    Output('Sobol_plot', 'figure'),
    Output('sobol_order','disabled'),
    Output('Sobol_plot','style'),
    Input('PolyObject', 'data'),
    Input('sobol_order', 'value'),
    Trigger('CU_button', 'n_clicks'),
    Input('ndims','data'),
    Input('model_select','value'),
    State('Sobol_plot', 'figure'),
    prevent_initial_call=True

)
def Plot_Sobol(mypoly, order, ndims, model,fig):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button' in changed_id:
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
                    labels = [r'$S_%d$' % i for i in range(1,(ndims)+1)]
                    to_plot = [sobol_indices[(i,)] for i in range((ndims))]

                elif order==2:
                    fig.update_yaxes(title=r'$S_{ij}$')
                    labels = [r'$S_{%d%d}$' % (i, j) for i in range(1,int(ndims)+1) for j in range(i + 1, int(ndims)+1)]
                    to_plot = [sobol_indices[(i, j)] for i in range(int(ndims)) for j in range(i + 1, int(ndims))]

                elif order==3:
                    fig.update_yaxes(title=r'$S_{ijk}$')
                    labels = [r'$S_{%d%d%d}$' % (i, j, k) for i in range(1,int(ndims)+1) for j in range(i + 1, int(ndims)+1) for k in
                                  range(j + 1, int(ndims)+1)]
                    to_plot = [sobol_indices[(i, j, k)] for i in range(int(ndims)) for j in range(i + 1, int(ndims)) for k in
                                   range(j + 1, int(ndims))]

                # fig.update_xaxes(nticks=len(sobol_indices),tickvals=labels,tickangle=45)
                data=go.Bar(
                x=np.arange(len(sobol_indices)),
                y=to_plot,marker_color='LightSkyBlue',marker_line_width=2,marker_line_color='black')
                fig = go.Figure(layout=layout,data=data)
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=np.arange(len(sobol_indices)),
                        ticktext=labels
                    ),
                    # uniformtext_minsize=8, uniformtext_mode='hide',
                    xaxis_tickangle=-30
                )
        if model=='analytical':
            style={'width': 'inherit', 'height':'35vh'}
        else:
            style = {'width': 'inherit', 'height': '35vh'}
        return fig, disabled, style
    else:
        raise PreventUpdate

#
# @app.callback(
#     Output('sobol_order','options'),
#     Input('ndims','data'),
#     Trigger('CU_button','n_clicks'),
#     State('sobol_order','options'),
#     prevent_intial_call=True
# )
# def SobolDisplay(ndims,options):
#     option_list=['Order 1','Order 2','Order 3'],
#     if ndims==2:
#         labels=[o for o in option_list[:1]]
#         return labels
#     elif ndims>2:
#         labels=[o for o in option_list]
#         return labels
#     else:
#         raise PreventUpdate
#







###################################################################
# Plotting Function: Polyfit plot
###################################################################
@app.callback(
    Output('plot_poly_3D', 'figure'),
    Output('plot_poly_3D','style'),
    Output('plot_poly_info','is_open'),
    Output('plot_poly_info','children'),
    Input('PolyObject', 'data'),
    Trigger('CU_button', 'n_clicks'),
    Input('ndims','data'),
    State('plot_poly_3D', 'figure'),
    prevent_initial_call=True
)
def Plot_poly_3D(mypoly, ndims,fig):
    hide={'display':'None'}
    default={'width':'600px'}
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button' in changed_id:
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
                return fig,default,False,None
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

                return fig,default,False,None

            else:
                added_text='''
                The Polyfit Plot exists for only **1D** and **2D** Polynomials, as we move to higher dimensions, 
                visualisation of data becomes computationally expensive and hence, we stick to 2D or 3D plots
                '''
                added_text = dcc.Markdown(convert_latex(added_text), dangerously_allow_html=True,
                                         style={'text-align': 'justify'})
                return fig,hide,True,added_text
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate
