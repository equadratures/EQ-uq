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
from navbar import navbar


import numpy as np
import equadratures as eq
import equadratures.distributions as db
import ast
import numexpr as ne
from utils import convert_latex

from app import app

EAN_VAR_DIST = ["gaussian"]
LOWER_UPPER_DIST = ["uniform"]
SHAPE_PARAM_DIST = ["lognormal"]
ALL_4 = ["beta", "truncated-gaussian"]

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
        dbc.Button("More Information", color="primary", id="datadriven-info-open", className="py-0"),
        dbc.Modal(
            [
                dbc.ModalHeader(dcc.Markdown('**More Information**')),
                dbc.ModalBody(dcc.Markdown(convert_latex(info_text), dangerously_allow_html=True)),
                dbc.ModalFooter(dbc.Button("Close", id="datadriven-info-close", className="py-0", color='primary')),
            ],
            id="datadriven-info",
            scrollable=True, size='lg'
        ),
    ]
)

###################################################################
# Parameter Definition Card
###################################################################


Upload_dataset=html.Div([
                    dcc.Upload(
                                id='upload-data-driven',
                                children=html.Div([
                                    'Upload Data  ',
                                    html.A('Select Files', style={'color': 'blue'}, id='filename_append'),
                                ]),
                                style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'}
                            ),
])

data_inputs = dbc.Row(
    [
    dbc.Col(
        dbc.FormGroup(
            [
                dbc.Label('Select output variable', html_for="output-select"),
                dcc.Dropdown(id="output-select",searchable=False)
            ],
        ),width=4
    ),
    dbc.Col(
        dbc.Form(
            [
            dbc.FormGroup(
                [
                dbc.Label('Number of input dimensions:', html_for="num-inputs",width=9),
                dbc.Col(html.Div(id='num-input'),width=3)
                ], row=True
            ),
            dbc.FormGroup(
                [
                dbc.Label('Number of output dimensions:', html_for="num-output",width=9),
                dbc.Col(html.Div(id='num-output'),width=3)
                ], row=True
            ),
            ]
        ), width=4
    ),
    dbc.Col(
        dbc.Form(
            [
            dbc.FormGroup(
                [
                dbc.Label('Parameter Definition:', html_for="mode-select",width=6),
                dbc.Col(dcc.Dropdown(id="mode-select",options=[
                    {'label': 'Manual', 'value': 'manual'},
                    {'label': 'Automatic', 'value':'auto'},
                ],searchable=False),width=4)
                ], row=True
            ),
            dbc.FormGroup(
                [
                    dbc.Label('Parameter Distribution:', html_for="distribution-select", width=6),
                    dbc.Col(dcc.Dropdown(id="distribution-select",options=[
            {'label': 'Uniform', 'value': 'uniform'},
            {'label': 'Gaussian', 'value': 'gaussian'},
            {'label': 'Truncated Gaussian', 'value': 'truncated-gaussian'},
            {'label': 'LogNormal', 'value': 'lognormal'},
            {'label': 'Beta', 'value': 'beta'}
        ], searchable=False,disabled=True),width=4)
                ], row=True
            ),
            ]
        ), width=4
    ),
    ]
)


TOP_CARD = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**Upload Data**", style={"color": "#000000"})),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                options=[
                                    {'label': 'Upload Dataset', 'value': 'Upload_data'},
                                    {'label': 'Dataset 1', 'value': 'Ds1'},
                                ],
                                className="m-1", id='dataset_selection',
                                placeholder='Select Dataset..', clearable=False),
                            width=3
                        ),
                        dbc.Col(Upload_dataset,width=5),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                            dbc.Alert('Click bin icons to delete columns as necessary', id='upload-help', color='info',
                                      is_open=False, dismissable=True, style={'margin-top': '0.4rem'}),
                            width=5)),

                dbc.Row(
                    dbc.Col(
                            dash_table.DataTable(data=[], columns=[], id='upload-data-table',
                                                 style_table={'overflowX': 'auto', 'overflowY': 'auto',
                                                              'height': '35vh','width':'auto'},
                                                 editable=True, fill_width=True, page_size=20)
                            , width=12), style={'margin-top': '10px'}),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(data_inputs, width=12),
                    ]
                )

            ],
            className='top_card',
        )
    ],
    id='data_top_card',
)


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
    ],
    placeholder='Solver method', clearable=False,
    value='numerical-integration',
    className="m-1", id='solver_method_data',
)

mean_form = dbc.FormGroup(
    [
        dbc.Label("Mean",html_for='mean'),
        dbc.Row(dbc.Col(
            dbc.Input(bs_size="sm", id='mean_datadriven', type='number', value=np.nan, placeholder='Mean...',
                className='ip_field', disabled=True)
        ), style={'align':'center'})
    ]
)

var_form = dbc.FormGroup(
    [
        dbc.Label("Variance",html_for='variance'),
        dbc.Row(dbc.Col(
            dbc.Input(bs_size="sm", id='variance_datadriven', type='number', value=np.nan,
                placeholder='Variance..,', className='ip_field', disabled=True)
        ), style={'align': 'center'})
    ]
)

r2_form = dbc.FormGroup(
    [
        dbc.Label("R2 score",html_for='r2_score'),
        dbc.Row(dbc.Col(
            dbc.Input(bs_size="sm", id='r2_score_datadriven', type='number', value=np.nan,
                placeholder='R2 Score..,', className='ip_field', disabled=True)
        ), style={'align':'center'})
    ]
)

sobol_form = dbc.FormGroup(
    [
        dbc.Label("Senstivity Indices",html_for='sobol_order_datadriven'),
        dbc.Row(dbc.Col(
            dcc.Dropdown(
                options=[
                    {'label': 'Order 1', 'value': 1},
                    {'label': 'Order 2', 'value': 2},
                    {'label': 'Order 3', 'value': 3},

                ],
                placeholder='Order 1', value=1,
                className="m-1", id='sobol_order_datadriven',
                disabled=True, clearable=False,
            ),
        ))
    ]
)

sobol_plot = dcc.Graph(id='Sobol_plot_datadriven', style={'width': 'inherit', 'height':'35vh'})

left_side = [
    dbc.Row([dbc.Col(method_dropdown,width=6)]),
    dbc.Row([dbc.Col(
        dbc.Button('Compute Polynomial', id='CU_button_datadriven', n_clicks=0, className='ip_buttons',color='primary',disabled=False))
    ]),
    dbc.Row([dbc.Col(dbc.Alert(id='poly-warning-datadriven',color='danger',is_open=False), width=3)]),
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
        dcc.Graph(id='plot_poly_3D_datadriven', style={'width': 'inherit','height':'60vh'}, figure=polyfig3D),
        dbc.Alert(id='plot_poly_info_datadriven',color='primary',is_open=False)
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


layout = dbc.Container(
    [
        html.H2("Uncertainty quantification for data-driven model", id='main_driven_text'),
        dbc.Row(
            [
                dbc.Col(dcc.Markdown(
                    'Upload data to construct polynomials and compute uncertainty',
                    id='info_driven_text'), width='auto'),
                dbc.Col(info,width='auto')
            ], align='center', style={'margin-bottom': '10px'}
        ),
        dbc.Row(dbc.Col(TOP_CARD, width=12), style={'margin-bottom': '10px'}),
dbc.Row(dbc.Col(COMPUTE_CARD,width=12),
            style={'margin-top':'10px'}),
        dcc.Store(id='UploadDF'),
        dcc.Store(id='ParamData'),
        dcc.Store(id='column-headers'),
        dcc.Store(id='BasisObj'),
        dcc.Store(id='PolyObj')
    ],
    fluid=True

)




def ParseData(content,filename):
    content_type,content_string=content.split(',')
    try:
        if 'csv' in filename:
            decoded = base64.b64decode(content_string)
            # print(decoded)
            # df=np.genfromtxt(io.StringIO(decoded.decode('utf-8')),delimiter=',')
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            cols=list(df.columns)
            data=df.to_numpy()
            # for i in range(len(df)):
            #     data.append(df[i])
            data=np.array(data)
            return data,cols
    except Exception:
        return None

    else:
        raise PreventUpdate

@app.callback(
    ServersideOutput('UploadDF','data'),
    ServersideOutput('column-headers','data'),
    Output('filename_append','children'),
    Input('upload-data-driven','filename'),
    Input('upload-data-driven','contents'),
)
def ParsedData(filename,content):
    if content is not None:
        (df,columns)=ParseData(content,filename)
        children=[filename]
        return df,columns,children
    else:
        raise PreventUpdate

@app.callback(
    Output('upload-data-table','data'),
    Output('upload-data-table','columns'),
    Input('filename_append','children'),
    Input("UploadDF",'data'),
    Input('column-headers','data')
)
def DatasetInfo(filename,df,cols):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'UploadDF' in changed_id:
            data=[]
            vals = df
            for i in range(vals.shape[0]):
                val_dict = {}
                for j,column in enumerate(cols):
                    if np.isnan(vals[i][j]):
                        vals[i][j] = 0
                    val_dict['{}'.format(column)]=vals[i][j]
                    if j == vals.shape[1] - 1:
                        data.append(val_dict)
            columns = [
                {'name': i, 'id': i, 'deletable': False, 'type': 'numeric', 'format': Format(precision=4)}
                for i in data[0].keys()]
            return data,columns
    else:
            raise PreventUpdate



@app.callback(
        Output('output-select','options'),
        Output('output-select','value'),
        Input('upload-data-table', 'columns'),
        prevent_initial_call=True)
def data_dropdown(columns):
    if columns is not None:
        options=[]
        for i in range(len(columns)):
            output=columns[i]['name']
            options.append({'label': output, 'value': output})
        value = output
        return options, value
    else:
        raise PreventUpdate

@app.callback(
        Output('num-input','children'),
        Output('num-output','children'),
        Input('upload-data-table', 'columns'),
        Input('output-select','value'),
        prevent_initial_call=True)
def InputVars(columns,select):
    if columns is not None:
        num=len(columns)-1
        return num,'1'
    else:
        raise PreventUpdate


@app.callback(
    Output('distribution-select','disabled'),
    Input('mode-select','value'),
    prevent_intial_call=True
)
def DisabledParam(mode):
    if mode=='manual':
        return False
    elif mode=='auto':
        return True
    else:
        raise PreventUpdate



def CreateParam(data,columns,distribution):
    if data is not None:
        if distribution is not None:
            dist=distribution
        else:
            pass
        param_objs=[]
        values = []
        lower = 0
        upper = 0
        options = []
        for index,i in enumerate(data[0].keys()):
            values = ([vals['{}'.format(i)] for vals in data])
            values = [vals for vals in values if values != 'nan']
            try:
                lower = min(values)
                upper = max(values)
            except NameError:
                print('Incorrect data')

            param_objs.append(eq.Parameter(distribution=dist,lower=lower,upper=upper,order=3))
        return param_objs

def CreateParamWeights(data,columns,distribution):
    if data is not None:
        param_objs=[]
        for index, i in enumerate(data[0].keys()):
            values = ([vals['{}'.format(i)] for vals in data])
            values = np.array(values)
            weight_obj=eq.Weight(values)
            param_objs.append(eq.Parameter(distribution='data',weight_function=weight_obj,order=3))
        return param_objs

@app.callback(
    ServersideOutput('ParamData','data'),
    ServersideOutput('BasisObj','data'),
    Input('upload-data-table','data'),
    Input('upload-data-table', 'columns'),
    Input('output-select','value'),
    Input('mode-select','value'),
    Input('distribution-select','value'),
    Input('CU_button_datadriven','n_clicks'),
    prevent_initial_call=True
)
def ComputeParams(data,columns,output,mode,distribution,n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button_datadriven' in changed_id:
        for i in range(len(data)):
            data[i].pop('{}'.format(output))
        if mode=='manual':
            param_objs=CreateParam(data,columns,distribution)
            mybasis=Set_Basis()
            return param_objs,mybasis
        else:
            param_objs=CreateParamWeights(data,columns,distribution)
            mybasis=Set_Basis()
            return param_objs,mybasis
    else:
        raise PreventUpdate



def Set_Basis():
    basis_set = eq.Basis('total-order')
    return basis_set



def Set_Polynomial(parameters, basis, method,x_data,y_data):
    mypoly = eq.Poly(parameters=parameters, basis=basis, method=method,
                     sampling_args= {'mesh': 'user-defined', 'sample-points': np.array(x_data), 'sample-outputs': np.array(y_data)})
    return mypoly

@app.callback(
    ServersideOutput('PolyObj', 'data'),
    Output('mean_datadriven', 'value'),
    Output('variance_datadriven', 'value'),
    Output('r2_score_datadriven', 'value'),
    Trigger('CU_button_datadriven', 'n_clicks'),
    Input('ParamData', 'data'),
    Input('BasisObj', 'data'),
    Input('upload-data-table','data'),
    Input('column-headers','data'),
    Input('solver_method_data', 'value'),
    Input('output-select','value'),
    prevent_initial_call=True
)
def SetModel(params,mybasis,data,cols,method,y):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print('ch',changed_id)
    if 'CU_button_datadriven' in changed_id:
        print(cols)
        y_data = []
        x_data = [[None for y in range(len(data[0].keys()) - 1)]
                  for x in range(len(data))]
        for i in range(len(data)):
            for ind, j in enumerate(cols):
                if j == '{}'.format(y):
                    y_data.append(data[i][j])
                else:
                    x_data[i][ind] = data[i][j]
        mypoly = Set_Polynomial(params, mybasis, method,x_data,y_data)
        try:
            mypoly.set_model()
        except KeyError:
            return None,None,None,None,False,True,"Incorrect Model evaluations"
            # except AssertionError:
            #     return None,None,None,True,None,False,True,"Incorrect Data uploaded"
            # except ValueError:
            #     return None, None, None, None, False, True, "Incorrect Model evaluations"
            # except IndexError:
            #     return None, None, None, None, False, True, "Incorrect Model evaluations"
        mean, var = mypoly.get_mean_and_variance()
        DOE=mypoly.get_points()
        print(mean,var)
        y_pred = mypoly.get_polyfit(np.array(x_data))
        r2_score = eq.datasets.score(np.array(y_data), y_pred, metric='r2')
        return mypoly, mean, var, r2_score ###
    else:
        raise PreventUpdate


@app.callback(
    Output('sobol_order_datadriven', 'options'),
    Input('CU_button_datadriven', 'n_clicks'),
    Input('column-headers','data'),
    State('sobol_order_datadriven', 'options')
    ,
    prevent_intial_call=True
)
def SobolCheck(n_clicks, cols, options):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button_datadriven' in changed_id:
        opt = options
        ndims=len(cols)-1
        if ndims == 1:
            return options
        elif ndims == 2:
            opt[0]['disabled'] = False
            opt[1]['disabled'] = False
            opt[2]['disabled'] = True
            return opt
        elif ndims >= 3:
            opt[0]['disabled'] = False
            opt[1]['disabled'] = False
            opt[2]['disabled'] = False
            return opt
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(
    Output('Sobol_plot_datadriven', 'figure'),
    Output('sobol_order_datadriven', 'disabled'),
    Output('Sobol_plot_datadriven', 'style'),
    Input('PolyObj', 'data'),
    Input('sobol_order_datadriven', 'value'),
    Trigger('CU_button_datadriven', 'n_clicks'),
    Input('column-headers','data'),
    State('Sobol_plot_datadriven', 'figure'),
    prevent_initial_call=True

)
def Plot_Sobol(mypoly, order, cols, fig):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button_datadriven' in changed_id:
        layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                  'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
        fig = go.Figure(layout=layout)
        fig.update_xaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
        fig.update_yaxes(color='black', linecolor='black', showline=True, tickcolor='black', ticks='outside')
        ndims=len(cols)-1
        if ndims == 1:
            disabled = True
        else:
            disabled = False

            if mypoly is not None:
                sobol_indices = mypoly.get_sobol_indices(order=order)
                layout = {'margin': {'t': 0, 'r': 0, 'l': 0, 'b': 0},
                          'paper_bgcolor': 'white', 'plot_bgcolor': 'white', 'autosize': True}
                fig = go.Figure(layout=layout)
                if order == 1:
                    fig.update_yaxes(title=r'$S_{i}$')
                    labels = [r'$S_%d$' % i for i in range(1, (ndims) + 1)]
                    to_plot = [sobol_indices[(i,)] for i in range((ndims))]

                elif order == 2:
                    fig.update_yaxes(title=r'$S_{ij}$')
                    labels = [r'$S_{%d%d}$' % (i, j) for i in range(1, int(ndims) + 1) for j in
                              range(i + 1, int(ndims) + 1)]
                    to_plot = [sobol_indices[(i, j)] for i in range(int(ndims)) for j in range(i + 1, int(ndims))]

                elif order == 3:
                    fig.update_yaxes(title=r'$S_{ijk}$')
                    labels = [r'$S_{%d%d%d}$' % (i, j, k) for i in range(1, int(ndims) + 1) for j in
                              range(i + 1, int(ndims) + 1) for k in
                              range(j + 1, int(ndims) + 1)]
                    to_plot = [sobol_indices[(i, j, k)] for i in range(int(ndims)) for j in range(i + 1, int(ndims)) for
                               k in
                               range(j + 1, int(ndims))]

                # fig.update_xaxes(nticks=len(sobol_indices),tickvals=labels,tickangle=45)
                data = go.Bar(
                    x=np.arange(len(sobol_indices)),
                    y=to_plot, marker_color='LightSkyBlue', marker_line_width=2, marker_line_color='black')
                fig = go.Figure(layout=layout, data=data)
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=np.arange(len(sobol_indices)),
                        ticktext=labels
                    ),
                    # uniformtext_minsize=8, uniformtext_mode='hide',
                    xaxis_tickangle=-30
                )
        style = {'width': 'inherit', 'height': '35vh'}
        return fig, disabled, style
    else:
        raise PreventUpdate


@app.callback(
    Output('plot_poly_3D_datadriven', 'figure'),
    Output('plot_poly_3D_datadriven','style'),
    Output('plot_poly_info_datadriven','is_open'),
    Output('plot_poly_info_datadriven','children'),
    Input('PolyObj', 'data'),
    Trigger('CU_button_datadriven', 'n_clicks'),
    Input('column-headers','data'),
    State('plot_poly_3D_datadriven', 'figure'),
    prevent_initial_call=True
)
def Plot_poly_3D(mypoly, cols,fig):
    hide={'display':'None'}
    default={'width':'600px'}
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'CU_button' in changed_id:
        if (mypoly is not None):
            ndims=len(cols)-1
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
