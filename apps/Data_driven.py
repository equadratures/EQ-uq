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
        dcc.Store(id='UploadDF'),
    ],
    fluid=True

)




def ParseData(content,filename):
    content_type,content_string=content.split(',')
    try:
        if 'csv' in filename:
            decoded = base64.b64decode(content_string)
            df=np.genfromtxt(io.StringIO(decoded.decode('utf-8')),delimiter=',')
            data=[]
            for i in range(len(df)):
                data.append(df[i])
            data=np.array(data)
            return data
        elif 'npy' in filename:
            r = base64.b64decode(content_string)
            data=np.load(io.BytesIO(r))
            return data
    except Exception:
        return None

    else:
        raise PreventUpdate

@app.callback(
    ServersideOutput('UploadDF','data'),
    Output('filename_append','children'),
    Input('upload-data-driven','filename'),
    Input('upload-data-driven','contents'),
)
def ParsedData(filename,content):
    if content is not None:
        df=ParseData(content,filename)
        children=[filename]
        return df,children
    else:
        raise PreventUpdate

@app.callback(
    Output('upload-data-table','data'),
    Output('upload-data-table','columns'),
    Input('filename_append','children'),
    Input("UploadDF",'data'),
)
def DatasetInfo(filename,df):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    if 'UploadDF' in changed_id:
            data=[]
            vals = df
            for i in range(vals.shape[0]):
                val_dict = {}
                for j in range(vals.shape[1]):
                    val_dict['col_{}'.format(j)]=vals[i][j]
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


# def CreateParam(data,distribution):
#     if data is not None:
#         ndims=data.shape[1]
#         if distribution is not None:
#             dist=distribution
#         else:
#             pass
#         param_objs=[]
#         for i in range(ndims):
#
#             param_objs.append()


