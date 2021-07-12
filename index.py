import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input,Output,State

import equadratures as eq
import equadratures.distributions as db

from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import jsonpickle
import ast
from equadratures import *
import numexpr as ne
from navbar import navbar
from app import app
from apps import Analytical
from apps import Offline
from utils import convert_latex

home_text=r'''
## UNCERTAINTY QUANTIFICATION

Uncertainty Quantification in laymen term means finding the uncertainty of our QOI (Quantity of interest) based on the
uncertainty in input parameters. It tries to determine how likely certain outcomes are if some aspects of the system are 
not exactly known. Figure $1$ represents the same. The uncertainty in our output y is dependent on the uncertainties in 
parameters $s1$ and $s2$ when propagated through our model $f(s1, s2)$.




....


'''

home_text=dcc.Markdown(convert_latex(home_text),dangerously_allow_html=True, style={'text-align':'justify'})

homepage = dbc.Container([home_text])

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div([homepage],id="page-content"),
    ],
    style={'padding-top': '70px'}
)


@app.callback(Output('page-content', 'children'),
    Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return homepage
    if pathname == '/Analytical':
        return Analytical.layout
    elif pathname == '/Offline':
        return Offline.layout
    else:
        raise PreventUpdate



if __name__ == '__main__':
    app.run_server(debug=True)