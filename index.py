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

home_text=r'''
## UNCERTAINTY QUANTIFICATION

....


'''

home_text=dcc.Markdown(home_text)

homepage = dbc.Container([home_text])

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div(homepage,id="page-content"),
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



if __name__ == '__main__':
    app.run_server(debug=True)