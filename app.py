import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc


external_stylesheets = [dbc.themes.SPACELAB, 'https://codepen.io/chriddyp/pen/bWLwgP.css']

app=dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True,meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=0.2, maximum-scale=1.2,minimum-scale=0.5'}])
