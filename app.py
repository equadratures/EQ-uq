from dash_extensions.enrich import Dash, FileSystemStore
import dash_bootstrap_components as dbc
from whitenoise import WhiteNoise
import os

os.makedirs("tmp/", exist_ok=True)
output_defaults=dict(backend=FileSystemStore(cache_dir="tmp/",threshold=100), session_check=True)

app = Dash(__name__, suppress_callback_exceptions=True, output_defaults=output_defaults,
        external_stylesheets=[dbc.themes.SPACELAB, 'https://codepen.io/chriddyp/pen/bWLwgP.css'],
        external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" ])
app.title = "Uncertainty Quantification with equadratures"

server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')
# To serve static files (e.g. images etc)
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')
