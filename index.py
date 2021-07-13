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



###################################################################
# Homepage text
###################################################################

home_text=r'''
## UNCERTAINTY QUANTIFICATION

<b>Uncertainty Quantification</b> in laymen term means finding the uncertainty of our <b>QOI</b> (Quantity of interest) based on the
uncertainty in input parameters. It determines how likely certain outcomes are if some aspects of the system are 
not exactly known. <b>Figure 1</b> represents the same. The uncertainty in our output y is dependent on the uncertainties in 
parameters $s_1$ and $s_2$ when propagated through our model $f(s_1, s_2)$.
<br><br>
<p align="center">
<img src="assets/model.png" alt="drawing" width="600" align='centre'/>
</p>

<br>

<h3> Motivation for using Equadratures </h3>

<b>Monte Carlo Approach</b> is a method that is most simple approach used for simulations and experiments 
where we evaluate our model $f(s1)$ at $N$ number of $s_1$ randomly sampled from $\mathbb{S}$ . Then we calculate the mean and variance of 
the collected model outputs.
<b>Alternatively</b>, we can use Effective Quadratures to compute the moments of $y$.
We simply declare the usual building blocks (a Parameter, Basis and Poly object), 
give the Poly our data (or function) with <b>set_model</b> and then run <b>get_mean_and_variance</b>
<br>
A basic example of the workflow of equadratures is
<br>
```python
s1 = Parameter(distribution='uniform', lower=0., upper=1., order=2)
mybasis = Basis('univariate')
mypoly = Poly(parameters=s1, basis=mybasis, method='numerical-integration')
mypoly.set_model(our_function)
mean, var = mypoly.get_mean_and_variance()
```

Here, Effective Quadratures has calculated the quadrature points in $s_1$ (dependent on our choice of distribution, order and the Basis).
 Then it has evaluated our model at these points, and used the results to construct a polynomial approximation (response surface),

$$f(s_1) \approx \sum_{i=1}^n x_ip_i(s_1)$$

....


'''

home_text=dcc.Markdown(convert_latex(home_text),dangerously_allow_html=True, style={'text-align':'justify'})

homepage = dbc.Container([home_text])

###################################################################
# App layout
###################################################################

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div([homepage],id="page-content"),
    ],
    style={'padding-top': '70px'}
)

###################################################################
# Callback to return page
###################################################################

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


###################################################################
# Run Server
###################################################################
if __name__ == '__main__':
    app.run_server(debug=True)