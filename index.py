import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, server
from apps import Analytical, Data_driven
from navbar import navbar
from utils import convert_latex


###################################################################
# Homepage text
###################################################################

home_text=r'''
## Uncertainty Quantification

**Uncertainty Quantification** in laymen term means finding the uncertainty of our **QOI** (Quantity of interest) based on the
uncertainty in input parameters. It determines how likely certain outcomes are if some aspects of the system are 
not exactly known. **Figure 1** represents the same. The uncertainty in our output y is dependent on the uncertainties in 
parameters $s_1$ and $s_2$ when propagated through our model $f(s_1, s_2)$.

<figure style="width:60%">
<img alt="Polynomial chaos" src="static/model.png" />
</figure>

### Motivation for using Equadratures

**Monte Carlo Approach** is a method that is most simple approach used for simulations and experiments 
where we evaluate our model $f(s1)$ at $N$ number of $s_1$ randomly sampled from $\mathbb{S}$ . Then we calculate the mean and variance of 
the collected model outputs.
**Alternatively**, we can use Effective Quadratures to compute the moments of $y$.
We simply declare the usual building blocks (a Parameter, Basis and Poly object), 
give the Poly our data (or function) with **set_model** and then run **get_mean_and_variance**. A basic example of the workflow of equadratures is

```python
s1 = Parameter(distribution='uniform', lower=0., upper=1., order=2)
mybasis = Basis('univariate')
mypoly = Poly(parameters=s1, basis=mybasis, method='numerical-integration')
mypoly.set_model(our_function)
mean, var = mypoly.get_mean_and_variance()
```

Here, Effective Quadratures has calculated the quadrature points in $s_1$ (dependent on our choice of distribution, order and the Basis). Then it has evaluated our model at these points, and used the results to construct a polynomial approximation (response surface),

$$f(s_1) \approx \sum_{i=1}^n x_ip_i(s_1)$$

### Models
There are three models present in the app **Analytical**, **Offline** and **Data-Driven** model. These models follow different approaches 
and workflow to compute the statistical moments and quantify uncertainty.

#### 1. Analytical Model
The Analytical Model is a **proof-of-concept app** where the model is an analytical function defined by the user. This 
???analytical??? app helps the user to define a model function, and its input parameters.It contains features such as adding a parameter (maximum 5), defining various properties such as distribution,
 statistical moments, min/max values and order. As different distributions have different attributes i.e. Uniform distribution has attributes min/max and order whereas Gaussian distribution has attributes mean, variance and order, the analytical model
 handles this dynamic selection to provide users with various distributions to use the application at its full capacity.

#### 2. Offline Model
The Offline Model is quite similar to the analytical model in terms of workflow, the major difference between models is the
application. The offline model is constructed for professional use. The parameter and basis definition is similar to the analytical model, but here the user is
expected to upload the **DOE evaluations** of their simulations and then quantify uncertainty.  
.

#### 3. Data-Driven Model
 The Data-Driven as the name suggests has a more **data-centric** workflow, unlike the previous models, the user uploads their data,
select their output variable and based on their selection equadratures construct input parameters. Here, two choices are provided to the user for 
Distribution Selection, first is **KDE** approach, which makes use of Kernel Density Estimation to produce accurate results but taking longer time to compute,
second is the **Canonical** approach, here the users manually select the distribution of the input parameters, this method makes use of scipy fit method to compute the 
statistical moments, which are then fed to Equadratures parameter. Using these methods, the user can gain insights regarding the senstivity of their dataset and the underlying statistical moments. 
.
 
'''

home_text=dcc.Markdown(convert_latex(home_text),dangerously_allow_html=True, style={'text-align':'justify'})

# disclaimer message
final_details = r'''
This app is currently hosted *on the cloud* via [Heroku](https://www.heroku.com). Resources are limited and the app may be slow when there are multiple users. If it is too slow please come back later! 

Please report any bugs to [ascillitoe@effective-quadratures.org](mailto:ascillitoe@effective-quadratures.org).
'''
final_details = dbc.Alert(dcc.Markdown(final_details),
        dismissable=True,is_open=True,color='info',style={'padding-top':'0.4rem','padding-bottom':'0.0rem'})

homepage = dbc.Container([home_text])

###################################################################
# 404 page
###################################################################
msg_404 = r'''
**Oooops** 

Looks like you might have taken a wrong turn!
'''

container_404 = dbc.Container([ 
    dbc.Row(
            [
                dcc.Markdown(msg_404,style={'text-align':'center'})
            ], justify="center", align="center", className="h-100"
    )
],style={"height": "90vh"}
)

###################################################################
# Footer
###################################################################
footer = html.Div(
        [
            html.P('App built by Simardeep Singh Sethi, Bryn Noel Ubald, and Ashley Scillitoe'),
            html.P(html.A('equadratures.org',href='https://equadratures.org/')),
            html.P('Copyright ?? 2021')
        ]
    ,className='footer', id='footer'
)

###################################################################
# App layout (adopted for all sub-apps/pages)
###################################################################

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div([homepage],id="page-content"),
        footer,
    ],
    style={'padding-top': '70px'}
)

###################################################################
# Callback to return page
###################################################################
@app.callback(Output('page-content', 'children'),
    Output('footer','style'),
    Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return homepage, {'display':'block'}
    if pathname == '/Analytical':
        return Analytical.layout, {'display':'block'}
    elif pathname == '/Data_driven':
        return Data_driven.layout, {'display':'block'}
    else:
        return container_404, {'display':'none'}

###################################################################
# Run Server
###################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
