# EQ-uq
An app for uncertainty quantification of computational models. The application uses [Equadratures](https://github.com/equadratures/equadratures) for creating parameters, basis and Polynomials to create models on user-defined data. Dash was used to design the layout of the application and it is deployed on Heroku.

The app can be viewed at http://www.uq.equadratures.org/

There are three models present in the app <b>Analytical</b>, <b>Offline</b> and <b>Data-Driven model</b>. These models follow different approaches and workflow to compute the statistical moments and quantify uncertainty.

<ol>
  <li><b>Analytical Model</b></li>
The Analytical Model is a proof-of-concept app where the model is an analytical function defined by the user. This “analytical” app helps the user to define a model function, and its input parameters.
  <li><b>Offline Model</b></li>
The Offline Model is quite similar to the analytical model in terms of workflow, the major difference between models is the application.The parameter and basis definition is similar to the analytical model, but here the user is expected to upload the DOE evaluations of their simulations and then quantify uncertainty.
  <li><b>Data-Driven Model</b></li>
The Data-Driven as the name suggests has a more data-centric workflow, unlike the previous models, the user uploads their data, select their output variable and based on their selection equadratures construct input parameters.
 </ol>
