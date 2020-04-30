# Multi Prophet
[![Build Status](https://vonum.semaphoreci.com/badges/multi-prophet.svg)](https://vonum.semaphoreci.com/projects/multi-prophet)
[![PyPI version](https://badge.fury.io/py/multi-prophet.svg)](https://badge.fury.io/py/multi-prophet)

Multi Prophet is a procedure for forecasting time series data for multipe
dependent variables based on [Facebook
Prophet](https://facebook.github.io/prophet/) package. If you have no prior
experience with Facebook Prophet, check out their docs.

Multi Prophet does not train a single model with many outputs, it just wraps
Facebook Prophet interface to support configuration and controll over multiple
models. Multi Prophet has a very similar interface as Facebook Prophet.

The main difference is that return values of each method is a dictionary where
each dependent value is a key, and the value is thereturn value of the linked
Facebook Prophet model.

If Prophet return value is a data frame, then MultiProphet return value will be:
``` python
{"dependent_variable1": df1, "dependent_variable2": df2}
```

### Installation
Multi Prophet is on PyPi.
`pip install multi-prophet`

### Getting started
Creating a basic model is almost the same as creating a Prophet model:
#### Prophet
```python
# dataframe needs to have columns ds and y
from fbprophet import Prophet

m = Prophet()
m.fit(df)

future = m.create_future_dataframe(df)
forecast = m.predict(future)
m.plot(forecast)
```

#### Multi Prophet
```python
# dataframe needs to have column ds, and it has y1 and y2 as dependent variables
from multi_prophet import MultiProphet

m = MultiProphet(columns=["y1", "y2"])
m.fit(df)

future = m.create_future_dataframe(df)
forecast = m.predict(future)
m.plot(forecast)
```

### Adding country holidays
#### Prophet
```python
m.add_country_holidays(country_name="US")
```

#### Multi Prophet
```python
# For all dependent variables
m.add_country_holidays("US")

# For selected dependent variables
m.add_country_holidays("US", columns=["y1"])
```

### Adding seasonality
#### Prophet
```python
m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
```

#### Multi Prophet
```python
# For all dependent variables
m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

# For selected dependent variables
m.add_seasonality(name="monthly", period=30.5, fourier_order=5, columns=["y1"])
```

### Adding regressors
```python
# Prophet
m.add_regressor("Matchday")

# Multi Prophet
# For all dependent variables
m.add_regressor("Matchday")

# For selected dependent variables
m.add_regressor("Matchday", columns=["y"])
```

### Ploting results
#### Prophet
```python
# Prophet
m.plot(forecast)
m.plot_components(forecast)

# With Plotly
from fbprophet.plot import plot_plotly, plot_components_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)
py.iplot(fig)

fig = plot_components_plotly(m, forecast)
py.iplot(fig)
```

#### Multi Prophet
```python
# Multi Prophet
m.plot(forecast)
m.plot_components(forecast)

# With Plotly
figures = m.plot(forecast, plotly=True)
for fig in figures.values():
    fig.show()

# or access by key
figures["y1"].show()

figures = m.plot_components(forecast, plotly=True)
for fig in figures.values():
    fig.show()

# or access by key
figures["y1"].show()
```

### Facebook Prophet model configuration
Facebook Prophet supports a lot of configuration through kwargs. There are
two ways to do it with Multi Prophet:
1. Through kwargs just as with Facebook Prophet
    * Prophet
```python
m = Prophet(growth="logistic")
m.fit(self.df, algorithm="Newton")
m.make_future_dataframe(7, freq="H")
m.add_regressor("Matchday", prior_scale=10)
```

    * Multi Prophet
```python
m = MultiProphet(growth="logistic")
m.fit(self.df, algorithm="Newton")
m.make_future_dataframe(7, freq="H")
m.add_regressor("Matchday", prior_scale=10)
```

2. Configuration through constructor
```python
# Same configuration for each dependent variable
m = MultiProphet(columns=["y1", "y2"],
                 growth="logistic",
                 weekly_seasonality=True,
                 n_changepoints=50)

# Different configuration for each model
config = {
    "y1": {"growth": "linear", "daily_seasonality": True},
    "y2": {"growth": "logistic", "weekly_seasonality": True}
}
m = MultiProphet(columns=["y1", "y2"], config=config)

# Adding regressors (dataframe has columns c1 and c2)
regressors = {
    "y1": [
        {"name": "c1", "prior_scale": 0.5},
        { "name": "c2", "prior_scale": 0.3}
    ],
    "y2": [{"name": "c2", "prior_scale": 0.3}]
}
m = MultiProphet(columns=["y1", "y2"], regressors=regressors)
```
