[![Build Status](https://vonum.semaphoreci.com/badges/multi-prophet.svg)](https://vonum.semaphoreci.com/projects/multi-prophet)

# Multi Prophet
Multi Prophet is a procedure for forecasting time series data for multipe
dependent variables based on [Facebook
Prophet](https://facebook.github.io/prophet/) package. If you have no prior
experience with Facebook Prophet, check out their docs.

Multi Prophet does not train a single model with many outputs, it just wraps
Facebook Prophet interface to support configuration and controll over multiple
models. Multi Prophet has a very similar interface as Facebook Prophet.

### Installation
Multi Prophet is on PyPi.
`pip install multi-prophet`

### Getting started
Creating a basic model is almost the same as creating a Prophet model:
```python
# Prophet
# dataframe needs to have columns ds and y
from fbprophet import Prophet

m = Prophet()
m.fit(df)

future = m.create_future_dataframe(df)
forecast = m.predict(future)
m.plot(forecast)

# Multi Prophet
# dataframe needs to have column ds, and it has y1 and y2 as dependent variables
from multi_prophet import MultiProphet

m = MultiProphet(columns=["y1", "y2"])
m.fit(df)

future = m.create_future_dataframe(df)
forecast = m.predict(future)
m.plot(forecast)
```

### Adding country holidays
```python
# Prophet
m.add_country_holidays(country_name="US")

# Multi Prophet
# For all dependent variables
m.add_country_holidays("US")

# For selected dependent variables
m.add_country_holidays("US", columns=["y1"])
```

### Adding seasonality
```python
# Prophet
m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

# Multi Prophet
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
```python
# Prophet
m.plot(forecast)
m.plot_components(forecast)

# With Plotly

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

### Facebook Prophet model configuration
Facebook Prophet supports a lot of configuration through kwargs. There are
two ways to do it with Multi Prophet:
1. Through kwargs just as with Facebook Prophet
```python
# Prophet
m = Prophet(growth="logistic")
m.fit(self.df, algorithm="Newton")
m.make_future_dataframe(7, freq="H")
m.add_regressor("Matchday", prior_scale=10)

# Multi Prophet
m = MultiProphet(growth="logistic")
m.fit(self.df, algorithm="Newton")
m.make_future_dataframe(7, freq="H")
m.add_regressor("Matchday", prior_scale=10)
```

2. Configuration through constructor
```python
# Multi Prophet
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
