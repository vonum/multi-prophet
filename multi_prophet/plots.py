from fbprophet.plot import plot, plot_components

def plotly_plot(model, forecast, **kwargs):
    return plot(model, forecast, **kwargs)

def plotly_components_plot(model, forecast, **kwargs):
    return plot_components(model, forecast, **kwargs)
