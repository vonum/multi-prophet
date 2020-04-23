from fbprophet.plot import plot_plotly, plot_components_plotly

def plotly_plot(model, forecast, **kwargs):
    return plot_plotly(model, forecast, **kwargs)

def plotly_components_plot(model, forecast, **kwargs):
    return plot_components_plotly(model, forecast, **kwargs)
