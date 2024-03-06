import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc

import logging
logger = logging.getLogger(__name__)

def fig_hist(data, f, labels, title, graphid):
    fig = go.Figure()
    for d, label in zip(data, labels):
        fig.add_trace(go.Histogram(x=f(d), histnorm='probability', name=label))

    fig.update_layout(barmode='overlay', title=title)
    fig.update_traces(opacity=0.75)
    
    return dcc.Graph(id=graphid, figure=fig)

def fig_line(data, f, labels, title, graphid, g=lambda x: x):
    fig = go.Figure()
    for d, label in zip(data, labels):
        y = g(f(d))
        fig.add_trace(go.Scatter(y=y, mode='lines', name=label))

    fig.update_layout(barmode='overlay', title=title)
    fig.update_traces(opacity=0.75)
    
    return dcc.Graph(id=graphid, figure=fig)