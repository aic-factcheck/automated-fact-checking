import numpy as np

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html


from utils.dbcache import DBCache

def define_result_options_row(db: DBCache, nslots: int, maxslots: int, temporal: bool):
    cols = []

    slot_style = {} if nslots > 1 else {"display": "none"}
    col_slot = dbc.Col([
        dbc.FormText(f"# of Slots"),
        dbc.Input(id='nslots', type='number', value=nslots, min=1, max=maxslots, step=1)],
        width="auto", style=slot_style)
    
    datemin = np.min(db.dates).date()
    datemax = np.max(db.dates).date()

    temporal_style = {} if temporal else {"display": "none"}

    col_time_span = dbc.Col([
            dbc.FormText("Time Span"),
            html.Div(
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=datemin,
                    max_date_allowed=datemax,
                    initial_visible_month=datemax,
                    start_date=datemin,
                    end_date=datemax,
                    first_day_of_week=1
                ))
        ], width=4, style=temporal_style)

    col_order = dbc.Col([
            dbc.FormText("Order by"),
            dcc.Dropdown(
                id=f"order-results",
                options=[
                    {"label": "Maximum Score", "value": "score"},
                    {"label": "Date ⇧", "value": "date_asc"},
                    {"label": "Date ⇩", "value": "date_desc"}],
                value="score", clearable=False)
        ], width=2, style=temporal_style)
    
    cols = [col_slot, col_time_span, col_order]
    # return dbc.Row(cols)
    return cols
