import dash_bootstrap_components as dbc

def define_search_row(cfg):

    claim_field = dbc.Col([
        dbc.Input(type="text", id="claim-txt", placeholder="Enter claim", value="", debounce=True),
        ], width=7)
    
    claim_field_tip = dbc.Tooltip("Use period to end the claim sentence.", target="claim-txt", placement="bottom")
    
    search_button = dbc.Col([dbc.Button('Search', id='claim-search',
                            color="primary", n_clicks=0)], width="auto")
    
    label_opts = []
    if cfg["show_search_titles"]:
        label_opts.append({"label": "Search Titles", "value": "search_titles"})
    if cfg["show_detailed_score"]:
        label_opts.append({"label": "Detailed Score", "value": "detailed_score"})

    checks = dbc.Col([dbc.Checklist(
        options=label_opts,
        value=["search_titles"],
        id="search-switches",
        switch=True,
        )], width="auto")
    
    row_items = [claim_field, search_button, checks, claim_field_tip]
            
    # return dbc.Row(row_items, align='start')
    return row_items
