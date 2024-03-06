
import dash_bootstrap_components as dbc
from dash import dcc

def define_slot_options(cfg, slotcfgs, nslots):

    def show(c):
        return {} if c else {"display": "none"}
    

    def width(c, v):
        return v if c else "0"

    models = cfg["retrieval_models"]
    scoring_models = cfg["scoring_models"]
    rte_models = cfg["nli_models"]

    slots = []
    for slotidx, slotcfg in enumerate(slotcfgs[0:nslots]):
        print(slotcfg)
        title = f"{slotidx+1}."
        model_name = slotcfg["retrieval_model_name"]
        model_k = slotcfg["k"]
        importance_model_name = slotcfg["importance"]
        importance_type = slotcfg["importance_type"]
        nli_model_name = slotcfg["nli_model_name"]

        c = cfg["show_model"]
        colid = {"type": "model_col", "id": slotidx}
        model_choice = dbc.Col([
            dbc.FormText(f"{title} Model"),
            dcc.Dropdown(
                id={"type": "model", "id": slotidx},
                options=[{"label": v["name"], "value": k}
                            for k, v in models.items()],
                value=model_name, clearable=False)],
                width=width(c, "3"), style=show(c), id=colid)
        model_choice_tip = dbc.Tooltip("Choose evidence retrieval model.", target=colid, placement="bottom")

        c = cfg["show_n_results"]
        colid = {"type": "model-k_col", "id": slotidx}
        n_results_choice = dbc.Col([
            dbc.FormText(f"Results"),
            dbc.Input(id={"type": "model-k", "id": slotidx},
                        type="number", min=1, max=1000, step=1, value=model_k)],
                width=width(c, "4"), style=show(c), id=colid)
        n_results_choice_tip = dbc.Tooltip("The number of evidence documents to retrieve.", target=colid, placement="bottom")

        c = cfg["show_importance"]
        colid = {"type": "importance-model_col", "id": slotidx}
        importance_model_choice = dbc.Col([
            dbc.FormText(f"Importance"),
            dcc.Dropdown(
                id={"type": "importance-model", "id": slotidx},
                options=[{"label": v["name"], "value": k}
                            for k, v in scoring_models.items()],
                value=importance_model_name),
            ], width=width(c, "2"), style=show(c), id=colid)
        importance_model_choice_tip = dbc.Tooltip("Importance model - FIX TOOLTIP.", target=colid, placement="bottom")

        c = cfg["show_granularity"]
        colid = {"type": "importance-type_col", "id": slotidx}
        granularity_choice = dbc.Col([
            dbc.FormText(f"Granularity"),
            dbc.RadioItems(
                id={"type": "importance-type", "id": slotidx},
                options=[
                    {'label': 'Sentence', 'value': 'sentence'},
                    {'label': 'Word', 'value': 'word'},
                ], value=importance_type)
            ], width=width(c, "auto"), style=show(c), id=colid)
        granularity_choice_tip = dbc.Tooltip("Granularity - FIX TOOLTIP.", target=colid, placement="bottom")

        c = cfg["show_nli_model"]
        colid = {"type": "nli-model_col", "id": slotidx}
        nli_model_choice = dbc.Col([
            dbc.FormText(f"NLI"),
            dcc.Dropdown(
                id={"type": "nli-model", "id": slotidx},
                options=[{"label": v["name"], "value": k}
                            for k, v in rte_models.items()],
                value=nli_model_name),
            ], width=width(c, "3"), style=show(c), id=colid)
        nli_model_choice_tip = dbc.Tooltip("Choose NLI model classifying evidence to SUPPORTS, REFUTES and NOT ENOUGH INFO classes.", target=colid, placement="bottom")
        
        slots.append(model_choice)
        slots.append(n_results_choice)
        slots.append(importance_model_choice)
        slots.append(granularity_choice)
        slots.append(nli_model_choice)

        slots.append(model_choice_tip)
        slots.append(n_results_choice_tip)
        slots.append(importance_model_choice_tip)
        slots.append(granularity_choice_tip)
        slots.append(nli_model_choice_tip)

        # slot = dbc.Col([
        #     dbc.Row([model_choice,
        #             n_results_choice,
        #             importance_model_choice,
        #             granularity_choice,
        #             nli_model_choice,
        #             ]),
        # ])
        # slots.append(slot)
    return slots
