from pathlib import Path
import stanza
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast, pipeline

def load_czert_ner_pipeline(model_name):
    # taken from zeroshot!
    # for Czech Czert models
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        cur_dev = torch.cuda.current_device()
        print(f"torch.cuda.current_device() = {cur_dev}")
        device = torch.device(f"cuda:{cur_dev}")
    else:
        device = torch.device("cpu")

    tokenizer = BertTokenizerFast(Path(model_name, "vocab.txt"), strip_accents=False, do_lower_case=False, truncate=True, model_max_length=512)
    ner_pipeline = pipeline("ner", model=model_name, tokenizer=tokenizer, aggregation_strategy="first", device=device)
    def ner_pipeline_pairs(text):
        ner_dicts = ner_pipeline(text)
        ner_pairs = [(text[e["start"]:e["end"]], e["entity_group"]) for e in ner_dicts]
        return ner_pairs
    return ner_pipeline_pairs


def load_transformer_ner_pipeline(model_name):
    # for general Transformer models
    ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="first", device_map='auto')
    def ner_pipeline_pairs(text):
        ner_dicts = ner_pipeline(text)
        ner_pairs = [(text[e["start"]:e["end"]], e["entity_group"]) for e in ner_dicts]
        return ner_pairs
    return ner_pipeline_pairs


def load_stanza_ner_pipeline(lang):
    # for English or stanza supported languages
    # for each text gives a triplet (ner, ner_type, ner-ner_type count in text)
    # the triplets are sorted by decreasing count
    stanza_nlp = stanza.Pipeline(lang, use_gpu = True, processors="tokenize,ner")
    def ner_pipeline_pairs(text):
        pass_doc = stanza_nlp(text)
        ner_pairs = [(ent.text, ent.type) for ent in pass_doc.ents] # text-type pairs
        return ner_pairs
    return ner_pipeline_pairs


def load_ner_pipeline(lang, ner_model_name=None):
    if lang == "cs":
        return load_czert_ner_pipeline(ner_model_name)
    elif lang == "sk":
        return load_transformer_ner_pipeline(ner_model_name)
    elif lang == "en":
        return load_stanza_ner_pipeline(lang)
    else:
        print(f"UNKNOWN language for NER: {lang}")