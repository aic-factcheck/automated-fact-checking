from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sentence_transformers.cross_encoder import CrossEncoder

from aic_nlp_utils.batch import batch_apply

class SupportRefuteNEIModel:
    '''This encapsulates any NLI model aimed at fact-checking claim classification. It maps SUPPORTS/REFUTES/NEI classes 
    '''
    def __init__(self, model_name, order: Optional[str], type_: str, claim_first=False):
        # actual order for this model: s - SUPORTS, r - REFUTES, n - NEI 
        # claim_first - order of claim and context in model input

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert type_ in set(["crossencoder", "default"])
        self.type_ = type_

        if type_ == "crossencoder":
            self.model = CrossEncoder(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

        if order:
            self.order = order.lower()
            assert self.order in ["srn", "rsn"]
        else:
            self.order = None
            id2label = self.model.config.id2label
            label2id = self.model.config.label2id
            assert len(id2label) == 3, id2label
            labels = [l for l in id2label.values()]
            allowed_labels = set(["s", "r", "n"])
            assert all([l in allowed_labels for l in labels]), labels

        self.claim_first = claim_first


    def predict(self, query_pairs, order: str="rsn", **kwargs):
        if self.claim_first:
            query_pairs = [(claim, context_) for context_, claim in query_pairs]

        if self.type_ == "crossencoder":
            C = self.model.predict(query_pairs, **kwargs)
        else:
            C = self._default_predict_helper(query_pairs, **kwargs)

        assert C.shape[1] == 3, f"three output classes expected, shape={C.shape}"
        if self.order is None:
            label2id = self.model.config.label2id
            idxs = [label2id[c] for c in order]
            C[:, :] = C[:, idxs]
        elif order != self.order:
            C[:,:2] = C[:, [1, 0]]
        return C
    
    def _default_predict_helper(self, inputs, batch_size=128, apply_softmax=False, **kwargs):
        def predict(inputs):
            max_length = self.tokenizer.model_max_length
            # print(f"inputs={inputs}")
            X = self.tokenizer(inputs, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            # print(f"X={X}")
            input_ids = X["input_ids"].to(self.device)
            attention_mask = X["attention_mask"].to(self.device)
            with torch.no_grad():
                Y = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
                if apply_softmax:
                    Y = F.softmax(Y, dim=1)
                return Y
            
        Ys = batch_apply(predict, inputs, batch_size=batch_size)
        Y = torch.vstack(Ys)
        Y = Y.detach().cpu().numpy()
        return Y



