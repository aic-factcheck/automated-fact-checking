# Automated Fact-Checking
Data, models, and code to reproduce our [Pipeline and Dataset Generation for Automated Fact-checking in Almost Any Language](https://arxiv.org/abs/2312.10171) paper.
Currently in review for [NCAA](https://link.springer.com/journal/521) journal.

```
@article{drchal2023pipeline,
  title={Pipeline and Dataset Generation for Automated Fact-checking in Almost Any Language},
  author={Drchal, Jan and Ullrich, Herbert and Mlyn{\'a}{\v{r}}, Tom{\'a}{\v{s}} and Moravec, V{\'a}clav},
  journal={arXiv preprint arXiv:2312.10171},
  year={2023}
}
```

## Code
* [QACG Data Generation](https://github.com/aic-factcheck/Zero-shot-Fact-Verification) -- our fork of the original QACG procedure.
* [ColBERTv2](https://github.com/aic-factcheck/ColBERTv2) -- our fork of ColBERTv2. The retrieval for FactSearch is realized via REST API.
* [anserini-indexing](https://github.com/aic-factcheck/anserini-indexing) -- wrapper for ANSERINI BM25.The retrieval for FactSearch is realized via REST API.
* FactSearch source is hosted in this repository.

## Data to Train QACG Models
The following datasets were created by machine translation using [DeepL](https://www.deepl.com/translator). See the paper for more details.

1. [SQuAD-cs](https://huggingface.co/datasets/ctu-aic/squad-cs) 
2. [QA2D-cs](https://huggingface.co/datasets/ctu-aic/qa2d-cs) 
3. [QA2D-pl](https://huggingface.co/datasets/ctu-aic/qa2d-pl) 
4. [QA2D-sk](https://huggingface.co/datasets/ctu-aic/qa2d-sk) 

## QACG Models
1. Question Generation model trained on a concatenation of Czech, English, Polish, and Slovak SQuAD datasets:
* [mt5-large-qg-sum](https://huggingface.co/ctu-aic/mt5-large-qg-sum)
  
2. Claim Generation model train on a concatenation of Czech, English, Polish, and Slovak QA2D datasets:
* [mt5-large-cg-sum](https://huggingface.co/ctu-aic/mt5-large-cg-sum)
  
## QACG Generated Data
All QACG-generated datasets are based on the corresponding Wikipedia snapshots using the QACG models above.
The QACG-mix combines all four languages, preserving the size of each individual language dataset. The QACG-sum is a four-times larger concatenation of all individual language datasets.
1. [QACG-cs](https://huggingface.co/datasets/ctu-aic/qacg-cs)
2. [QACG-en](https://huggingface.co/datasets/ctu-aic/qacg-en)
3. [QACG-pl](https://huggingface.co/datasets/ctu-aic/qacg-pl)
4. [QACG-sk](https://huggingface.co/datasets/ctu-aic/qacg-sk)
5. [QACG-mix](https://huggingface.co/datasets/ctu-aic/qacg-mix)
6. [QACG-sum](https://huggingface.co/datasets/ctu-aic/qacg-sum)

## ColBERTv2 Evidence Retrieval
[colbertv2-QACG-SUM](https://huggingface.co/ctu-aic/colbertv2-qacg-sum/tree/main)

## NLI Veracity Evaluation
[nli-QACG-sum](https://huggingface.co/ctu-aic/nli-qacg-sum)

## NLI Annotations
[Here](data/ncaa/nli_annotations)

## Evidence Retrieval Annotations
[Here](data/ncaa/er_annotations)
