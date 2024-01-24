# multilingual-fact-checking
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

**❗Note: the missing annotations and platform will be uploaded by early February 2024❗**

## Code
* [QACG Data Generation](https://github.com/aic-factcheck/Zero-shot-Fact-Verification)
* [ColBERTv2](https://github.com/aic-factcheck/ColBERTv2)
* [FactSearch](https://github.com/aic-factcheck/Zero-shot-Fact-Verification) (TBD)

Overall information on running the code (TBD)

## Data to Train QACG Models
1. [SQuAD-cs](https://huggingface.co/datasets/ctu-aic/squad-cs) 
2. [QA2D-cs](https://huggingface.co/datasets/ctu-aic/qa2d-cs) 
3. [QA2D-pl](https://huggingface.co/datasets/ctu-aic/qa2d-pl) 
4. [QA2D-sk](https://huggingface.co/datasets/ctu-aic/qa2d-sk) 

## QACG Models
1. Question Generation:
* [mt5-large-qg-sum](https://huggingface.co/ctu-aic/mt5-large-qg-sum)
  
2. Claim Generation:
* [mt5-large-cg-sum](https://huggingface.co/ctu-aic/mt5-large-cg-sum)
  
## QACG Generated Data
1. [QACG-CS](https://huggingface.co/datasets/ctu-aic/qacg-cs)
2. [QACG-EN](https://huggingface.co/datasets/ctu-aic/qacg-en)
3. [QACG-PL](https://huggingface.co/datasets/ctu-aic/qacg-pl)
4. [QACG-SK](https://huggingface.co/datasets/ctu-aic/qacg-sk)
5. [QACG-MIX](https://huggingface.co/datasets/ctu-aic/qacg-mix)
6. [QACG-SUM](https://huggingface.co/datasets/ctu-aic/qacg-sum)

## ColBERTV2 Evidence Retrieval
[colbertv2-QACG-SUM](https://huggingface.co/ctu-aic/colbertv2-qacg-sum/tree/main)

## NLI Veracity Evaluation
[nli-QACG-SUM](https://huggingface.co/ctu-aic/nli-qacg-sum)

## NLI Annotations
(TBD)
## Evidence Retrieval Annotations
(TBD)
