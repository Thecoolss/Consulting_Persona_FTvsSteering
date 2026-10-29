# Evaluation Results

## Mean scores by method x dimension

| method                    |   consistency |   content_relevance |   fluency |   naturalness |   persona_adherence |
|:--------------------------|--------------:|--------------------:|----------:|--------------:|--------------------:|
| baseline                  |          1.02 |                4.92 |      4.72 |          1    |                1.22 |
| finetuning                |          4.96 |                4.76 |      3.32 |          4.34 |                4.98 |
| prompting:detailed        |          4.96 |                4.52 |      3.4  |          3.86 |                5    |
| prompting:medium          |          4.8  |                4.4  |      3.42 |          3.92 |                5    |
| prompting:minimal         |          4.04 |                3.48 |      3.94 |          2.34 |                4.98 |
| steering:layer12_scale6.0 |          1.49 |                1.31 |      1.06 |          1    |                3.9  |
| steering:v2_L12_a0.5_pn   |          2.82 |                4.27 |      3.76 |          1.63 |                3.7  |

## Overall ranking (mean across all dimensions)

1. **finetuning** — 4.47
2. **prompting:detailed** — 4.35
3. **prompting:medium** — 4.31
4. **prompting:minimal** — 3.76
5. **steering:v2_L12_a0.5_pn** — 3.24
6. **baseline** — 2.58
7. **steering:layer12_scale6.0** — 1.75

## Best method per dimension

- consistency: **prompting:detailed** (4.96)
- content_relevance: **baseline** (4.92)
- fluency: **baseline** (4.72)
- naturalness: **finetuning** (4.34)
- persona_adherence: **prompting:detailed** (5.00)

## Pairwise significance (Wilcoxon, Holm-corrected)

| Dimension | A | B | mean A | mean B | p (Holm) | effect (rbc) | sig |
|---|---|---|---|---|---|---|---|
| persona_adherence | baseline | finetuning | 1.22 | 4.98 | 0.0000 | -1.00 | yes |
| persona_adherence | baseline | prompting:detailed | 1.22 | 5.00 | 0.0000 | -1.00 | yes |
| persona_adherence | baseline | prompting:medium | 1.22 | 5.00 | 0.0000 | -1.00 | yes |
| persona_adherence | baseline | prompting:minimal | 1.22 | 4.98 | 0.0000 | -1.00 | yes |
| persona_adherence | baseline | steering:layer12_scale6.0 | 1.22 | 3.90 | 0.0000 | -1.00 | yes |
| persona_adherence | baseline | steering:v2_L12_a0.5_pn | 1.22 | 3.70 | 0.0000 | -1.00 | yes |
| persona_adherence | finetuning | prompting:detailed | 4.98 | 5.00 | 1.0000 | -1.00 | no |
| persona_adherence | finetuning | prompting:medium | 4.98 | 5.00 | 1.0000 | -1.00 | no |
| persona_adherence | finetuning | prompting:minimal | 4.98 | 4.98 | 1.0000 | +0.00 | no |
| persona_adherence | finetuning | steering:layer12_scale6.0 | 4.98 | 3.90 | 0.0115 | +1.00 | yes |
| persona_adherence | finetuning | steering:v2_L12_a0.5_pn | 4.98 | 3.70 | 0.0005 | +1.00 | yes |
| persona_adherence | prompting:detailed | prompting:medium | 5.00 | 5.00 | 1.0000 | +0.00 | no |
| persona_adherence | prompting:detailed | prompting:minimal | 5.00 | 4.98 | 1.0000 | +1.00 | no |
| persona_adherence | prompting:detailed | steering:layer12_scale6.0 | 5.00 | 3.90 | 0.0115 | +1.00 | yes |
| persona_adherence | prompting:detailed | steering:v2_L12_a0.5_pn | 5.00 | 3.70 | 0.0005 | +1.00 | yes |
| persona_adherence | prompting:medium | prompting:minimal | 5.00 | 4.98 | 1.0000 | +1.00 | no |
| persona_adherence | prompting:medium | steering:layer12_scale6.0 | 5.00 | 3.90 | 0.0115 | +1.00 | yes |
| persona_adherence | prompting:medium | steering:v2_L12_a0.5_pn | 5.00 | 3.70 | 0.0005 | +1.00 | yes |
| persona_adherence | prompting:minimal | steering:layer12_scale6.0 | 4.98 | 3.90 | 0.0127 | +1.00 | yes |
| persona_adherence | prompting:minimal | steering:v2_L12_a0.5_pn | 4.98 | 3.70 | 0.0007 | +1.00 | yes |
| persona_adherence | steering:layer12_scale6.0 | steering:v2_L12_a0.5_pn | 3.90 | 3.71 | 1.0000 | +0.18 | no |
| content_relevance | baseline | finetuning | 4.92 | 4.76 | 1.0000 | +0.78 | no |
| content_relevance | baseline | prompting:detailed | 4.92 | 4.52 | 0.0151 | +0.90 | yes |
| content_relevance | baseline | prompting:medium | 4.92 | 4.40 | 0.0041 | +0.92 | yes |
| content_relevance | baseline | prompting:minimal | 4.92 | 3.48 | 0.0000 | +0.95 | yes |
| content_relevance | baseline | steering:layer12_scale6.0 | 4.92 | 1.31 | 0.0000 | +1.00 | yes |
| content_relevance | baseline | steering:v2_L12_a0.5_pn | 4.92 | 4.27 | 0.0027 | +0.93 | yes |
| content_relevance | finetuning | prompting:detailed | 4.76 | 4.52 | 0.7589 | +0.52 | no |
| content_relevance | finetuning | prompting:medium | 4.76 | 4.40 | 0.1763 | +0.63 | no |
| content_relevance | finetuning | prompting:minimal | 4.76 | 3.48 | 0.0000 | +0.90 | yes |
| content_relevance | finetuning | steering:layer12_scale6.0 | 4.83 | 1.31 | 0.0000 | +1.00 | yes |
| content_relevance | finetuning | steering:v2_L12_a0.5_pn | 4.84 | 4.27 | 0.0057 | +0.85 | yes |
| content_relevance | prompting:detailed | prompting:medium | 4.52 | 4.40 | 1.0000 | +0.20 | no |
| content_relevance | prompting:detailed | prompting:minimal | 4.52 | 3.48 | 0.0003 | +0.81 | yes |
| content_relevance | prompting:detailed | steering:layer12_scale6.0 | 4.52 | 1.31 | 0.0000 | +1.00 | yes |
| content_relevance | prompting:detailed | steering:v2_L12_a0.5_pn | 4.51 | 4.27 | 1.0000 | +0.31 | no |
| content_relevance | prompting:medium | prompting:minimal | 4.40 | 3.48 | 0.0026 | +0.70 | yes |
| content_relevance | prompting:medium | steering:layer12_scale6.0 | 4.40 | 1.31 | 0.0000 | +1.00 | yes |
| content_relevance | prompting:medium | steering:v2_L12_a0.5_pn | 4.39 | 4.27 | 1.0000 | +0.16 | no |
| content_relevance | prompting:minimal | steering:layer12_scale6.0 | 3.54 | 1.31 | 0.0000 | +0.96 | yes |
| content_relevance | prompting:minimal | steering:v2_L12_a0.5_pn | 3.53 | 4.27 | 0.0051 | -0.67 | yes |
| content_relevance | steering:layer12_scale6.0 | steering:v2_L12_a0.5_pn | 1.31 | 4.29 | 0.0000 | -1.00 | yes |
| fluency | baseline | finetuning | 4.72 | 3.32 | 0.0000 | +0.82 | yes |
| fluency | baseline | prompting:detailed | 4.72 | 3.40 | 0.0000 | +0.90 | yes |
| fluency | baseline | prompting:medium | 4.72 | 3.42 | 0.0000 | +1.00 | yes |
| fluency | baseline | prompting:minimal | 4.72 | 3.94 | 0.0007 | +0.72 | yes |
| fluency | baseline | steering:layer12_scale6.0 | 4.73 | 1.06 | 0.0000 | +1.00 | yes |
| fluency | baseline | steering:v2_L12_a0.5_pn | 4.72 | 3.76 | 0.0000 | +0.84 | yes |
| fluency | finetuning | prompting:detailed | 3.32 | 3.40 | 1.0000 | -0.03 | no |
| fluency | finetuning | prompting:medium | 3.32 | 3.42 | 1.0000 | +0.07 | no |
| fluency | finetuning | prompting:minimal | 3.32 | 3.94 | 0.0287 | -0.58 | yes |
| fluency | finetuning | steering:layer12_scale6.0 | 3.35 | 1.06 | 0.0000 | +1.00 | yes |
| fluency | finetuning | steering:v2_L12_a0.5_pn | 3.32 | 3.76 | 0.1841 | -0.39 | no |
| fluency | prompting:detailed | prompting:medium | 3.40 | 3.42 | 1.0000 | +0.00 | no |
| fluency | prompting:detailed | prompting:minimal | 3.40 | 3.94 | 0.0245 | -0.62 | yes |
| fluency | prompting:detailed | steering:layer12_scale6.0 | 3.43 | 1.06 | 0.0000 | +1.00 | yes |
| fluency | prompting:detailed | steering:v2_L12_a0.5_pn | 3.40 | 3.76 | 0.7122 | -0.37 | no |
| fluency | prompting:medium | prompting:minimal | 3.42 | 3.94 | 0.0378 | -0.67 | yes |
| fluency | prompting:medium | steering:layer12_scale6.0 | 3.45 | 1.06 | 0.0000 | +1.00 | yes |
| fluency | prompting:medium | steering:v2_L12_a0.5_pn | 3.42 | 3.76 | 0.5221 | -0.31 | no |
| fluency | prompting:minimal | steering:layer12_scale6.0 | 3.94 | 1.06 | 0.0000 | +1.00 | yes |
| fluency | prompting:minimal | steering:v2_L12_a0.5_pn | 3.94 | 3.76 | 1.0000 | +0.25 | no |
| fluency | steering:layer12_scale6.0 | steering:v2_L12_a0.5_pn | 1.06 | 3.78 | 0.0000 | -1.00 | yes |
| consistency | baseline | finetuning | 1.02 | 4.96 | 0.0000 | -1.00 | yes |
| consistency | baseline | prompting:detailed | 1.02 | 4.96 | 0.0000 | -1.00 | yes |
| consistency | baseline | prompting:medium | 1.02 | 4.80 | 0.0000 | -1.00 | yes |
| consistency | baseline | prompting:minimal | 1.02 | 4.04 | 0.0000 | -1.00 | yes |
| consistency | baseline | steering:layer12_scale6.0 | 1.02 | 1.49 | 0.0266 | -0.88 | yes |
| consistency | baseline | steering:v2_L12_a0.5_pn | 1.02 | 2.82 | 0.0001 | -1.00 | yes |
| consistency | finetuning | prompting:detailed | 4.96 | 4.96 | 1.0000 | +0.00 | no |
| consistency | finetuning | prompting:medium | 4.96 | 4.80 | 1.0000 | +0.43 | no |
| consistency | finetuning | prompting:minimal | 4.96 | 4.04 | 0.0151 | +0.88 | yes |
| consistency | finetuning | steering:layer12_scale6.0 | 4.96 | 1.49 | 0.0000 | +1.00 | yes |
| consistency | finetuning | steering:v2_L12_a0.5_pn | 4.96 | 2.82 | 0.0000 | +1.00 | yes |
| consistency | prompting:detailed | prompting:medium | 4.96 | 4.80 | 1.0000 | +0.67 | no |
| consistency | prompting:detailed | prompting:minimal | 4.96 | 4.04 | 0.0144 | +0.76 | yes |
| consistency | prompting:detailed | steering:layer12_scale6.0 | 4.98 | 1.49 | 0.0000 | +1.00 | yes |
| consistency | prompting:detailed | steering:v2_L12_a0.5_pn | 4.96 | 2.82 | 0.0000 | +1.00 | yes |
| consistency | prompting:medium | prompting:minimal | 4.80 | 4.04 | 0.0697 | +0.76 | no |
| consistency | prompting:medium | steering:layer12_scale6.0 | 4.86 | 1.49 | 0.0000 | +1.00 | yes |
| consistency | prompting:medium | steering:v2_L12_a0.5_pn | 4.80 | 2.82 | 0.0000 | +1.00 | yes |
| consistency | prompting:minimal | steering:layer12_scale6.0 | 4.02 | 1.49 | 0.0000 | +0.85 | yes |
| consistency | prompting:minimal | steering:v2_L12_a0.5_pn | 4.04 | 2.82 | 0.0119 | +0.70 | yes |
| consistency | steering:layer12_scale6.0 | steering:v2_L12_a0.5_pn | 1.49 | 2.86 | 0.0044 | -0.79 | yes |
| naturalness | baseline | finetuning | 1.00 | 4.34 | 0.0000 | -1.00 | yes |
| naturalness | baseline | prompting:detailed | 1.00 | 3.86 | 0.0000 | -1.00 | yes |
| naturalness | baseline | prompting:medium | 1.00 | 3.92 | 0.0000 | -1.00 | yes |
| naturalness | baseline | prompting:minimal | 1.00 | 2.34 | 0.0000 | -1.00 | yes |
| naturalness | baseline | steering:layer12_scale6.0 | 1.00 | 1.00 | 1.0000 | +0.00 | no |
| naturalness | baseline | steering:v2_L12_a0.5_pn | 1.00 | 1.63 | 0.0000 | -1.00 | yes |
| naturalness | finetuning | prompting:detailed | 4.34 | 3.86 | 0.1725 | +0.45 | no |
| naturalness | finetuning | prompting:medium | 4.34 | 3.92 | 0.4216 | +0.35 | no |
| naturalness | finetuning | prompting:minimal | 4.34 | 2.34 | 0.0000 | +1.00 | yes |
| naturalness | finetuning | steering:layer12_scale6.0 | 4.33 | 1.00 | 0.0000 | +1.00 | yes |
| naturalness | finetuning | steering:v2_L12_a0.5_pn | 4.33 | 1.63 | 0.0000 | +1.00 | yes |
| naturalness | prompting:detailed | prompting:medium | 3.86 | 3.92 | 1.0000 | -0.04 | no |
| naturalness | prompting:detailed | prompting:minimal | 3.86 | 2.34 | 0.0001 | +0.85 | yes |
| naturalness | prompting:detailed | steering:layer12_scale6.0 | 3.90 | 1.00 | 0.0000 | +1.00 | yes |
| naturalness | prompting:detailed | steering:v2_L12_a0.5_pn | 3.90 | 1.63 | 0.0000 | +0.95 | yes |
| naturalness | prompting:medium | prompting:minimal | 3.92 | 2.34 | 0.0000 | +0.94 | yes |
| naturalness | prompting:medium | steering:layer12_scale6.0 | 3.92 | 1.00 | 0.0000 | +1.00 | yes |
| naturalness | prompting:medium | steering:v2_L12_a0.5_pn | 3.92 | 1.63 | 0.0000 | +1.00 | yes |
| naturalness | prompting:minimal | steering:layer12_scale6.0 | 2.29 | 1.00 | 0.0000 | +1.00 | yes |
| naturalness | prompting:minimal | steering:v2_L12_a0.5_pn | 2.29 | 1.63 | 0.0003 | +0.93 | yes |
| naturalness | steering:layer12_scale6.0 | steering:v2_L12_a0.5_pn | 1.00 | 1.63 | 0.0000 | -1.00 | yes |

## Cost / effort comparison

| Method       | Data Prep        | Compute Cost     | Inference Overhead | Setup Complexity |
|--------------|------------------|------------------|--------------------|------------------|
| Baseline     | None             | ~$0              | None               | Trivial          |
| Prompting    | None             | ~$0              | Longer context     | Trivial          |
| Fine-tuning  | 636 gen. pairs   | ~0.35 GPU-h train| Same as base       | High             |
| Steering     | 247 gen. pairs   | ~0.3 GPU-h ext+sweep | Minimal (one add) | Medium         |
