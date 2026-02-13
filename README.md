# Hutter Prize Compressor — Reproducible Experiments

Companion repository for https://cmpr.ai/hutter/

Each experiment is one commit with self-contained source code.
Between experiments the repo is cleaned to a blank slate.

## To reproduce an experiment

```
git log --oneline              # find the experiment
git checkout <commit>          # check out that commit
bash reproduce.sh              # run it
```

Pre-trained models and papers are published at
https://cmpr.ai/hutter/archive/ — each experiment's commit message
includes the relevant archive URL.

## Data

Most experiments use enwik9: http://mattmahoney.net/dc/enwik9.zip

## Experiments

| Commit | Experiment | Archive |
|--------|-----------|---------|
| `89b0664` | KN scaling: byte KN to 1.784 bpc on full enwik9, zero structure | [20260212](https://cmpr.ai/hutter/archive/20260212/) |
| `efbaac3` | Weight construction + ES isomorphism: all 82k params from data, 1.89 bpc | [20260211](https://cmpr.ai/hutter/archive/20260211/) |
| `86fbef8` | Total interpretation: 128-bit Boolean automaton (Q1-Q7), 11 tools | [20260211](https://cmpr.ai/hutter/archive/20260211/) |
| `74ddb51` | Pattern chains: factor map, reverse isomorphism, viewer | [20260208](https://cmpr.ai/hutter/archive/20260208/) |
| `f808449` | Pattern priors, skip-patterns, weight construction | [20260207](https://cmpr.ai/hutter/archive/20260207/) |
| `8ce5e5f` | SN visibility (full W_hh + UM comparison + calibration) | [20260207](https://cmpr.ai/hutter/archive/20260207/) |
| `23a9427` | Saturation + pattern chains | [20260206](https://cmpr.ai/hutter/archive/20260206/) |
| `160190b` | Bayes from table | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `11a7231` | Bayes patterns | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `60acfd9` | Spectral radius | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `db2b4ca` | SVD components | [20260131_4](https://cmpr.ai/hutter/archive/20260131_4/) |
| `51ef90b` | Memory depth | [20260131_4](https://cmpr.ai/hutter/archive/20260131_4/) |
| `21e719f` | Memory trace (PCA) | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `c9a60c0` | AC–RNN trace | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
