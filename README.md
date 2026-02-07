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
| `23a9427` | Saturation + pattern chains | [20260206](https://cmpr.ai/hutter/archive/20260206/) |
| `160190b` | Bayes from table | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `11a7231` | Bayes patterns | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `60acfd9` | Spectral radius | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `db2b4ca` | SVD components | [20260131_4](https://cmpr.ai/hutter/archive/20260131_4/) |
| `51ef90b` | Memory depth | [20260131_4](https://cmpr.ai/hutter/archive/20260131_4/) |
| `21e719f` | Memory trace (PCA) | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
| `c9a60c0` | AC–RNN trace | [20260131_6](https://cmpr.ai/hutter/archive/20260131_6/) |
