## MODELS

- CGFLLM: Causal Graph Fuzzy LLM (MISO)
- MCGFLLM: Multivariate Causal Graph Fuzzy LLM (MIMO)
- CGFuzzy: Causal Graph Fuzzy (MISO)
- MCGFuzzy: Multivariate Causal Graph Fuzzy (MIMO)


## DATASETS

| Dataset | Description                        | Target Variable        | Samples  | Number of Variables | Granularity |
|:-------:|------------------------------------|:----------------------:|:--------:|:--------------:|:-----------:|
| SONDA   | Radiation                          | glo_avg                | 35.000   | 12             | 1 min       |
| WEC     | Wind Energy Production             | power                  | 43.802   | 9              | 1 h         |
| DEC     | Domestic Electricity Consumption   | active_power           | 100.000  | 14             | 1 min       |
| AQ      | Air Quality                        | PM2.5                  | 35.052   | 10             | 1 h         |
| WTH     | Weather                            | WetBulbFarenheit       | 35.065   | 10             | 1 h         |
| ETT     | Electricity Transformer Temperature| OT                     | 69.680   | 7              | 15 min      |


## EXPERIMENTS

| Model          | Description                               | Database               | 
|:--------------:|-------------------------------------------|:----------------------:|
| CGFuzzy        | Causal Graph Fuzzy (MISO)                 | CGFuzzy.db             |
| MCGFuzzy       | Multivariate Causal Graph Fuzzy (MIMO)    | MCGFuzzy.db            |
| CGFLLM         | Causal Graph Fuzzy LLM (MISO)             | CGFuzzy.db             |
| MCGFLLM        | Multivariate Causal Graph Fuzzy LLM (MIMO)| MCGFuzzy.db            |
| CGFLLM Freeze  | Causal Graph Fuzzy LLM (MISO)             | CGFuzzy_Freeze.db      |
| MCGFLLM Freeze | Multivariate Causal Graph Fuzzy LLM (MIMO)| MCGFuzzy_Freeze.db     |
