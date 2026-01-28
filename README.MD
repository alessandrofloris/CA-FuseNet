# CA-FuseNet

Crowd-Aware FuseNet (CA-FuseNet) is a multimodal architecture for Human Activity Recognition (HAR).

## Quickstart

```bash
python -m pip install -e .
python -m ca_fusenet.scripts.train
```

## Configs

Hydra config lives in `src/ca_fusenet/configs`.
Override any field from the command line:

```bash
python -m ca_fusenet.scripts.train model=rgb_baseline
```

## Structure

- `src/ca_fusenet/scripts/train.py`: training entry point
- `src/ca_fusenet/configs`: Hydra config tree
- `src/ca_fusenet/utils`: shared utilities

# TODO 

- Creare una struttura di salvataggio degli artifacts dei modelli più strutturata
- Configurazione modello pose only tramite hydra
- Reintrodurre validazione sample opzionale dentro __getitem__
- Creare struttura di salvataggio migliore per il debug di estrazione di tublets
- Aggiungere padding alle tublets