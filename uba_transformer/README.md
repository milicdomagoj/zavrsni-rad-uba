# Insider Threat Detection — Transformer Encoder (CERT r4.2)

Produkcijsko rjesenje za detekciju insajderskih prijetnji nad CERT r4.2 skupom
podataka koristeci **Transformer Encoder** (multi-head self-attention) nad
vremenskim sljedovima ponasanja korisnika. Pripremljeno za istrazivacki rad / zavrsni rad.

## Mogucnosti
- Potpuna EDA svih log datoteka (stupci, tipovi, praznine, raspodjela klasa)
- Bogati feature engineering (54+ znacajki): osnovne dnevne agregacije,
  z-score po korisniku (baseline odstupanja), rolling statistike (7 dana),
  novelty indikatori, vrijeme od zadnje aktivnosti, frekvencija aktivnosti
- Generiranje vremenskih sljedova s automatskim odabirom duljine (30/50/100/200)
- Obrada neuravnotezenosti: class weighting + Focal Loss (+ poduzorkovanje negativa)
- Transformer Encoder: pozicijsko kodiranje, multi-head self-attention, residual
  veze, layer norm, feed-forward, dropout
- Optuna optimizacija hiperparametara (slojevi, glave, dimenzije, dropout, LR)
- Trening: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, gradient clipping,
  mixed precision (ubrzava na NVIDIA RTX)
- Evaluacija: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, konfuzija,
  ROC/PR krivulje, krivulje ucenja
- Interpretabilnost: permutacijska vaznost znacajki + analiza tezina paznje
  (opcionalno SHAP, vidi nize)

## Struktura
```
uba_transformer/
├── src/
│   ├── config.py               # sve postavke i putanje
│   ├── data_loading.py         # ucitavanje + EDA
│   ├── feature_engineering.py  # dnevne + napredne ponasajne znacajke
│   ├── sequences.py            # sljedovi, podjela po korisniku, skaliranje, imbalance
│   ├── transformer_model.py    # Transformer Encoder + Focal Loss
│   ├── training.py             # odabir sekvence, Optuna, trening, evaluacija, interpretabilnost
│   └── visualization.py        # svi grafovi
├── main_transformer.py         # glavni pokretac cijelog pipelinea
├── notebooks/pipeline.ipynb    # isti pipeline u modularnom notebooku
├── results/  models/  data_cache/
└── requirements.txt
```

## Priprema
```
pip install -r requirements.txt
```
Za GPU na Windowsu instaliraj TensorFlow s CUDA podrskom prema sluzbenim uputama
(npr. preko WSL2 ili `pip install tensorflow[and-cuda]`). RTX 3050 podrzava
mixed precision (ukljuceno u config.py).

U `src/config.py` provjeri `CERT_DIR` i `ANSWERS_DIR`.

## Pokretanje
```
python main_transformer.py
```
Prvo pokretanje gradi dnevne znacajke iz sirovih CERT podataka (cita http ~14 GB)
i kesira ih u `data_cache/daily_features.parquet`. Rezultati i grafovi idu u
`results/`, model u `models/`.

## Vazne postavke (config.py)
- `SEQ_LENGTHS`, `STRIDE` — duljine prozora i korak
- `OPTUNA_TRIALS`, `OPTUNA_EPOCHS` — opseg pretrage hiperparametara
- `MAX_EPOCHS`, `BATCH_SIZE` — finalni trening
- `USE_MIXED_PRECISION` — ubrzanje na NVIDIA GPU

## Napomena o interpretabilnosti (SHAP)
Primarna, pouzdana metoda vaznosti je **permutacijska vaznost** (model-agnosticna,
ukljucena i testirana). SHAP za Transformer nad sekvencama je racunski skup i
osjetljiv na verzije; ako ga zelis, koristi `shap.GradientExplainer` na uzorku
sljedova. Permutacijska vaznost daje robusne i lako objasnjive rezultate.

## Napomena o ocjenjivanju
Trening/val/test dijele se PO KORISNIKU (isti korisnik nije u dva skupa), sto
mjeri generalizaciju na nove korisnike i daje posten, stroziji rezultat.
