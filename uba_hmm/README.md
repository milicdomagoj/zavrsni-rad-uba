# Nenadzirana detekcija insajdera: HMM + NHE (hyper-binning)

Nenadzirani pristup detekciji insajderskih prijetnji nad CERT r4.2:
  1. Za SVAKOG korisnika uci se zaseban Gaussov skriveni Markovljev model (HMM)
     koji modelira NJEGOVO normalno ponasanje kroz vrijeme.
  2. Po danu se racuna uvjetna log-vjerojatnost pod tim modelom -- niska
     vrijednost znaci da je dan iznenadjujuc (potencijalna anomalija).
  3. Log-vjerojatnosti se metodom Normalized Histogram Entropy (NHE) +
     hyper-binning pretvaraju u anomalijski skor.

Oznake (tko je insajder) NE koriste se pri ucenju ni bodovanju -- samo na kraju
za mjerenje uspjesnosti. To je pravo NENADZIRANO ucenje.

## Radi na grafickoj (GPU)
HMM je implementiran u PyTorchu i batchan PREKO SVIH KORISNIKA, pa se uci
paralelno na GPU-u (cuda) ako je dostupan. Napomena: HMM po korisniku je malen
model, pa je dobitak od GPU-a skromniji nego kod Transformera, ali kod koristi
grafiku i brz je.

## Struktura
```
uba_hmm/
├── src/
│   ├── config.py             # putanje i postavke
│   ├── data_loading.py       # ucitavanje CERT-a
│   ├── feature_engineering.py# dnevne znacajke (+ cache)
│   ├── hmm_torch.py          # batchani Gaussov HMM (PyTorch, GPU)
│   └── nhe_scoring.py        # NHE + hyper-binning bodovanje
├── main_hmm_nhe.py           # glavni pokretac
└── results/
```

## Priprema
```
pip install torch pandas numpy scikit-learn pyarrow
```
U src/config.py provjeri CERT_DIR i ANSWERS_DIR.

VAZNO (usteda vremena): ako vec imas izracunat cache iz Transformer projekta,
kopiraj ga da se izbjegne ponovno citanje 14 GB:
  kopiraj  uba_transformer\data_cache\daily_features.parquet
  u        uba_hmm\data_cache\daily_features.parquet

## Pokretanje
```
python main_hmm_nhe.py
```

## Postavke (vrh main_hmm_nhe.py)
  N_STATES = 4     # broj skrivenih stanja HMM-a
  EM_ITERS = 25    # iteracija Baum-Welch ucenja
  N_BINS   = 256   # finoca hyper-binninga
  HMM_FEATURES     # popis znacajki za HMM
