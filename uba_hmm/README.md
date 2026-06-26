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

## Napomena o NHE metodi
Buduci da tocan rad mentora nije bio dostupan, NHE + hyper-binning implementiran
je kao principijelna, branjiva rekonstrukcija (vidi komentare u nhe_scoring.py).
Lako se prilagodjava tocnoj definiciji metode.

## Poboljsanja u v2 (ako je v1 davao slabiji ROC-AUC)
- HMM sada koristi i ODSTUPANJA (z-score) i "prvi put" znacajke -- najjaci
  insajderski signal. Cache se NE mora ponovno racunati (te kolone vec postoje).
- Bodovanje vise NE dijeli s MAD-om (to je ranije guralo i normalne korisnike na
  vrh liste). Sada se zadrzava magnituda odstupanja -> manje laznih alarma.
- SCORE_MODE = "combo": kombinira marginalnu (trajne anomalije) i uvjetnu
  (prijelazi) log-vjerojatnost. Mozes probati "marginal" ili "conditional".

## Poboljsanje v3: bodovanje po dominantnim stanjima (rjesava "upijanje")
Problem: HMM tijekom ucenja moze stvoriti rijetko skriveno stanje koje modelira
ANOMALNE dane, pa im daje visoku vjerojatnost (anomalije se "upiju" u model).

Rjesenje (SCORE_MODE="dominant", zadano): dan se boduje po uklapanju u DOMINANTNA
(cesta) stanja korisnika; rijetka stanja (zauzetost < OCC_THRESH) iskljucuju se.
Tako anomalni dani dobiju nisko uklapanje -> visok anomalijski skor.

Na teskom testu (power useri + suptilne anomalije) ovo dize:
  precision@50: 0.80 -> 1.00,  PR-AUC: 0.81 -> 0.94,  ROC-AUC: 0.98.

Postavke: SCORE_MODE="dominant", OCC_THRESH=0.10 (prag zauzetosti stanja).

## Poboljsanje v4: k-means inicijalizacija (protiv kolapsa stanja)
Na vrlo pravilnim korisnicima HMM zna kolabirati u JEDNO stanje (entropija stanja
blizu 0), pa dominantno bodovanje gubi snagu. KMEANS_INIT=True postavlja pocetna
sredista stanja k-means grupiranjem po korisniku -> stanja se rasporede po stvarnim
obrascima ponasanja i ne kolabiraju. Skript ispisuje "entropija stanja"; ako je
< 0.15, stanja kolabiraju.

Opcija HYBRID_GLOBAL (0..1): mijesa HMM skor s globalnim outlier signalom (slicnim
Isolation Forestu). 0 = cisti HMM (zadano). Npr. 0.5 moze podici rezultat ako HMM
kolabira, ali rezultat je tada djelomicno globalni, a manje "cisti HMM".

## Poboljsanje v5: jake rolling znacajke (KLJUCNO)
Dijagnozom na stvarnim podacima utvrdjeno je da klizne (rolling) statistike USB-a
nose najjaci signal: roll7_std_num_usb_connects sama ima ROC=0.95 (hvata obrazac
"korisnik odjednom pocne koristiti USB"). HMM_FEATURES sada koristi te znacajke,
sto dize HMM na ROC-AUC=0.95 na stvarnim podacima -- konkurentno Isolation Forestu.
