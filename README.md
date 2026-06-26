# Primjena umjetne inteligencije za analitiku korisničkog ponašanja u kibernetičkoj sigurnosti

Praktični dio završnog rada — **detekcija insajderskih prijetnji** nad skupom podataka
**CERT Insider Threat r4.2**. Repozitorij sadrži tri zasebna, neovisno pokretljiva
modela koji isti problem rješavaju iz tri različita kuta: klasičnim strojnim učenjem,
vremenskim vjerojatnosnim modelom te dubokim učenjem.

## Tri implementirana modela

| Projekt | Model | Tip | ROC-AUC | Ključna ideja |
|---|---|---|---|---|
| [`uba_sustav/`](uba_sustav/) | Isolation Forest | nenadzirano, statičko | **0,948** | relativne (z-score) značajke po korisniku |
| [`uba_hmm/`](uba_hmm/) | HMM + NHE | nenadzirano, vremensko | **0,950** | po jedan Gaussov HMM po korisniku + rolling USB značajke |
| [`uba_transformer/`](uba_transformer/) | Transformer Encoder | nadzirano, vremensko | **0,994** | self-attention nad sljedovima ponašanja (PR-AUC 0,888) |

Sva tri modela koriste isti pristup značajkama: dnevne agregacije aktivnosti
(prijave, USB, datoteke, e-pošta, web) obogaćene **odstupanjima od vlastitog profila
korisnika** (z-score), kliznim statistikama (rolling, 7 dana) i indikatorima prvog
pojavljivanja. Taj relativni pristup pokazao se kao glavni doprinos rada.

## Struktura repozitorija

```
zavrsni-rad-uba/
├── uba_sustav/        # Klasični ML: Isolation Forest + relativne značajke
├── uba_hmm/           # HMM + NHE (hyper-binning), nenadzirano, GPU
├── uba_transformer/   # Transformer Encoder (PyTorch), GPU
└── README.md          # ova datoteka
```

Svaki projekt ima vlastiti `README.md` s detaljnim uputama, `src/` mapu s izvornim
kodom i glavnu skriptu (`main_*.py`) koja pokreće cijeli postupak.

## Skup podataka (nije uključen u repozitorij)

Modeli rade nad javnim skupom **CERT Insider Threat Test Dataset r4.2** (Carnegie
Mellon University, CERT Division / ExactData, pod pokroviteljstvom DARPA-e):

🔗 https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

Potrebne su datoteke `r4.2.tar.bz2` (logovi, nekoliko GB) i `answers.tar.bz2`
(ground truth). Podaci se **ne nalaze u ovom repozitoriju** zbog veličine
(`http.csv` je ~14,5 GB) i uvjeta korištenja. Nakon preuzimanja i raspakiravanja,
u `config.py` / `cert_full.py` svakog projekta postavi putanje `CERT_DIR` i
`ANSWERS_DIR` na lokaciju podataka.

Karakteristike skupa: 1000 korisnika, 70 insajdera, 330.285 (korisnik, dan) redaka,
1364 zlonamjerna dana (0,41 % — izrazito neuravnotežen problem).

## Pokretanje

Svaki projekt je samostalan. Tipičan tijek (primjer za Isolation Forest):

```bash
cd uba_sustav
pip install -r requirements.txt
python main_cert_full.py
```

Za HMM (`uba_hmm`) i Transformer (`uba_transformer`) pogledaj njihove `README.md`
datoteke. **Savjet za uštedu vremena:** prvo pokretanje čita cijeli `http` log
(~14 GB) i traje 10–30 min, ali se izračunate dnevne značajke spremaju u predmemoriju
(`data_cache/daily_features.parquet`). Tu predmemoriju možeš kopirati među projektima
da izbjegneš ponovno čitanje sirovih podataka.

## Programsko okruženje

- Python 3.11, Windows (NVIDIA GeForce RTX 3050)
- **PyTorch** za HMM i Transformer (rad na GPU). TensorFlow je napušten jer GPU ne
  radi na izvornom Windowsu za verzije ≥ 2.11 — sve je prebačeno na PyTorch.
- scikit-learn (Isolation Forest), pandas, numpy, matplotlib, pyarrow

---

*Repozitorij je izrađen kao praktični dio završnog rada na Fakultetu organizacije i
informatike (FOI).*
