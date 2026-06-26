# UBA sustav – Detekcija insajderskih prijetnji pomoću umjetne inteligencije

Programsko rješenje praktičnog dijela završnog rada
*„Primjena umjetne inteligencije za analitiku korisničkog ponašanja u kibernetičkoj sigurnosti"*.

Sustav nad skupom podataka **CERT r4.2** izlučuje dnevne značajke ponašanja
korisnika te nenadziranim algoritmima strojnog učenja (Isolation Forest,
One-Class SVM, Local Outlier Factor) detektira anomalije koje upućuju na
insajderske prijetnje. Ključni doprinos je pristup **relativnih značajki po
korisniku** (z-score u odnosu na vlastiti profil + indikatori prvog pojavljivanja).

## Struktura projekta

```
uba_sustav/
├── src/
│   ├── cert_full.py              # učitavanje CERT-a + izlučivanje značajki (streaming)
│   ├── cert_features_baseline.py # relativne značajke po korisniku
│   ├── model.py                  # modeli za detekciju anomalija + evaluacija
│   └── visualize.py              # grafovi i tablice rezultata
├── main_cert_full.py             # GLAVNI pokretač (svi korisnici)
├── provjera.py                   # dijagnostika podataka i oznaka prije obrade
├── provjera_insajdera.py         # dokaz detekcije insajdera (precision@N)
├── results/                      # generirani grafovi i tablice metrika
└── requirements.txt
```

## Priprema

1. Instaliraj ovisnosti:
   ```
   pip install -r requirements.txt
   ```
2. Raspakiraj CERT r4.2 (`r4.2.tar.bz2`) i `answers.tar.bz2` (npr. na Desktop).
3. U `src/cert_full.py` na vrhu provjeri/prilagodi putanje:
   ```python
   CERT_DIR    = r"C:\Users\Domagoj\Desktop\r4.2"
   ANSWERS_DIR = r"C:\Users\Domagoj\Desktop\answers"
   ```

## Pokretanje

Preporučeni redoslijed:

1. Dijagnostika (provjeri da su podaci i oznake ispravno učitani):
   ```
   python provjera.py
   ```
   Mora pisati `Pronadjeno insajdera (r4.2): 70`.

2. Puna obrada (svi korisnici):
   ```
   python main_cert_full.py
   ```
   Prvo pokretanje čita cijeli `http` (~14 GB) i traje 10–30 min; rezultat
   (matrica značajki) kešira se u `data/cert_cache/`, pa su sljedeća pokretanja
   trenutna. Rezultati se spremaju u `results/` (sufiks `_cert_full`), uz
   tablice `metrike_apsolutne.csv` i `metrike_relativne.csv`.

3. Dokaz detekcije insajdera:
   ```
   python provjera_insajdera.py
   ```

## Napomene o CERT specifičnostima (kod ih već rješava)

- datoteke su CSV iako često bez `.csv` nastavka
- format datuma `MM/DD/YYYY HH:MM:SS`; interna domena `dtaa.com`
- `http` se čita u dijelovima (streaming) radi ograničene memorije
- oznake (ground truth) grade se iz `insiders.csv` (prozor `[start, end]` insajdera)
