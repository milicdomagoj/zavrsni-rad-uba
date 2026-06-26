# Transformer Encoder - PyTorch verzija (radi na GPU / RTX 3050)

## Zasto PyTorch
TensorFlow >= 2.11 NE podrzava GPU na obicnom (native) Windowsu - vidljivo iz
upozorenja "TensorFlow GPU support is not available on native Windows". PyTorch
podrzava GPU na Windowsu IZRAVNO, bez WSL2. Isti GPU vec je radio za NCDE model
(ispis je pokazivao "uredjaj=cuda"), pa ce i ovaj Transformer raditi na RTX 3050.

## Instalacija
1. Instaliraj PyTorch s GPU (CUDA) podrskom prema https://pytorch.org/get-started/locally/
   (odaberi Windows + Pip + CUDA; dobit ces naredbu poput:
    pip install torch --index-url https://download.pytorch.org/whl/cu121)
2. Ostalo:
   pip install pandas numpy scikit-learn matplotlib pyarrow

## Pokretanje
    python main_transformer_torch.py

Provjeri putanje u src/config.py (CERT_DIR, ANSWERS_DIR). Keš znacajki
(data_cache/daily_features.parquet) vec postoji pa se 14 GB preskace.

## "Vise podataka" - kako je rijeseno
U TF verziji trening je koristio mali poduzorkovani skup. PyTorch verzija prema
zadanim postavkama (USE_ALL_TRAIN=True u main_transformer_torch.py) trenira na
SVIM trening prozorima (~17000+), uz tezinski gubitak zbog neuravnotezenosti.
GPU to lako podnosi. Mozes mijenjati:
  - SEQ_LEN      (duljina prozora; 200 je bila najbolja)
  - EPOCHS       (broj epoha; vise = potencijalno bolje)
  - BATCH_SIZE   (256 je dobro za RTX 3050; smanji na 128 ako ponestane VRAM-a)
  - USE_ALL_TRAIN (True = svi podaci; False = poduzorkovanje s NEG_PER_POS)
  - USE_FOCAL    (True = focal loss umjesto tezinskog BCE)

## Provjera da koristi GPU
Na pocetku ispisa treba pisati:  [*] Uredjaj: GPU (NVIDIA GeForce RTX 3050)
Ako pise CPU, PyTorch nije instaliran s CUDA podrskom - reinstaliraj prema koraku 1.

## Rezultati
Spremaju se u results/ (metrics_torch.json, roc_curve_torch.png,
pr_curve_torch.png, confusion_matrix_torch.png, feature_importance_torch.png,
learning_curves_torch.png) i model u models/transformer_torch.pt.

## Napomena
Ako ti VRAM (4 GB na RTX 3050) bude tijesan kod SEQ_LEN=200 i velikog batcha,
smanji BATCH_SIZE na 128 ili 64. Trening na svim podacima na GPU-u i dalje je
bitno brzi nego TF na CPU-u.

## Stroza, realnija evaluacija (vazno!)
Ako rezultati izgledaju "predobro" (npr. PR-AUC ~1.0), to je zato sto je zadatak
s dugim prozorom (200 dana) i oznakom "bilo koji dan" prelagan. Zato su dodane
realnije postavke u main_transformer_torch.py:

  SEQ_LEN = 30         # kraci prozor = manje konteksta, tezi zadatak
  STRIDE = 7           # veci korak = manje skoro-duplikata medju prozorima
  LABEL_MODE = "last"  # prozor je insajder SAMO ako je zadnji dan zlonamjeran
                       # (predikcija "je li korisnik DANAS insajder")
  N_FOLDS = 5          # 5-struka unakrsna provjera po korisniku
                       # -> rezultat je PROSJEK +/- st.dev., a ne jedna podjela

Ovo daje NIZE ali POSTENIJE i pouzdanije rezultate, sto je za rad bolje
(rezultat blizu 1.0 djeluje sumnjivo; rezultat poput 0.85 +/- 0.05 je vjerodostojan).

Za laksu (visu) varijantu vrati: SEQ_LEN=200, LABEL_MODE="any", N_FOLDS=0.
