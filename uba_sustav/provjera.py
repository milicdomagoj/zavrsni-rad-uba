import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src import cert_full as cf
import pandas as pd

print("=" * 60)
print(" DIJAGNOSTIKA CERT PODATAKA")
print("=" * 60)

print(f"\nCERT_DIR    = {cf.CERT_DIR}")
print(f"ANSWERS_DIR = {cf.ANSWERS_DIR}")

print("\n[1] Provjera log datoteka i njihovih stupaca:")
for base in ["logon", "device", "http", "email", "file"]:
    try:
        path = cf._find_file(base)
        cols = cf._read_header(base)
        size_mb = os.path.getsize(path) / 1e6
        print(f"    {base:7s} OK  ({size_mb:8.1f} MB)  stupci: {cols}")
    except Exception as e:
        print(f"    {base:7s} GRESKA: {e}")

print("\n[2] Provjera insiders.csv (ground truth):")
try:
    ins = cf.load_insiders()
    print(f"    Pronadjeno insajdera (r4.2): {ins['user'].nunique()}")
    print(f"    Ukupno prozora aktivnosti:   {len(ins)}")
    if len(ins):
        print("\n    Prvih 5 insajdera:")
        print(ins.head().to_string(index=False))
        print(f"\n    Raspon datuma: {ins['start'].min()} -> {ins['end'].max()}")
    else:
        print("    UPOZORENJE: 0 insajdera ucitano - provjeri format insiders.csv!")
except Exception as e:
    print(f"    GRESKA: {e}")
    print(f"\n    Sadrzaj {cf.ANSWERS_DIR}:")
    for root, dirs, files in os.walk(cf.ANSWERS_DIR):
        for f in files[:20]:
            print(f"      {os.path.join(root, f)}")

print("\n" + "=" * 60)
print(" Ako su insajderi pronadjeni (broj > 0), mozes pokrenuti:")
print("   python main_cert_full.py")
print("=" * 60)
