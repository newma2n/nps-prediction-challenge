"""Étape 35 — captures d'écran de l'application, page par page.

    python scripts/captures.py          (l'application doit tourner sur localhost:8501)

Selenium + Chrome headless : on attend que le contenu soit réellement rendu (le rendu Streamlit
passe par un websocket, qu'une capture « à froid » ne voit pas), puis on capture la page entière.
Produit reports/captures/NN_<page>.png.
"""
from __future__ import annotations

import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "reports" / "captures"
PAGES = ["Vue d'ensemble", "Prioriser les appels", "Analyser un client", "Données",
         "Modèle et performance", "Équité", "Drivers de détraction", "Méthode", "Monitoring", "À propos"]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().replace("'", "").replace(" ", "_")


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    opts = Options()
    for a in ("--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1440,1000", "--force-device-scale-factor=1"):
        opts.add_argument(a)
    drv = webdriver.Chrome(options=opts)
    ok = 0
    try:
        for i, page in enumerate(PAGES, 1):
            drv.get(f"{URL}/?page={urllib.parse.quote(page)}")
            # Attendre le titre H1 de la page, puis laisser les images et tableaux arriver.
            WebDriverWait(drv, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
            WebDriverWait(drv, 60).until(lambda d: page.split()[0].lower() in d.find_element(By.CSS_SELECTOR, "h1").text.lower())
            time.sleep(4)
            hauteur = drv.execute_script("return document.documentElement.scrollHeight")
            drv.set_window_size(1440, min(max(hauteur, 1000), 6000))
            time.sleep(1.5)
            fichier = SORTIE / f"{i:02d}_{_slug(page)}.png"
            drv.save_screenshot(str(fichier))
            taille = fichier.stat().st_size
            ok += taille > 60_000
            print(f"{'OK  ' if taille > 60_000 else 'VIDE'} {fichier.name} ({taille // 1024} Ko, hauteur {hauteur}px)")
    finally:
        drv.quit()
    print(f"\n{ok}/{len(PAGES)} pages capturées -> {SORTIE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
