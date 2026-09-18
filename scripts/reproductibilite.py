"""Étape 38 — test à blanc : le pipeline rejoué depuis une copie vierge donne-t-il les mêmes chiffres ?

    python scripts/reproductibilite.py

Copie `src/`, `prompts/`, `config.yaml` et `data/raw/` seuls dans un répertoire temporaire, y exécute
`python -m src.run` (~20 min), puis compare les indicateurs clés à ceux du dépôt. Écrit
reports/reproductibilite.json, lu par l'étude (phase 5 § 9) et par l'application.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _indicateurs(R: dict) -> dict:
    return {
        "NPS M1": R["arbitrage_des_3"]["nps"]["M1"],
        "NPS M3": R["arbitrage_des_3"]["nps"]["M3"],
        "part des 3 -> détracteur": R["arbitrage_des_3"]["part_3_vers_detracteur"],
        "modèle retenu": R["selection_modele_final"]["retenu"],
        "version retenue": R["selection_modele_final"]["version_retenue"],
        "kappa final (silencieux)": round(R["modele_final"]["metriques_retenues"]["kappa_quadratique"], 6),
        "macro-F1 final": round(R["modele_final"]["metriques_retenues"]["macro_f1"], 6),
        "précision@K": R["evaluation_metier"]["precision_at_k"]["precision_at_k"],
        "NPS silencieux prédit": R["nps_simule"]["nps_silencieux_predit"],
        "écart d'équité (âge)": R["audit_equite"]["tranche_age"]["ecart_max"],
    }


def main():
    original = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="nps_repro_"))
    for d in ("src", "prompts"):
        shutil.copytree(RACINE / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(RACINE / "config.yaml", tmp / "config.yaml")
    shutil.copytree(RACINE / "data" / "raw", tmp / "data" / "raw")
    print(f"Copie vierge : {tmp}")

    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "src.run"], cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace")
    duree = round(time.time() - t0, 1)
    res = {"duree_s": duree, "retour": proc.returncode, "repertoire": str(tmp)}
    p_res = tmp / "reports" / "resultats.json"
    if proc.returncode == 0 and p_res.exists():
        copie = json.load(open(p_res, encoding="utf-8"))
        io, ic = _indicateurs(original), _indicateurs(copie)
        comparaison = [{"indicateur": k, "original": io[k], "copie": ic[k], "identique": io[k] == ic[k]} for k in io]
        res["comparaison"] = comparaison
        res["reproductible"] = all(c["identique"] for c in comparaison)
        for c in comparaison:
            print(f"{'OK ' if c['identique'] else 'DIF'} {c['indicateur']:28s} {c['original']!s:>22s} {c['copie']!s:>22s}")
        print(f"\n{'Reproductible à l’identique' if res['reproductible'] else 'ÉCARTS constatés'} — {duree} s")
    else:
        res["erreur_fin"] = proc.stderr[-1500:]
        print("Échec du pipeline sur la copie vierge :", proc.stderr[-600:])
    with open(RACINE / "reports" / "reproductibilite.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
