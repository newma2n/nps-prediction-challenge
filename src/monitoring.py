"""Étape 36 — monitoring implémenté : dérive des entrées (PSI), dérive des prédictions,
et démonstration du déclencheur sur un mois simulé.

    python -m src.monitoring

Référence = population d'entraînement (répondants S3). Courant = population scorée.
Un troisième jeu, « mois simulé », perturbe les données pour montrer l'alerte se déclencher.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import RACINE, charger_config

SEUIL_ALERTE, SEUIL_VIGILANCE = 0.20, 0.10


def psi(reference: pd.Series, courant: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. < 0,10 stable · 0,10–0,20 vigilance · > 0,20 alerte."""
    if pd.api.types.is_numeric_dtype(reference) and reference.nunique() > bins:
        bords = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
        bords[0], bords[-1] = -np.inf, np.inf
        r = pd.cut(reference, bords).value_counts(normalize=True, sort=False)
        c = pd.cut(courant, bords).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(reference.dropna().astype(str)) | set(courant.dropna().astype(str)))
        r = reference.astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
        c = courant.astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
    r, c = r.values + 1e-6, c.values + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def statut(v: float) -> str:
    return "ALERTE" if v > SEUIL_ALERTE else ("vigilance" if v > SEUIL_VIGILANCE else "stable")


def rapport_derive(ref: pd.DataFrame, cur: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    lignes = [{"Variable": c, "PSI": psi(ref[c], cur[c])} for c in colonnes if c in ref and c in cur]
    d = pd.DataFrame(lignes).sort_values("PSI", ascending=False)
    d["Statut"] = d["PSI"].map(statut)
    return d


def main():
    cfg = charger_config()
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    df = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    num, cat, pipe = art["num"], art["cat"], art["pipeline"]
    i_det = art["classes"].index("Detracteur")
    colonnes = num + cat

    ref = df[df["repondant_S3"]]
    cur = df[~df["repondant_S3"]]

    # 1. Dérive des entrées : référence vs population scorée (attendue faible — même base)
    d_entrees = rapport_derive(ref, cur, colonnes)

    # 2. Dérive des prédictions
    p_ref = pipe.predict_proba(ref[colonnes])[:, i_det]
    p_cur = pipe.predict_proba(cur[colonnes])[:, i_det]
    psi_pred = psi(pd.Series(p_ref), pd.Series(p_cur))

    # 3. Mois simulé : hausse tarifaire de 12 %, migration vers le mensuel, plus de fibre.
    rng = np.random.default_rng(cfg["seed"])
    sim = cur.copy()
    sim["Monthly Charge"] = sim["Monthly Charge"] * 1.12
    bascule = rng.random(len(sim)) < 0.15
    sim.loc[bascule, "Contract"] = "Month-to-Month"
    sim.loc[rng.random(len(sim)) < 0.10, "Internet Type"] = "Fiber Optic"
    sim["charge_par_service"] = sim["Monthly Charge"] / sim["nb_services"].replace(0, np.nan)
    d_sim = rapport_derive(ref, sim, colonnes)
    p_sim = pipe.predict_proba(sim[colonnes])[:, i_det]
    psi_pred_sim = psi(pd.Series(p_ref), pd.Series(p_sim))
    part_det = {"reference": float((p_ref > 0.5).mean()), "courant": float((p_cur > 0.5).mean()),
                "mois_simule": float((p_sim > 0.5).mean())}

    # Déclencheur
    alertes_sim = d_sim[d_sim["Statut"] == "ALERTE"]["Variable"].tolist()
    declenche = bool(alertes_sim) or psi_pred_sim > SEUIL_ALERTE or abs(part_det["mois_simule"] - part_det["reference"]) > 0.05

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    top = d_sim.head(12).iloc[::-1]
    axes[0].barh(top["Variable"], top["PSI"], color=["#c0392b" if s == "ALERTE" else "#e67e22" if s == "vigilance" else "#27ae60" for s in top["Statut"]])
    axes[0].axvline(SEUIL_VIGILANCE, ls=":", color="k", lw=0.7); axes[0].axvline(SEUIL_ALERTE, ls="--", color="k", lw=0.7)
    axes[0].set_title("PSI par variable\nmois simulé vs référence", fontsize=9); axes[0].set_xlabel("PSI")
    for p_, lab, col in ((p_ref, "référence (entraînement)", "#95a5a6"), (p_cur, "population scorée", "#2c7fb8"), (p_sim, "mois simulé", "#c0392b")):
        axes[1].hist(p_, bins=30, alpha=0.55, label=lab, color=col, density=True)
    axes[1].set_title(f"Distribution de P(Détracteur)\nPSI courant {psi_pred:.3f} · mois simulé {psi_pred_sim:.3f}", fontsize=9)
    axes[1].legend(fontsize=7); axes[1].set_xlabel("P(Détracteur)")
    plt.tight_layout()
    fig_dir = RACINE / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "06_monitoring.png", dpi=130, bbox_inches="tight"); plt.close()

    res = {
        "seuils": {"vigilance": SEUIL_VIGILANCE, "alerte": SEUIL_ALERTE, "variation_part_detracteurs": 0.05},
        "derive_entrees_courant": d_entrees.round(4).to_dict(orient="records"),
        "psi_predictions_courant": round(psi_pred, 4),
        "derive_entrees_mois_simule": d_sim.round(4).to_dict(orient="records"),
        "psi_predictions_mois_simule": round(psi_pred_sim, 4),
        "part_detracteurs_predits": {k: round(v, 4) for k, v in part_det.items()},
        "variables_en_alerte_mois_simule": alertes_sim,
        "reentrainement_declenche_mois_simule": declenche,
        "scenario_simule": "tarif +12 %, 15 % des contrats basculés en mensuel, 10 % migrés vers la fibre",
    }
    (RACINE / "reports").mkdir(exist_ok=True)
    with open(RACINE / "reports" / "monitoring.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"PSI prédictions courant={psi_pred:.3f} · mois simulé={psi_pred_sim:.3f} · "
          f"alertes simulées={alertes_sim} · réentraînement déclenché={declenche}")
    return res


if __name__ == "__main__":
    main()
