"""Étape 36 — monitoring implémenté (§ 4.9 de l'énoncé) : dérive des entrées (PSI), dérive des
prédictions, performance sur les nouvelles réponses d'enquête, et déclencheurs de réentraînement
(seuil de dérive, chute de performance, volume minimal de nouveaux labels, échéance).

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

    # Déclencheur de dérive
    alertes_sim = d_sim[d_sim["Statut"] == "ALERTE"]["Variable"].tolist()
    declenche = bool(alertes_sim) or psi_pred_sim > SEUIL_ALERTE or abs(part_det["mois_simule"] - part_det["reference"]) > 0.05

    # 4. Performance sur les NOUVELLES réponses d'enquête. Chaque mois, ~150 clients silencieux
    #    répondent (tirés selon la même propension biaisée S3 : les extrêmes répondent plus). On
    #    mesure kappa et rappel Détracteur sur chaque lot et en cumul, et on déclenche si la
    #    performance chute ou si le volume cumulé justifie un réentraînement.
    from sklearn.metrics import cohen_kappa_score, recall_score
    from .protocol import propensions
    from .target import ORDRE
    R = json.load(open(RACINE / "reports" / "resultats.json", encoding="utf-8"))
    kappa_ref = R["modele_final"]["metriques_retenues"]["kappa_quadratique"]
    rappel_ref = R["modele_final"]["metriques_retenues"]["par_classe"]["Detracteur"]["rappel"]
    p_rep = propensions(cur, cfg, "S3_MNAR", cfg["seed"])
    p_rep = p_rep / p_rep.sum()
    idx_tous = rng.choice(len(cur), size=150 * 6, replace=False, p=p_rep)
    pred_cur = np.asarray(art["pipeline_decision"].predict(cur[colonnes])).astype(str)
    y_cur = cur["cible_M3"].astype(str).values
    # L'équité se surveille au même titre que la performance. L'audit de la phase 5 a établi un
    # écart de rappel Détracteur entre tranches d'âge ; un écart mesuré une fois puis jamais repris
    # est un constat, pas un garde-fou. On recalcule donc le rappel par tranche d'âge sur les
    # réponses cumulées de chaque mois, et on lève une alerte dès que l'écart dépasse le seuil de
    # signalement fixé à l'étape 1 (config.yaml → criteres_succes.ecart_equite_signalement).
    ages_cur = cur["tranche_age"].astype(str).values
    SEUIL_EQUITE = float(cfg.get("criteres_succes", {}).get("ecart_equite_signalement", 0.10))
    MIN_DETRACTEURS_GROUPE = 10   # en dessous, le rappel d'un groupe n'est pas interprétable

    def _rappels_par_age(sel):
        pris = np.zeros(len(cur), dtype=bool); pris[np.asarray(sel, dtype=int)] = True
        out = {}
        for g in sorted(set(ages_cur)):
            m = pris & (ages_cur == g) & (y_cur == "Detracteur")
            if int(m.sum()) < MIN_DETRACTEURS_GROUPE:
                continue
            out[g] = {"detracteurs": int(m.sum()), "rappel": round(float((pred_cur[m] == "Detracteur").mean()), 4)}
        return out

    lots, cumul_idx = [], []
    MIN_LABELS, CHUTE_KAPPA, CHUTE_RAPPEL = 500, 0.05, 0.10
    for mois in range(6):
        idx = idx_tous[mois * 150:(mois + 1) * 150]
        cumul_idx = list(cumul_idx) + list(idx)
        k_lot = cohen_kappa_score(y_cur[idx], pred_cur[idx], weights="quadratic", labels=ORDRE)
        r_lot = recall_score(y_cur[idx], pred_cur[idx], labels=["Detracteur"], average="macro", zero_division=0)
        k_cum = cohen_kappa_score(y_cur[cumul_idx], pred_cur[cumul_idx], weights="quadratic", labels=ORDRE)
        r_cum = recall_score(y_cur[cumul_idx], pred_cur[cumul_idx], labels=["Detracteur"], average="macro", zero_division=0)
        eq = _rappels_par_age(cumul_idx)
        ecart = (round(max(v["rappel"] for v in eq.values()) - min(v["rappel"] for v in eq.values()), 4)
                 if len(eq) >= 2 else None)
        lots.append({"mois": mois + 1, "nouvelles_reponses": len(idx), "cumul": len(cumul_idx),
                     "kappa_lot": round(float(k_lot), 4), "rappel_detracteur_lot": round(float(r_lot), 4),
                     "kappa_cumul": round(float(k_cum), 4), "rappel_detracteur_cumul": round(float(r_cum), 4),
                     "rappel_detracteur_par_age_cumul": eq, "ecart_equite_age_cumul": ecart,
                     "groupes_suivis": len(eq),
                     "alerte_equite": bool(ecart is not None and ecart > SEUIL_EQUITE),
                     "chute_performance": bool(k_cum < kappa_ref - CHUTE_KAPPA or r_cum < rappel_ref - CHUTE_RAPPEL),
                     "volume_atteint": len(cumul_idx) >= MIN_LABELS})
    suivi = {"reference": {"kappa": round(kappa_ref, 4), "rappel_detracteur": round(rappel_ref, 4)},
             "seuils": {"min_nouveaux_labels": MIN_LABELS, "chute_kappa": CHUTE_KAPPA, "chute_rappel_detracteur": CHUTE_RAPPEL,
                        "echeance_max_mois": 6, "ecart_equite_age": SEUIL_EQUITE,
                        "min_detracteurs_par_groupe": MIN_DETRACTEURS_GROUPE},
             "lots": lots,
             "premier_mois_volume_atteint": next((l["mois"] for l in lots if l["volume_atteint"]), None),
             "chute_detectee": any(l["chute_performance"] for l in lots),
             "premier_mois_alerte_equite": next((l["mois"] for l in lots if l["alerte_equite"]), None),
             "alerte_equite_detectee": any(l["alerte_equite"] for l in lots),
             "note_equite": ("le rappel Détracteur est recalculé par tranche d'âge sur les réponses cumulées ; une alerte est levée "
                             "au-delà de " + f"{SEUIL_EQUITE:.0%}" + " d'écart entre le groupe le mieux et le moins bien servi. Un groupe comptant "
                             "moins de " + str(MIN_DETRACTEURS_GROUPE) + " détracteurs n'est pas évalué : le rappel y serait trop bruité pour décider."),
             "note": "réponses tirées parmi les silencieux selon la propension S3 (les extrêmes répondent plus) : la performance sur les nouveaux répondants est donc optimiste par rapport à la base — exactement le biais que le monitoring réel subira"}

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
        "suivi_nouvelles_reponses": suivi,
        # Les cinq déclencheurs de réentraînement : énoncés ici avec leur règle
        # ET leur état sur le mois simulé, pour qu'un lecteur puisse vérifier qu'ils se déclenchent.
        "declencheurs_reentrainement": [
            {"declencheur": "dérive des entrées", "regle": "PSI > " + str(SEUIL_ALERTE) + " sur au moins une variable du modèle",
             "etat_mois_simule": bool(alertes_sim)},
            {"declencheur": "chute de performance", "regle": "kappa cumulé < référence − 0,05 ou rappel Détracteur < référence − 0,10",
             "etat_mois_simule": bool(suivi["chute_detectee"])},
            {"declencheur": "volume de nouveaux labels", "regle": "500 nouvelles réponses d'enquête cumulées",
             "etat_mois_simule": suivi["premier_mois_volume_atteint"] is not None},
            {"declencheur": "équité", "regle": "écart de rappel Détracteur entre tranches d'âge > " + f"{suivi['seuils']['ecart_equite_age']:.0%}",
             "etat_mois_simule": bool(suivi["alerte_equite_detectee"])},
            {"declencheur": "échéance", "regle": "réentraînement au plus tard tous les 6 mois", "etat_mois_simule": True},
        ],
    }
    (RACINE / "reports").mkdir(exist_ok=True)
    with open(RACINE / "reports" / "monitoring.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"PSI prédictions courant={psi_pred:.3f} · mois simulé={psi_pred_sim:.3f} · "
          f"alertes simulées={alertes_sim} · réentraînement déclenché={declenche} · "
          f"volume de labels atteint au mois {suivi['premier_mois_volume_atteint']} · chute détectée={suivi['chute_detectee']} · "
          f"alerte équité={suivi['alerte_equite_detectee']} (mois {suivi['premier_mois_alerte_equite']})")
    return res


if __name__ == "__main__":
    main()
