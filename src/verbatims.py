"""Étape 13 — verbatims synthétiques (§ 4.4 de l'énoncé).

    python -m src.verbatims            # génère data/processed/verbatims.parquet

Deux chemins, même schéma de sortie :

  1. API Anthropic (Claude) — activé si ANTHROPIC_API_KEY est présent dans l'environnement ou
     dans un fichier .env à la racine. Prompt versionné dans prompts/verbatim_v1.txt, appels par
     lots, reprise sur incident, identifiant de modèle épinglé. La Messages API n'expose ni seed
     ni (sur les modèles actuels) temperature : la reproductibilité vient du COMMIT des textes.

  2. Générateur local à gabarits stochastiques — utilisé sinon. Les fragments ont été rédigés par
     un LLM (Claude) ; l'assemblage est local, seedé, donc strictement reproductible. La colonne
     `source` dit lequel des deux a produit chaque ligne. Ne pas présenter ce chemin comme un
     appel API : il ne l'est pas.

Conditionnement (commun aux deux chemins) : tonalité tirée de la classe M3, avec un BRUIT de 25 %
(tonalité redistribuée au hasard) et des cas contre-intuitifs, comme l'énoncé le demande.
Features utilisées : contrat, ancienneté, type d'internet, offre, parrainage, charge mensuelle.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RACINE, charger_config

MODELE_LLM = "claude-opus-5"
TAUX_BRUIT = 0.25
TON = {"Detracteur": "frustré", "Passif": "neutre", "Promoteur": "enthousiaste"}


def _charger_env():
    p = RACINE / ".env"
    if p.exists():
        for ligne in p.read_text(encoding="utf-8").splitlines():
            if "=" in ligne and not ligne.strip().startswith("#"):
                k, v = ligne.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _profil(r: pd.Series) -> dict:
    return {
        "contrat": str(r["Contract"]), "anciennete_mois": int(r["Tenure in Months"]),
        "internet": str(r["Internet Type"]), "offre": str(r["Offer"]),
        "parrainages": int(r["Number of Referrals"]), "facture_mensuelle": round(float(r["Monthly Charge"]), 1),
        "support_premium": str(r["Premium Tech Support"]), "streaming": str(r["Streaming TV"]),
    }


def tonalites(df: pd.DataFrame, cible: pd.Series, seed: int) -> pd.Series:
    """Tonalité visée : la classe, redistribuée au hasard pour 25 % des clients."""
    rng = np.random.default_rng(seed)
    t = cible.map(TON).astype(object).values.copy()
    bruit = rng.random(len(t)) < TAUX_BRUIT
    t[bruit] = rng.choice(list(TON.values()), size=bruit.sum())
    return pd.Series(t, index=df.index)


# ----------------------------------------------------------------------------------
# Chemin 2 — gabarits stochastiques (fragments rédigés par un LLM, assemblage local seedé)
# ----------------------------------------------------------------------------------
FRAGMENTS = {
    "frustré": {
        "ouverture": [
            "Troisième appel ce mois-ci pour le même problème.", "Encore une coupure hier soir, en pleine visio.",
            "Je viens de recevoir ma facture et je ne comprends pas le montant.", "Le débit annoncé n'a rien à voir avec ce que j'ai réellement.",
            "J'ai attendu 40 minutes avant d'avoir quelqu'un.", "Le technicien devait passer mardi, personne n'est venu.",
            "Franchement, je suis fatigué de devoir relancer.", "Ça fait des semaines que je signale ce souci.",
            "Je paie {facture} $ par mois pour un service qui ne suit pas.", "On m'avait promis une remise, je ne la vois nulle part.",
        ],
        "corps": [
            "Le conseiller a été correct mais n'a rien pu résoudre.", "On m'a transféré trois fois sans réponse claire.",
            "Je n'ai toujours pas d'explication sur les {facture} $ facturés.", "L'offre {offre} qu'on m'a proposée n'a rien changé au fond du problème.",
            "En {anciennete} mois chez vous, c'est la première fois que je songe sérieusement à partir.", "Mes voisins n'ont pas ces soucis avec la concurrence.",
            "La {internet} est censée être stable, ce n'est pas mon expérience.", "J'ai dû tout réexpliquer depuis le début.",
        ],
        "cloture": [
            "Je regarde ce que proposent les autres opérateurs.", "Si ce n'est pas réglé sous quinze jours, je résilie.",
            "J'attends un retour écrit, pas une promesse au téléphone.", "Je demande un geste commercial à la hauteur.",
            "Mon contrat {contrat} me laisse libre de partir, et j'y pense.", "",
        ],
    },
    "neutre": {
        "ouverture": [
            "Appel pour une question de facturation.", "Demande d'information sur les options disponibles.",
            "Le client souhaite modifier son mode de paiement.", "Je voulais vérifier la date de fin de mon engagement.",
            "Question sur l'activation de l'option streaming.", "Le client signale un ralentissement ponctuel, résolu depuis.",
            "Rien de grave, juste une vérification de mon forfait.", "Je cherche à comprendre ce que couvre l'offre {offre}.",
        ],
        "corps": [
            "Réponse obtenue, pas d'action supplémentaire.", "Le conseiller a expliqué le détail des {facture} $ mensuels.",
            "Ça fonctionne, sans plus ; je n'ai pas de plainte particulière.", "La {internet} fait le travail pour mon usage.",
            "Client depuis {anciennete} mois, pas de changement notable.", "Le service est dans la moyenne de ce que j'ai connu.",
            "Je n'ai pas encore testé le support premium.", "On verra à l'échéance du contrat {contrat}.",
        ],
        "cloture": [
            "Merci pour les précisions.", "Je réfléchis avant de changer quoi que ce soit.", "Dossier clos.",
            "Rappel prévu si besoin.", "", "",
        ],
    },
    "enthousiaste": {
        "ouverture": [
            "Juste un mot pour dire que le problème a été réglé en dix minutes.", "Installation faite ce matin, tout marche parfaitement.",
            "Très bonne surprise avec l'offre {offre}, merci.", "Le conseiller a été d'une patience remarquable.",
            "Après {anciennete} mois, toujours aussi satisfait.", "La {internet} est stable depuis le premier jour.",
            "J'ai recommandé le service à {parrainages} proches, c'est dire.", "Le support premium vaut vraiment son prix.",
        ],
        "corps": [
            "Le débit est au rendez-vous, même en soirée.", "La facture de {facture} $ est claire et sans surprise.",
            "On sent que les équipes connaissent leur sujet.", "Le passage au contrat {contrat} s'est fait sans accroc.",
            "Le streaming ne coupe jamais, ma famille est ravie.", "J'ai comparé, je ne trouve pas mieux ailleurs.",
            "Petite réserve sur le temps d'attente, mais le résultat est là.", "Une seule intervention en {anciennete} mois, et réussie.",
        ],
        "cloture": [
            "Continuez comme ça.", "Je resterai chez vous sans hésiter.", "Bravo à l'équipe.", "Merci encore.", "", "",
        ],
    },
}


def _gabarit(rng: np.random.Generator, ton: str, p: dict) -> str:
    f = FRAGMENTS[ton]
    parts = [rng.choice(f["ouverture"])]
    if rng.random() < 0.8:
        parts.append(rng.choice(f["corps"]))
    if rng.random() < 0.6:
        parts.append(rng.choice(f["cloture"]))
    texte = " ".join(x for x in parts if x)
    return (texte.replace("{facture}", str(p["facture_mensuelle"])).replace("{offre}", p["offre"])
            .replace("{anciennete}", str(p["anciennete_mois"])).replace("{internet}", p["internet"])
            .replace("{contrat}", p["contrat"]).replace("{parrainages}", str(max(p["parrainages"], 1))))


def generer_local(df: pd.DataFrame, tons: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    textes = [_gabarit(rng, tons.loc[i], _profil(df.loc[i])) for i in df.index]
    return pd.DataFrame({"Customer ID": df["Customer ID"].values, "verbatim": textes,
                         "tonalite_visee": tons.values, "source": "gabarit_stochastique_seed", "modele": None})


# ----------------------------------------------------------------------------------
# Chemin 1 — API Anthropic
# ----------------------------------------------------------------------------------
def generer_api(df: pd.DataFrame, tons: pd.Series, chemin_sortie: Path, taille_lot: int = 25) -> pd.DataFrame:
    import anthropic
    client = anthropic.Anthropic()
    prompt = (RACINE / "prompts" / "verbatim_v1.txt").read_text(encoding="utf-8")

    deja = pd.read_parquet(chemin_sortie) if chemin_sortie.exists() else pd.DataFrame(columns=["Customer ID"])
    faits = set(deja["Customer ID"]) if len(deja) else set()
    lignes = deja.to_dict(orient="records") if len(deja) else []
    a_faire = [i for i in df.index if df.loc[i, "Customer ID"] not in faits]

    for debut in range(0, len(a_faire), taille_lot):
        lot = a_faire[debut:debut + taille_lot]
        clients = [{"id": df.loc[i, "Customer ID"], "tonalite": tons.loc[i], **_profil(df.loc[i])} for i in lot]
        contenu = prompt.replace("{{CLIENTS}}", json.dumps(clients, ensure_ascii=False, indent=1))
        for tentative in range(3):
            try:
                rep = client.messages.create(model=MODELE_LLM, max_tokens=4000,
                                             messages=[{"role": "user", "content": contenu}])
                texte = next(b.text for b in rep.content if b.type == "text")
                debut_j, fin_j = texte.find("["), texte.rfind("]") + 1
                for obj in json.loads(texte[debut_j:fin_j]):
                    lignes.append({"Customer ID": obj["id"], "verbatim": obj["verbatim"],
                                   "tonalite_visee": tons.loc[df.index[df["Customer ID"] == obj["id"]][0]],
                                   "source": "anthropic_api", "modele": rep.model})
                break
            except Exception as e:
                if tentative == 2:
                    raise
                time.sleep(2 ** tentative)
        pd.DataFrame(lignes).to_parquet(chemin_sortie, index=False)  # reprise possible à tout moment
        print(f"  {min(debut + taille_lot, len(a_faire))}/{len(a_faire)} générés")
    return pd.DataFrame(lignes)


def main():
    cfg = charger_config()
    _charger_env()
    df = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    tons = tonalites(df, df["cible_M3"], cfg["seed"])
    sortie = RACINE / "data" / "processed" / "verbatims.parquet"

    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"Clé API détectée — génération via {MODELE_LLM}, prompt prompts/verbatim_v1.txt")
        v = generer_api(df, tons, sortie)
    else:
        print("Aucune clé API — générateur local à gabarits stochastiques (seedé).")
        v = generer_local(df, tons, cfg["seed"])
        v.to_parquet(sortie, index=False)

    concordance = float((v.set_index("Customer ID").loc[df["Customer ID"], "tonalite_visee"].values
                         == df["cible_M3"].map(TON).values).mean())
    print(f"{len(v)} verbatims · source={v['source'].iloc[0]} · concordance tonalité/classe={concordance:.1%} "
          f"(bruit visé {TAUX_BRUIT:.0%}) -> {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
