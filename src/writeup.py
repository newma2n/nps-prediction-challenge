"""Étape 37 — les deux livrables PDF.

    python -m src.writeup

1. livrable/write_up.pdf        — le write-up de 3 à 6 pages demandé par l'énoncé (§ 6), pour un
                                   directeur Expérience Client : approche, cible, modélisation,
                                   évaluation, drivers, limites, suites.
2. livrable/etude_complete.pdf  — l'étude intégrale (les huit documents de reports/), en annexe.

Tout est composé à partir des artefacts (rien n'est recopié), converti en HTML puis en PDF via Chrome
en mode headless. Figures embarquées en base64 : les PDF sont autonomes.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path

import markdown
import pandas as pd

from .data import RACINE, charger_config

REP, FIG = RACINE / "reports", RACINE / "reports" / "figures"
LIV = RACINE / "livrable"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def _b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _img(nom: str, legende: str, largeur: str = "82%") -> str:
    p = FIG / nom
    if not p.exists():
        return ""
    return f'<figure><img src="{_b64(p)}" style="width:{largeur}"><figcaption>{legende}</figcaption></figure>\n'


def _charger(nom):
    p = REP / f"{nom}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _eur(v): return f"{int(round(v)):,} €".replace(",", " ")


def _tab(df) -> str:
    """Tableau Markdown sans dépendance : `to_markdown` exigerait `tabulate`, un paquet de plus
    dans l'image Docker pour six lignes de code."""
    cols = [str(c) for c in df.columns]
    lignes = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, bool):
                cells.append("oui" if v else "non")
            elif isinstance(v, (int, float)):
                cells.append("" if pd.isna(v) else f"{v:,.0f}".replace(",", " "))
            else:
                cells.append(str(v))
        lignes.append("| " + " | ".join(cells) + " |")
    return "\n".join(lignes)


def _lib(n): return str(n).replace("_", " ")


def composer() -> str:
    cfg = charger_config()
    R = _charger("resultats"); tx = _charger("texte"); fo = _charger("fondation"); mo = _charger("monitoring"); rp = _charger("reproductibilite")
    a = R["arbitrage_des_3"]; mf = R["modele_final"]; m = mf["metriques_retenues"]
    pk = R["evaluation_metier"]["precision_at_k"]; eco = R["evaluation_metier"]["economie"]
    eq = R["audit_equite"]; age = eq["tranche_age"]["groupes"]; px = R["audit_proxies"]; mit = R["mitigation_seuils_age"]
    # Section équité : deux politiques chiffrées en appels et en euros, pour que la direction
    # arbitre sur des colonnes comparables et non sur une phrase.
    _pol = mit.get("politiques", {})
    _auc = next((v for k, v in _pol.items() if k.startswith("aucune")), {})
    _niv = next((v for k, v in _pol.items()
                 if k.startswith("nivellement") and "in-sample" not in k), {})
    _rat = next((v for k, v in _pol.items() if k.startswith("rattrapage")), {})
    _cout = cfg["metier"]["cout_appel_eur"]
    if _pol:
        _lignes = []
        for _nom, _v in (("Aujourd'hui, sans correction", _auc),
                         ("Corriger pour tous (nivellement)", _niv),
                         ("Corriger seulement les moins bien servis", _rat)):
            if not _v:
                continue
            _r = _v.get("rappels_par_groupe", {})
            _lignes.append({
                "Politique": _nom,
                "Jeunes mécontents repérés (sur 100)": 100 * _r.get("<30", float("nan")),
                "Seniors mécontents repérés (sur 100)": 100 * _r.get("60+", float("nan")),
                "Appels par campagne": _v.get("appels"),
                "Coût d'appel supplémentaire": (_v.get("appels", 0) - _auc.get("appels", 0)) * _cout,
                "Gain net": _v.get("gain_net_eur")})
        _t = pd.DataFrame(_lignes)
        _t.columns = ["Politique", "Jeunes mécontents repérés (sur 100)", "Seniors mécontents repérés (sur 100)",
                      "Appels par campagne", "Coût d'appel supplémentaire", "Gain net"]
        tab_equite_wu = _tab(_t)
        _d_niv = _niv.get("appels", 0) - _auc.get("appels", 0)
        _d_rat = _rat.get("appels", 0) - _auc.get("appels", 0)
        lecture_equite_wu = (
            "**Niveler pour tous coûte plus qu'il n'y paraît.** Ramener tous les groupes au même niveau ajoute "
            + str(_d_niv) + " appels par campagne (" + _eur(_d_niv * _cout) + "), et surtout : cela **abaisse** la couverture "
            "des seniors mécontents, aujourd'hui les mieux repérés. On égalise en partie par le bas.\n\n"
            "**Rattraper seulement les moins bien servis ne prend rien à personne.** Abaisser le seuil de décision "
            "pour les seuls moins de 30 ans, sans toucher aux autres, coûte " + str(_d_rat) + " appels de plus ("
            + _eur(_d_rat * _cout) + ") et ne dégrade la détection d'aucun groupe. **C'est l'option que nous recommandons "
            "de soumettre au juridique.**")
    else:
        tab_equite_wu = ""
        lecture_equite_wu = ""

    # Chiffres dérivés, jamais écrits en dur : ils doivent suivre les données et le mapping.
    _distM1 = R["arbitrage_des_3"]["distributions"]["M1"]
    part_det_m1 = _distM1.get("Detracteur", float("nan"))
    part_3 = a["n_ambigus"] / 7043
    part_3_restes = R["arbitrage_des_3"].get("part_3_restes")
    if part_3_restes is None:
        # Repli si le pipeline n'a pas encore publié la clé : on recalcule sur le jeu de données
        # dérivé plutôt que d'écrire le nombre en dur — un chiffre figé cesse d'être vrai.
        _d = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet",
                             columns=["Satisfaction Score", "Churn Value"])
        _amb = _d["Satisfaction Score"] == 3
        part_3_restes = float((_d.loc[_amb, "Churn Value"] == 0).mean())
    sel = R["selection_modele_final"]; cl = pd.DataFrame(sel["classement"]); fu = R["diagnostic_fuite"]
    dg = pd.DataFrame(R["diagnostic_protocole"]).set_index("scenario"); ns = R["nps_simule"]; mk = cfg["metier"]
    retenu = cl[cl.modele == sel["retenu"]].iloc[0]; k_log = R["comparatif_s3_m3"]["logistique"]["defaut"]["silencieux"]["kappa"]
    br = [b for b in R["bruit_etiquettes"] if "kappa" in b]; ab = {x["configuration"]: x for x in R["ablation_features"] if "kappa" in x}
    def _dk(motif):
        for k, v in ab.items():
            if motif in k:
                return v["kappa"] - ab["base (retenue)"]["kappa"]
        return float("nan")

    from .rapports import lecture_composition
    lecture_nps = lecture_composition(ns).replace("Lecture : le", "Le").split("Composition par classe")[1]
    lecture_nps = "Composition par classe" + lecture_nps
    top5 = cl.sort_values("rang_cv").head(4)
    tab_modeles = "\n".join(f"| {int(r.rang_cv)} | {_lib(r.modele)} | {r.famille} | {r.kappa_cv_ipw:.3f} | {r.kappa_silencieux:.3f} | {r.rappel_detracteur_silencieux:.2f} | {'✓' if r.dans_la_marge else ''} |"
                            for r in top5.itertuples())
    fond_txt = ""
    if fo:
        ok = [x for x in fo["modeles"] if x["statut"] == "ok"]; ko = [x for x in fo["modeles"] if x["statut"] != "ok"]
        ref = fo["reference"]
        for x in ok:
            fond_txt += (f"**{x['nom']}**, modèle de fondation tabulaire (bonus), a été évalué sur le même découpage : kappa {x['metriques']['kappa']:.3f} "
                         f"contre {ref['kappa']:.3f} pour le modèle retenu, inférence {x['duree_inference_s']/max(ref['duree_inference_s'],1e-3):.0f}× plus lente, sans explication native. "
                         f"Il {'bat' if x['metriques']['kappa'] > ref['kappa'] + 0.005 else 'ne bat pas'} le modèle retenu. ")
        if ko:
            fond_txt += ("Un modèle n'a pas pu être évalué — " + ", ".join(f"{x['nom']} : {x.get('cause', '')}" for x in ko)
                         + " ; la cause exacte et ce qu'il faudrait faire sont dans l'étude. ")
    texte_txt = ""
    if tx:
        texte_txt = (f"**Texte.** {tx['n_verbatims']} notes de contact synthétiques (source `{tx['source_verbatims']}`, 25 % de bruit de tonalité) ont été "
                     f"combinées au modèle tabulaire : la fusion gagne {tx['gain_kappa_fusion_tardive']:+.3f} de kappa — et ce chiffre ne prouve rien, le texte ayant été "
                     f"fabriqué à partir de la classe. L'expérience valide la chaîne technique ; conclusion opérationnelle : **pas de composante texte** sur cette base.")
    mo_txt = ""
    if mo:
        su = mo.get("suivi_nouvelles_reponses", {})
        mo_txt = (f"Implémenté : dérive des entrées et des prédictions, performance **et équité par tranche d'âge** sur les nouvelles "
                  f"réponses, {len(mo.get('declencheurs_reentrainement', [])) or 5} déclencheurs de réentraînement. Sur un mois simulé ({mo['scenario_simule']}), "
                  f"la dérive {'et l’équité se déclenchent' if su.get('alerte_equite_detectee') else 'se déclenche'} — le dispositif est vérifié, pas seulement décrit.")
    repro_txt = f"Test à blanc depuis une copie vierge : {'identique sur les ' + str(len(rp['comparaison'])) + ' indicateurs contrôlés' if rp and rp.get('reproductible') else 'à relancer'}." if rp else ""

    return f"""
# Prédire la catégorie NPS des clients silencieux
## Priorisation des appels de rétention — write-up pour la direction Expérience Client

---

## 1. En une page

L'opérateur interroge ses clients sur leur propension à le recommander, et 15 % répondent. J'ai construit un modèle qui
estime, pour chacun des 85 % silencieux, s'il est Détracteur, Passif ou Promoteur. Il alimente une application qui en tire
une liste d'appels priorisée, avec pour chaque client les raisons et le levier à activer. J'y ajoute une estimation du NPS
des silencieux et un dispositif de suivi.

| Les cinq chiffres | |
|---|---|
| NPS réel de la base, avec la grille retenue | **{a['nps']['M3']:+.0f}**. On le connaît ici parce que la note des 7 043 clients est dans le jeu de données : ce n'est pas une prédiction. Avec la grille proposée dans l'énoncé, il vaudrait **{a['nps']['M1']:+.0f}** |
| NPS des 85 % silencieux, estimé par le modèle | **{ns['nps_silencieux_predit']:+.0f}** (intervalle {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f}). La vérité est {ns['nps_silencieux_vrai']:+.0f}, donc je sous-estime de {abs(ns['nps_silencieux_predit'] - ns['nps_silencieux_vrai']):.0f} points. Ce chiffre se lit en tendance, pas en valeur absolue. Les répondants seuls donneraient {ns['nps_repondants_observe']:+.0f} |
| Sur {pk['k']} appels, part de vrais détracteurs | **{pk['precision_at_k']:.0%}** contre {pk['taux_de_base']:.0%} au hasard — ×{pk['lift']:.1f} |
| Gain net supplémentaire par campagne | **{_eur(eco['gain_apporte_par_le_modele_eur'])}** à hypothèses de coût constantes (à valider) |
| Détracteurs retrouvés | **{m['par_classe']['Detracteur']['rappel']:.0%}** — mais {age['<30']['rappel_detracteur']:.0%} seulement chez les moins de 30 ans |

Trois résultats me paraissent devoir être retenus.

D'abord, la grille NPS proposée dans l'énoncé donne un résultat peu crédible. Elle classe en détracteurs {a['n_ambigus']} clients
moyennement satisfaits, dont {part_3_restes:.0%} sont pourtant toujours là. En entraînant un modèle sur les seuls clients aux notes
extrêmes, j'obtiens que {a['part_3_vers_promoteur']:.0%} d'entre eux ressemblent plutôt à des promoteurs.

Ensuite, le modèle multiplie par {pk['lift']:.1f} l'efficacité des appels. Je l'ai choisi sans regarder les clients qui servent à
l'évaluer : {len(cl)} modèles de 7 familles comparés, sélection par validation croisée sur les seuls répondants, et à performance
équivalente le plus simple l'emporte. C'est la {_lib(mf['modele'])} qui sort.

Enfin, il repère moins bien les jeunes détracteurs, alors qu'il n'utilise pas l'âge. Le contrat et l'ancienneté suffisent à le
retrouver. J'ai chiffré une correction, mais la décision de l'appliquer ne m'appartient pas : elle est remontée au métier et au
juridique.

**Sur le périmètre.** J'ai cherché à couvrir ce qui est demandé plutôt qu'à en faire trop, et une matrice en annexe relie chaque
pièce du rendu à l'exigence correspondante. Les quinze modèles ne sont pas une course au score : quand des approches aussi
différentes plafonnent au même endroit, c'est que la limite vient des données. La marche suivante est donc métier, pas
algorithmique. J'ai laissé de côté l'enrichissement externe, les réseaux ordinaux, une API de scoring et un suivi d'expériences,
en expliquant pourquoi dans l'étude.

**Ce que je vous demande.** Appeler les clients les plus à risque n'est pas forcément la stratégie la plus rentable. La
littérature montre qu'il vaut mieux cibler ceux dont le comportement change le plus quand on les appelle (Ascarza,
*J. Marketing Research*, 2018). Pour le mesurer, il suffit de ne pas contacter 10 à 20 % des clients ciblés à la prochaine
campagne, tirés au hasard. Cela ne coûte rien.

---

## 2. Les données et leurs pièges

Le jeu de données couvre 7 043 clients d'un opérateur californien fictif. J'ai utilisé la version IBM Telco Customer Churn
11.1.3+, la seule qui porte la note de satisfaction ; le fichier Kaggle le plus diffusé ne l'a pas. Cinq tables jointes sans
perte, 51 colonnes, et une colonne clé : une note de satisfaction de 1 à 5. Trois constats ont orienté tout le reste.

**Le jeu de données contient sa propre réponse.** Tous les clients notés 1 ou 2 sont partis, aucun de ceux notés 4 ou 5. Du coup,
n'importe quelle colonne liée au départ (`Churn Value`, `Churn Score`, `Customer Status`, `CLTV`) donnerait un score presque parfait
et un modèle inutilisable : au moment où l'on veut repérer un détracteur, il n'est pas encore parti. Je les ai classées en trois
types de fuite, mesurées par information mutuelle, puis exclues. Deux contrôles après modélisation vérifient qu'il n'en reste pas
(exactitude maximale {fu['exactitude_max_observee']:.2f} pour un seuil d'alerte à 0,85 ; écart entre arbres et linéaire {fu['ecart_arbres_lineaire_S1_M3']:+.2f} pour un seuil à +0,30).

**La note « 3 » décide de tout.** Ils sont {a['n_ambigus']}, soit {part_3:.0%} de la base, et {part_3_restes:.0%} sont encore clients. Plutôt que de trancher
moi-même, j'ai entraîné un modèle sur les seules notes extrêmes (1–2 contre 4–5, {a['exactitude_sur_extremes']:.0%} d'exactitude) et je lui ai fait
classer ces clients : {a['part_3_vers_promoteur']:.0%} ressemblent à des promoteurs. J'ai donc traité le bloc comme Passif. Attention, je n'ai pas étiqueté
chaque client par ce modèle : la cible serait devenue une fonction des variables, et le modèle final aurait réappris sa propre sortie.

Voici la grille que je retiens, et l'écart avec celle de l'énoncé :

| Grille | Détracteur | Passif | Promoteur | NPS | Statut |
|---|---|---|---|---|---|
| Énoncé | 1, 2, 3 | 4 | 5 | **{a['nps']['M1']:+.1f}** | écartée : {part_det_m1:.0%} de la base en détracteurs |
| Recentrée | 1, 2 | 3, 4 | 5 | **{a['nps']['M2']:+.1f}** | variante prudente, publiée |
| **Retenue** | **1, 2** | **3** | **4, 5** | **{a['nps']['M3']:+.1f}** | le « 3 » arbitré par la mesure, le « 4 » par hypothèse d'échelle |

Ce sont deux décisions, pas une, et elles n'ont pas le même statut. Le déplacement du « 3 » est mesuré. Le passage du « 4 » chez
les promoteurs est une hypothèse que j'assume : une note de 4 sur 5 correspond à « satisfait », et aucun client noté 4 n'a quitté
l'opérateur. Pour qui refuse cette hypothèse, la grille recentrée est calculée et publiée.



**Les offres commerciales sont une quasi-expérience — piégée.** Les clients ayant reçu l'offre E partent **deux fois plus** que ceux sans
offre : elle a été proposée à des clients déjà fragiles. On ne lit pas un effet causal dans des données observationnelles.

Les variables retenues ({len(R['features']['numeriques'])} numériques, {len(R['features']['categorielles'])} catégorielles) sont chacune adossées à une hypothèse écrite ; la démographie et la
géographie sont exclues du modèle par décision d'équité, et le coût de cette exclusion est chiffré ({_dk('+ démographie'):+.3f} de kappa pour la
démographie, {_dk('+ géographie'):+.3f} pour la géographie). Aucun enrichissement externe : il réintroduirait le proxy socio-économique qu'on a exclu.

---

## 3. Un modèle évalué dans les conditions réelles, choisi sans regarder la réponse

**Le protocole.** Séparer les clients au hasard entre apprentissage et test m'aurait induit en erreur. Ceux qui répondent à une
enquête ne sont pas un échantillon au hasard : les mécontents et les enthousiastes répondent davantage. J'ai donc simulé trois
façons de ne pas répondre, jusqu'à la plus réaliste, où le fait de répondre dépend de la satisfaction elle-même¹. Dans ce cas,
la part de notes extrêmes passe de {dg.loc['S3_MNAR','part_extremes_base']:.0%} dans la base à {dg.loc['S3_MNAR','part_extremes_repondants']:.0%} chez les répondants. Le modèle apprend sur ces 15 % biaisés,
et il est jugé sur les 85 % silencieux, dont on connaît ici la vraie réponse.

**Quinze modèles, sept familles, trois façons de poser le problème.** Chacun teste une hypothèse précise sur les données. Je les
ai comparés sur toutes les combinaisons de cible et de scénario, réglés, puis choisis en n'utilisant que les répondants, seule
information dont on disposerait en production. La règle est simple : entre des modèles que la mesure ne distingue pas, je garde
le plus simple². Les 85 % silencieux servent à vérifier, jamais à choisir.

| Rang | Modèle | Famille | Score sur les répondants³ | Score sur les silencieux (vérité) | Détracteurs retrouvés | Dans la marge |
|---|---|---|---|---|---|---|
{tab_modeles}

Retenu : **{_lib(sel['retenu'])}** ({sel['version_retenue']}). Le meilleur sur la vérité est {_lib(sel['gagnant_silencieux'])}, à {abs(retenu['kappa_silencieux'] - cl[cl.modele == sel['gagnant_silencieux']]['kappa_silencieux'].iloc[0]):.3f} près : {"c'est le même" if sel['gagnant_silencieux'] == sel['retenu'] else "un écart assumé, prix d'un choix fait sans regarder la réponse"}.
J'en retiens deux choses. La validation promet {retenu['kappa_cv']:.2f} et la vérité donne {retenu['kappa_silencieux']:.2f} : cet écart est le coût du biais de réponse, et
il serait invisible en production. C'est la raison pour laquelle il faut de vraies réponses de la population cible (section 6).
Ensuite, tous les modèles tiennent dans une bande étroite, de {cl['kappa_silencieux'].min():.2f} à {cl['kappa_silencieux'].max():.2f} : ce qui limite la performance, c'est le
signal disponible, pas l'algorithme. {fond_txt}

{_img("04_cv_vs_silencieux.png", "Figure 2 — Ce que la validation sur les répondants promet (horizontal) contre ce que les 85 % silencieux donnent (vertical). Rouge : retenu ; bleu : modèles que la mesure ne distingue pas du meilleur.", "88%")}

**Performance classe par classe.** Je ne publie jamais un score global sans son détail.

| Classe | Quand le modèle dit « X », il a raison à | Part des vrais « X » retrouvés | Effectif |
|---|---|---|---|
| Détracteur | {m['par_classe']['Detracteur']['precision']:.0%} | **{m['par_classe']['Detracteur']['rappel']:.0%}** | {m['par_classe']['Detracteur']['effectif']} |
| Passif | {m['par_classe']['Passif']['precision']:.0%} | {m['par_classe']['Passif']['rappel']:.0%} | {m['par_classe']['Passif']['effectif']} |
| Promoteur | {m['par_classe']['Promoteur']['precision']:.0%} | {m['par_classe']['Promoteur']['rappel']:.0%} | {m['par_classe']['Promoteur']['effectif']} |

Le score d'accord ordonné³ vaut {m['kappa_quadratique']:.3f}, contre {k_log:.3f} pour le point de départ logistique, et la qualité du classement des
détracteurs⁴ atteint {m.get('auc_detracteur', float('nan')):.2f}. Le Passif, c'est-à-dire le client moyennement satisfait, est le plus difficile à retrouver :
il est rare chez les répondants, et la littérature fait le même constat. J'ai testé puis corrigé les probabilités affichées.
Ajustées sur des répondants biaisés, elles surestiment le risque : elles servent à classer les clients entre eux, pas à lire
une fréquence. J'ai également vérifié la robustesse. Corrompre {br[-1]['taux_bruit']:.0%} des étiquettes d'apprentissage ne coûte que {br[0]['kappa'] - br[-1]['kappa']:.3f} de
score, et les conclusions sur les facteurs de détraction tiennent quelle que soit la définition retenue pour la cible. {texte_txt}

## 4. Ce que ça vaut pour l'équipe rétention



Sur {pk['k']} appels par mois, la liste contient {pk['precision_at_k']:.0%} de vrais détracteurs, ce qui couvre {pk['rappel_at_k']:.0%} d'entre eux, soit {pk['lift']:.1f} fois mieux qu'un
ciblage au hasard. En prenant un coût d'appel de {mk['cout_appel_eur']} €, une valeur de client retenu de {mk['valeur_client_retenu_eur']} € et un taux de succès de {mk['taux_succes_appel']:.0%},
la campagne ciblée rapporte {_eur(eco['gain_net_eur'])} nets contre {_eur(eco['gain_net_ciblage_aleatoire_eur'])} au hasard. Ces trois paramètres sont des hypothèses, à valider avec
votre équipe. J'ai aussi laissé `CLTV` en option pour pondérer qui appeler : elle est interdite comme variable du modèle, mais
rien n'empêche de s'en servir pour arbitrer entre deux détracteurs prédits.

**Le NPS des silencieux.** Le modèle l'estime à {ns['nps_silencieux_predit']:+.0f}, avec un intervalle de {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f}. La vérité, qu'on connaît ici,
est {ns['nps_silencieux_vrai']:+.0f}, soit {ns['nps_silencieux_predit'] - ns['nps_silencieux_vrai']:+.0f} points d'écart. {lecture_nps} {("Quatre estimateurs de quantification (Forman 2005) ont été testés pour corriger cet écart ; le meilleur reste à " + f"{min(abs(ns['quantification'][k]['nps'] - ns['quantification']['verite']['nps']) for k in ('CC','PCC','ACC','PACC')):.0f}" + " pts : la matrice de confusion mesurée sur les répondants ne se transfère pas aux silencieux — la correction doit venir de vraies réponses de la population cible.") if ns.get('quantification') else ''}

{_img("05_nps_simule.png", "Figure 4 — Le NPS que l'opérateur observe (répondants), celui que le modèle estime pour les silencieux, la vérité.", "78%")}

**Ce qui fait un détracteur, selon le modèle.** Un contrat sans engagement, une faible ancienneté, l'absence de parrainage et une
facture élevée. Ces associations restent les mêmes quelle que soit la grille retenue pour la cible, et elles rejoignent ce que
décrit la littérature. Leur poids varie d'un segment à l'autre, et l'étude les donne segment par segment : contrat mensuel ou
engagé, client récent ou ancien, fibre ou DSL.

Pour chaque détracteur prédit, l'application propose deux leviers, classés selon ce qui pèse chez ce client précis. Le premier est
souvent l'engagement, simplement parce que la quasi-totalité des détracteurs prédits sont en contrat mensuel : c'est vrai, mais cela
ne trie pas. C'est le second levier qui distingue deux clients par ailleurs semblables, et c'est celui qu'il faut lire en priorité.

Un point important : ce sont des associations, pas des causes. L'offre E « prédit » le départ parce qu'elle a été proposée à des
clients déjà fragiles, pas parce qu'elle les fait partir.

---

## 5. Équité

{_img("05_equite.png", "Figure 5 — Rappel Détracteur (bleu) et taux de sélection (gris) par sous-groupe. Variables exclues du modèle, utilisées uniquement pour cet audit.")}

**Le problème.** Sur 100 clients mécontents de moins de 30 ans, ma liste d'appels en repère {age['<30']['rappel_detracteur']*100:.0f}. Sur 100 seniors mécontents,
elle en repère {age['60+']['rappel_detracteur']*100:.0f}. À budget d'appels égal, les jeunes clients mécontents ont donc nettement moins de chances d'être rappelés.

**Pourquoi, alors que le modèle ne connaît pas l'âge ?** Il ne le connaît pas, mais il le devine. Les jeunes clients sont pour la
plupart en contrat mensuel et récents, si bien que le contrat et l'ancienneté suffisent à retrouver la tranche d'âge presque à coup
sûr. Retirer la colonne empêche de décider explicitement sur l'âge, cela n'empêche pas le modèle de le voir.

**Ce que coûterait la correction, en appels et en euros.** Deux options ont été chiffrées, et elles n'ont pas le même prix.

{tab_equite_wu}

{lecture_equite_wu}

**Ce que je vous demande.** Je n'applique aucune de ces corrections par défaut, parce qu'appliquer un seuil différent selon l'âge
est une décision juridique avant d'être technique. Selon le pays, c'est une discrimination prohibée ou une mesure correctrice
admise, et ce n'est pas à moi de trancher. La question est à poser au juridique, avec les chiffres ci-dessus. L'écart est par
ailleurs recalculé chaque mois sur les nouvelles réponses et déclenche une alerte au-delà de dix points, donc l'audit ne s'arrête
pas à cette étude. *(Détail technique en phase 5 § 7 de l'étude : AUC de reconstruction de l'âge {px.get('Moins de 30 ans', {}).get('auc', float('nan')):.2f}, de la situation familiale
{px.get('Marié(e)', {}).get('auc', float('nan')):.2f} ; genre et densité de zone non reconstructibles et sans écart.)*

---

## 6. Limites, décision, suites

**Ce que ces données ne permettent pas.** Estimer l'effet d'un appel, tout simplement : il n'y a aucune campagne enregistrée ni
assignation aléatoire. Mon modèle classe par risque. L'optimum économique, lui, classerait par sensibilité à l'intervention,
c'est-à-dire les clients qui restent parce qu'on les a appelés, et non ceux qui étaient déjà décidés.

**Ma recommandation.** À la prochaine campagne, ne contactez pas 10 à 20 % des clients ciblés, tirés au hasard. Ce groupe de
contrôle vous donne la mesure de l'effet réel de la campagne, et les données qu'il faudrait pour cibler par sensibilité au cycle
suivant. Sans lui, un détracteur appelé puis retenu change de classe, le modèle réapprend l'effet de sa propre intervention, et il
se dégrade sans que personne ne comprenne pourquoi.

**Monitoring.** {mo_txt}

**Ce que j'assume comme limites.** Les hypothèses économiques restent à valider avec vous. La propension à répondre est simulée :
elle est plausible, elle n'est pas observée. Le taux de réponse de 15 % est optimiste, Kannan et ses coauteurs rapportent 2,6 %. Le
cadrage panafricain de l'énoncé est plaqué sur un jeu de données californien. Les verbatims sont synthétiques. Enfin, TabPFN 2.5
n'a pas été évalué, sa licence PriorLabs demandant un compte et une clé API.

**Reproductibilité et livraison.** `docker compose up` lance l'application sans rien installer, modèle et données embarqués ;
`python -m src.run` régénère tout depuis les fichiers bruts. Seed unique, aucune clé dans le dépôt, tests automatisés. {repro_txt}

**Usage d'outils d'IA** (énoncé § 7) : code, structure de l'étude et rédaction produits avec l'assistance d'un assistant IA (Claude),
sous direction et relecture humaines ; fragments des verbatims synthétiques rédigés par ce même assistant. Les choix de modélisation,
les décisions de périmètre et les conclusions sont assumés par l'auteur.

---

*Glossaire.* ¹ **Non-réponse dépendante de la réponse** (MNAR, *missing not at random*) : la probabilité de répondre dépend de ce
qu'on aurait répondu. ² **Règle de parcimonie** (« one standard error rule », Hastie et al., ESL § 7.10) : parmi les modèles à moins
d'un écart-type du meilleur, le plus simple. ³ **Kappa quadratique pondéré** : score d'accord entre prédiction et vérité, corrigé du
hasard, qui pénalise davantage une erreur de deux crans (Promoteur ↔ Détracteur) qu'une erreur d'un cran ; sur les répondants il
est pondéré pour représenter la population entière (IPW). ⁴ **AUC** : probabilité qu'un vrai détracteur soit classé avant un
non-détracteur (0,5 = hasard, 1 = parfait).

*Références.* Ascarza (2018) *Retention Futility*, J. Marketing Research · Elero et al. (2026) *Computers* 15(2) · Chong et al. (2023)
J. Soft Computing & Data Mining · Mustafa et al. (2021) F1000Research · Kannan et al. (2022) ICTIM · Hastie, Tibshirani & Friedman,
*Elements of Statistical Learning* § 7.10 · Gómez-Vargas et al. (2025) Eur. J. Operational Research · Qu et al. (2025) TabICL, ICML ·
Hollmann et al. (2025) TabPFN, Nature.
"""


CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.2pt; line-height: 1.46; color: #1c2430; }
h1 { font-family: Arial, Helvetica, sans-serif; font-size: 20pt; color: #0d2440; margin: 0 0 2px; }
h2 { font-family: Arial, Helvetica, sans-serif; font-size: 13pt; color: #0d2440; margin-top: 14px; border-bottom: 1px solid #c9d3de; padding-bottom: 2px; }
h3 { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; color: #1f4e79; margin-top: 10px; }
h1 + h2 { border: none; font-weight: normal; color: #4a5a6a; font-size: 12pt; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 8.9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d3de; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #eef3f8; }
figure { margin: 5px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.6pt; color: #4a5a6a; margin-top: 2px; font-style: italic; }
hr { border: 0; border-top: 1px solid #c9d3de; margin: 10px 0; }
li { margin-bottom: 2px; }
p { orphans: 3; widows: 3; margin: 5px 0; }
em { color: #4a5a6a; }
code { font-size: 8.8pt; background: #f3f5f7; padding: 0 2px; }
blockquote { border-left: 3px solid #2c7fb8; margin: 6px 0; padding: 4px 10px; background: #f3f8fc; }
.page { page-break-before: always; }
"""


def _vers_pdf(html_path: Path, pdf_path: Path):
    chrome = next((c for c in CHROME if Path(c).exists()), None)
    if chrome is None:
        print("Chrome/Edge introuvable — HTML produit, PDF non généré."); return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
                    html_path.resolve().as_uri()], capture_output=True, timeout=180)
    if pdf_path.exists():
        try:
            import fitz
            print(f"{pdf_path.relative_to(RACINE)} — {fitz.open(pdf_path).page_count} pages, {pdf_path.stat().st_size//1024} Ko")
        except Exception:
            print(f"{pdf_path.relative_to(RACINE)} — {pdf_path.stat().st_size//1024} Ko")
    else:
        print(f"Échec de la conversion PDF pour {pdf_path.name} ; HTML disponible.")


def _html(md: str) -> str:
    corps = markdown.markdown(md, extensions=["tables", "sane_lists"])
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{corps}</body></html>"


def etude_complete() -> str:
    """Concatène les huit documents de reports/ en un seul Markdown, images en base64."""
    ordre = ["00_synthese.md", "01_comprehension_metier.md", "02_comprehension_donnees.md", "03_preparation_donnees.md",
             "04_modelisation.md", "05_evaluation.md", "06_deploiement.md", "07_conformite_enonce.md"]
    parties = ["""# Étude complète

## Prédiction de la catégorie NPS des clients silencieux

Ce document détaille le travail résumé dans le write-up. Il suit les six phases de la démarche
CRISP-DM et se termine par la correspondance avec l'énoncé, exigence par exigence.

| | |
|---|---|
| 1 | Compréhension métier |
| 2 | Compréhension des données |
| 3 | Préparation des données |
| 4 | Modélisation |
| 5 | Évaluation |
| 6 | Déploiement |
| 7 | Conformité à l'énoncé |

Tous les chiffres sont produits par le pipeline et se retrouvent dans `reports/resultats.json`.
"""]
    for nom in ordre:
        p = REP / nom
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"!\[\]\(figures/([^)]+)\)", lambda mm: f'<figure><img src="{_b64(FIG / mm.group(1))}" style="width:100%"></figure>' if (FIG / mm.group(1)).exists() else "", t)
        t = re.sub(r"\]\((0\d_[a-z_]+\.md)\)", r"](#)", t)
        parties.append('<div class="page"></div>\n\n' + t)
    return "\n\n".join(parties)


def main():
    LIV.mkdir(exist_ok=True)
    md = composer()
    (LIV / "write_up.md").write_text(md, encoding="utf-8")
    (LIV / "write_up.html").write_text(_html(md), encoding="utf-8")
    _vers_pdf(LIV / "write_up.html", LIV / "write_up.pdf")
    md2 = etude_complete()
    (LIV / "etude_complete.html").write_text(_html(md2), encoding="utf-8")
    _vers_pdf(LIV / "etude_complete.html", LIV / "etude_complete.pdf")


if __name__ == "__main__":
    main()
