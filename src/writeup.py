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


def _img(nom: str, legende: str, largeur: str = "100%") -> str:
    p = FIG / nom
    if not p.exists():
        return ""
    return f'<figure><img src="{_b64(p)}" style="width:{largeur}"><figcaption>{legende}</figcaption></figure>\n'


def _charger(nom):
    p = REP / f"{nom}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _eur(v): return f"{int(round(v)):,} €".replace(",", " ")
def _lib(n): return str(n).replace("_", " ")


def composer() -> str:
    cfg = charger_config()
    R = _charger("resultats"); tx = _charger("texte"); fo = _charger("fondation"); mo = _charger("monitoring"); rp = _charger("reproductibilite")
    a = R["arbitrage_des_3"]; mf = R["modele_final"]; m = mf["metriques_retenues"]
    pk = R["evaluation_metier"]["precision_at_k"]; eco = R["evaluation_metier"]["economie"]
    eq = R["audit_equite"]; age = eq["tranche_age"]["groupes"]; px = R["audit_proxies"]; mit = R["mitigation_seuils_age"]
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
    top5 = cl.sort_values("rang_cv").head(6)
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
            fond_txt += "TabPFN n'a pas pu être évalué (poids à accès contrôlé sur Hugging Face) ; sa place est documentée dans l'étude. "
    texte_txt = ""
    if tx:
        texte_txt = (f"**Texte.** {tx['n_verbatims']} notes de contact synthétiques (source `{tx['source_verbatims']}`, 25 % de bruit de tonalité) ont été "
                     f"combinées au modèle tabulaire : la fusion gagne {tx['gain_kappa_fusion_tardive']:+.3f} de kappa — et ce chiffre ne prouve rien, le texte ayant été "
                     f"fabriqué à partir de la classe. L'expérience valide la chaîne technique ; conclusion opérationnelle : **pas de composante texte** sur cette base.")
    mo_txt = ""
    if mo:
        su = mo.get("suivi_nouvelles_reponses", {})
        mo_txt = (f"Un monitoring est implémenté : dérive des entrées et des prédictions (indice PSI), performance sur les nouvelles réponses d'enquête, "
                  f"déclencheurs par seuil de dérive, chute de performance, volume de labels ({su.get('seuils', {}).get('min_nouveaux_labels', 500)} nouvelles réponses) et échéance. "
                  f"Sur un mois simulé ({mo['scenario_simule']}), le déclencheur de dérive {'se déclenche' if mo['reentrainement_declenche_mois_simule'] else 'ne se déclenche pas'} "
                  f"(variables en alerte : {', '.join(mo['variables_en_alerte_mois_simule']) or 'aucune'}).")
    repro_txt = f"Test à blanc depuis une copie vierge : {'identique sur les ' + str(len(rp['comparaison'])) + ' indicateurs contrôlés' if rp and rp.get('reproductible') else 'à relancer'}." if rp else ""

    return f"""
# Prédire la catégorie NPS des clients silencieux
## Priorisation des appels de rétention — write-up pour la direction Expérience Client

---

## 1. En une page

Un opérateur télécom interroge ses clients sur leur propension à le recommander ; **15 % répondent**. Ce travail
construit un modèle qui estime, pour chacun des 85 % silencieux, s'il est Détracteur, Passif ou Promoteur ; une
application qui en tire une liste d'appels priorisée avec, pour chaque client, les raisons et le levier ; une
estimation du NPS des silencieux ; et un dispositif de suivi.

| Les cinq chiffres | |
|---|---|
| NPS estimé de la base complète | **{a['nps']['M3']:+.0f}** (secteur +19 à +34) — contre **{a['nps']['M1']:+.0f}** avec la grille de l'énoncé prise au pied de la lettre |
| NPS des 85 % silencieux, estimé par le modèle | **{ns['nps_silencieux_predit']:+.0f}** (intervalle {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f}) ; les répondants seuls donnent {ns['nps_repondants_observe']:+.0f} |
| Sur {pk['k']} appels, part de vrais détracteurs | **{pk['precision_at_k']:.0%}** contre {pk['taux_de_base']:.0%} au hasard — ×{pk['lift']:.1f} |
| Gain net supplémentaire par campagne | **{_eur(eco['gain_apporte_par_le_modele_eur'])}** à hypothèses de coût constantes (à valider) |
| Détracteurs retrouvés | **{m['par_classe']['Detracteur']['rappel']:.0%}** — mais {age['<30']['rappel_detracteur']:.0%} seulement chez les moins de 30 ans |

**Trois résultats.** (1) **La grille NPS de l'énoncé fabrique un résultat faux** : elle classe en détracteurs {a['n_ambigus']} clients
« moyennement satisfaits » dont 84 % sont restés ; un modèle entraîné sur les seuls extrêmes montre qu'ils penchent à
{a['part_3_vers_promoteur']:.0%} côté promoteur. (2) **Le modèle multiplie par {pk['lift']:.1f} l'efficacité des appels**, et il a été choisi honnêtement : {len(cl)} modèles
de 7 familles comparés, sélection par validation croisée sur les répondants *sans regarder* les clients d'évaluation, règle de
parcimonie ; retenu : {_lib(mf['modele'])}. (3) **Il rate davantage les jeunes détracteurs**, sans variable d'âge : les variables
contractuelles reconstruisent l'âge ; une correction est chiffrée, la décision de l'appliquer est remontée au métier et au juridique.

**Une décision demandée.** Appeler les clients les plus à risque n'est pas la stratégie la plus rentable : il faut cibler ceux dont le
comportement change le plus quand on les appelle (Ascarza, *J. Marketing Research*, 2018). Le mesurer ne demande qu'une chose,
gratuite : **ne pas contacter 10 à 20 % des clients ciblés à la prochaine campagne, tirés au hasard.**

---

## 2. Les données et leurs pièges

7 043 clients d'un opérateur californien fictif (IBM Telco), cinq tables jointes sans perte, 51 colonnes. La colonne clé est une
**satisfaction de 1 à 5** déclarée par le client. Trois faits ont structuré le travail.

**Le dataset contient sa propre réponse.** Les clients à satisfaction 1–2 sont **tous partis** ; ceux à 4–5, **aucun**. Toute colonne
liée au départ (`Churn Value`, `Churn Score`, `Customer Status`, `CLTV`…) donnerait un score presque parfait et un modèle inutilisable —
au moment de détecter un détracteur, il n'est pas encore parti. Ces colonnes sont classées en trois natures de fuite (conséquence,
disponibilité, modèle amont), **mesurées** par information mutuelle, et exclues ; deux diagnostics vérifient après modélisation qu'aucune
fuite ne subsiste (exactitude maximale {fu['exactitude_max_observee']:.2f} pour un seuil d'alerte à 0,85 ; écart arbres/linéaire {fu['ecart_arbres_lineaire_S1_M3']:+.2f} pour un seuil à +0,30).

**La note « 3 » décide de tout.** {a['n_ambigus']} clients, 38 % de la base, 84 % encore présents. Plutôt que de décider où ils vont, un modèle
entraîné sur les seuls extrêmes (1–2 contre 4–5, exactitude {a['exactitude_sur_extremes']:.0%}) les a classés : {a['part_3_vers_promoteur']:.0%} ressemblent à des promoteurs. Le bloc
est traité comme **Passif**. On a évité d'étiqueter chaque client individuellement par ce modèle — la cible serait devenue une fonction
des variables, et le modèle final aurait réappris sa propre sortie.

{_img("03_cible_mappings.png", "Figure 1 — Trois façons de construire la cible, et le NPS qui en résulte. Bande verte : benchmark télécom publié.")}

**Les offres commerciales sont une quasi-expérience — piégée.** Les clients ayant reçu l'offre E partent **deux fois plus** que ceux sans
offre : elle a été proposée à des clients déjà fragiles. On ne lit pas un effet causal dans des données observationnelles.

Les variables retenues ({len(R['features']['numeriques'])} numériques, {len(R['features']['categorielles'])} catégorielles) sont chacune adossées à une hypothèse écrite ; la démographie et la
géographie sont exclues du modèle par décision d'équité, et le coût de cette exclusion est chiffré ({_dk('+ démographie'):+.3f} de kappa pour la
démographie, {_dk('+ géographie'):+.3f} pour la géographie). Aucun enrichissement externe : il réintroduirait le proxy socio-économique qu'on a exclu.

---

## 3. Un modèle évalué dans les conditions réelles, choisi sans regarder le test

**Le protocole.** Un découpage aléatoire aurait été trompeur : les répondants ne sont pas un échantillon au hasard — les mécontents et
les enthousiastes répondent plus. Trois mécanismes de non-réponse ont été simulés ; dans le scénario réaliste (la propension à
répondre dépend de la satisfaction), la part de notes extrêmes passe de **{dg.loc['S3_MNAR','part_extremes_base']:.0%} dans la base à {dg.loc['S3_MNAR','part_extremes_repondants']:.0%} chez les répondants**. Le
modèle est entraîné sur ces 15 % biaisés et évalué sur les 85 % silencieux, dont on connaît la vérité.

**Quinze modèles, sept familles, trois formulations.** Régression logistique (baseline), logistique ordinale, régression à seuils
(linéaire et boosting), arbre, forêt aléatoire, cinq boostings (HGB, LightGBM, XGBoost, CatBoost, AdaBoost), KNN, SVM, réseau de
neurones, Naive Bayes — chacun présent pour tester une hypothèse nommée. Comparés sur 3 scénarios × 3 mappings, réglés par recherche
aléatoire, puis **sélectionnés par validation croisée sur les répondants, pondérée par l'inverse de la propension estimée** — la seule
information disponible en production — avec une règle de parcimonie : parmi les modèles à moins d'un écart-type du meilleur, le plus
simple l'emporte. Les silencieux ne servent qu'au test.

| Rang CV | Modèle | Famille | Kappa CV (IPW) | Kappa test | Rappel Dét. test | Dans la marge |
|---|---|---|---|---|---|---|
{tab_modeles}

Retenu : **{_lib(sel['retenu'])}** ({sel['version_retenue']}). Meilleur kappa CV brut : {_lib(sel['meilleur_cv_brut'])} ; meilleur sur le test : {_lib(sel['gagnant_silencieux'])}
({cl[cl.modele == sel['gagnant_silencieux']]['kappa_silencieux'].iloc[0]:.3f} contre {retenu['kappa_silencieux']:.3f} pour le retenu — {'le même' if sel['gagnant_silencieux'] == sel['retenu'] else 'écart assumé, prix d’une sélection honnête'}). La corrélation entre ce que la validation
croisée promet et ce que le test donne est de ρ = {sel['spearman_cv_ipw_vs_silencieux']:.2f} ; pour le modèle retenu, la CV promet {retenu['kappa_cv']:.2f} et le test donne {retenu['kappa_silencieux']:.2f} : **c'est le
coût du biais de réponse, invisible en production**. Tous les kappas de test tiennent dans une bande étroite ({cl['kappa_silencieux'].min():.2f}–{cl['kappa_silencieux'].max():.2f}) : le
plafond n'est pas algorithmique, il est dans le signal. {fond_txt}

{_img("04_cv_vs_silencieux.png", "Figure 2 — Ce que la validation croisée sur les répondants promet (x) contre ce que les 85 % silencieux donnent (y). Rouge : retenu ; bleu : dans la marge de parcimonie.", "88%")}

**Performance, par classe — jamais globale seule.**

| Classe | Précision | Rappel | F1 | Effectif |
|---|---|---|---|---|
| Détracteur | {m['par_classe']['Detracteur']['precision']:.2f} | **{m['par_classe']['Detracteur']['rappel']:.2f}** | {m['par_classe']['Detracteur']['f1']:.2f} | {m['par_classe']['Detracteur']['effectif']} |
| Passif | {m['par_classe']['Passif']['precision']:.2f} | {m['par_classe']['Passif']['rappel']:.2f} | {m['par_classe']['Passif']['f1']:.2f} | {m['par_classe']['Passif']['effectif']} |
| Promoteur | {m['par_classe']['Promoteur']['precision']:.2f} | {m['par_classe']['Promoteur']['rappel']:.2f} | {m['par_classe']['Promoteur']['f1']:.2f} | {m['par_classe']['Promoteur']['effectif']} |

Kappa quadratique **{m['kappa_quadratique']:.3f}**, macro-F1 {m['macro_f1']:.3f}, AUC Détracteur {m.get('auc_detracteur', float('nan')):.2f} (baseline logistique : kappa {k_log:.3f}). Le Passif, milieu ambigu
et rare chez les répondants, est le plus difficile — comme dans la littérature. La calibration des probabilités a été testée puis
**corrigée** : ajustée sur des répondants biaisés, elle {'écrasait la classe Passif et ' if mf['calibration']['passif_ecrase'] else ''}surestime P(Détracteur). Décision : les probabilités servent à **classer**
les clients, pas à lire une fréquence. Robustesse vérifiée : {br[-1]['taux_bruit']:.0%} d'étiquettes corrompues coûtent {br[0]['kappa'] - br[-1]['kappa']:.3f} de kappa ; les
conclusions sur les drivers tiennent sous les trois mappings de la cible. {texte_txt}

---

## 4. Ce que ça vaut pour l'équipe rétention

{_img("05_calibration_gain.png", "Figure 3 — Gauche : fiabilité des probabilités (sous la diagonale : surestimées). Droite : part des détracteurs captés selon le nombre d'appels.")}

À K = {pk['k']} appels par mois : **précision {pk['precision_at_k']:.0%}**, rappel {pk['rappel_at_k']:.0%}, lift ×{pk['lift']:.1f}. Avec un coût d'appel de {mk['cout_appel_eur']} €, une valeur de
client retenu de {mk['valeur_client_retenu_eur']} € et un taux de succès de {mk['taux_succes_appel']:.0%} — **paramètres à valider avec l'équipe** — la campagne ciblée rapporte
{_eur(eco['gain_net_eur'])} nets contre {_eur(eco['gain_net_ciblage_aleatoire_eur'])} au hasard. `CLTV`, interdite comme variable du modèle, est proposée en option pour pondérer *qui appeler*.

**Le NPS des silencieux.** Le modèle l'estime à **{ns['nps_silencieux_predit']:+.0f}** (IC 95 % {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f}) ; la vérité, connue ici, est {ns['nps_silencieux_vrai']:+.0f}
({ns['nps_silencieux_predit'] - ns['nps_silencieux_vrai']:+.0f} pts). {lecture_nps}

{_img("05_nps_simule.png", "Figure 4 — Le NPS que l'opérateur observe (répondants), celui que le modèle estime pour les silencieux, la vérité.", "78%")}

**Ce qui fait un détracteur, selon le modèle** : contrat sans engagement, faible ancienneté, absence de parrainage, facture élevée —
associations robustes au mapping de la cible et cohérentes avec la littérature. Elles ne pèsent pas partout pareil : les drivers sont
donnés **par segment** (contrat mensuel / engagé, client récent / ancien, fibre / DSL). Pour chaque détracteur prédit, l'application
recommande **un levier** : engagement si contrat mensuel, revue tarifaire si charge par service élevée, parcours d'accueil si client
récent, support premium sinon. Ce sont des **associations**, pas des causes — l'offre E « prédit » le départ parce qu'elle a été
proposée à des clients fragiles.

---

## 5. Équité

{_img("05_equite.png", "Figure 5 — Rappel Détracteur (bleu) et taux de sélection (gris) par sous-groupe. Variables exclues du modèle, utilisées uniquement pour cet audit.")}

L'écart de **{eq['tranche_age']['ecart_max']*100:.0f} points** de rappel entre tranches d'âge ({age['<30']['rappel_detracteur']:.0%} chez les moins de 30 ans, {age['60+']['rappel_detracteur']:.0%} chez les 60 ans et plus) est le point à
escalader : c'est le cas que l'énoncé décrit comme inacceptable, en sens inverse. Sa cause a été mesurée : les variables retenues
**reconstruisent** l'âge (AUC {px.get('Moins de 30 ans', {}).get('auc', float('nan')):.2f}) et la situation familiale (AUC {px.get('Marié(e)', {}).get('auc', float('nan')):.2f}) — exclure une colonne n'empêche pas le modèle de la voir. Genre et
densité de zone ne sont pas reconstructibles et ne montrent pas d'écart. Une **mitigation par seuils de décision par tranche d'âge** a
été chiffrée : elle égalise le rappel ({mit['rappel_cible']:.0%} partout) au prix de la précision chez les jeunes et de {mit['appels_apres'] - mit['appels_avant']:+d} appels. Traiter différemment
selon l'âge est un arbitrage juridique et métier : non appliqué par défaut, remonté à l'Expérience Client et au juridique. L'audit est
dans le monitoring et se répète à chaque réentraînement.

---

## 6. Limites, décision, suites

**Ce qu'on ne peut pas faire avec ces données** : estimer l'effet d'un appel. Aucune campagne enregistrée, aucune assignation aléatoire.
Ce modèle classe par **risque** ; l'optimum économique classe par **sensibilité à l'intervention** — les clients qui restent *parce
qu'on les a appelés*, pas ceux qui étaient déjà décidés.

**La décision.** À la prochaine campagne, ne pas contacter 10 à 20 % des clients ciblés, tirés au hasard. Ce groupe de contrôle achète la
mesure de l'effet réel de la campagne et les données qui permettraient, au cycle suivant, de cibler par sensibilité. Sans lui, un
détracteur appelé puis retenu change de classe et le modèle réapprend l'effet de sa propre intervention — il se dégrade sans que
personne ne voie pourquoi.

**Monitoring.** {mo_txt}

**Limites assumées** : hypothèses économiques à valider ; propension à répondre simulée (plausible, non observée) ; taux de réponse de
15 % optimiste (2,6 % chez Kannan et al.) ; cadrage panafricain plaqué sur un dataset californien ; verbatims synthétiques sans clé API ;
TabPFN non évalué faute d'accès aux poids.

**Reproductibilité et livraison.** `python -m src.run` régénère données, cible, quinze modèles, sélection et évaluation ; `python -m
src.rapports` l'étude complète (huit documents, une vingtaine de figures) ; l'application Streamlit (sept pages) consomme les mêmes
artefacts ; tests automatisés ; seed unique ; aucune clé dans le dépôt. {repro_txt}

**Usage d'outils d'IA** (énoncé § 7) : code, structure de l'étude et rédaction produits avec l'assistance d'un assistant IA (Claude),
sous direction et relecture humaines ; fragments des verbatims synthétiques rédigés par ce même assistant. Les choix de modélisation,
les décisions de périmètre et les conclusions sont assumés par l'auteur.

---

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
figure { margin: 8px 0; text-align: center; page-break-inside: avoid; }
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
    parties = ["# Étude complète — Prédiction de la catégorie NPS des clients silencieux\n\n"
               "*Annexe au write-up. Un document par phase CRISP-DM, généré par `python -m src.rapports`.*\n"]
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
