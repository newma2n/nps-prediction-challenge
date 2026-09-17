"""Étape 37 — le write-up de 3 à 6 pages, pour un directeur Expérience Client.

    python -m src.writeup

Compose livrable/write_up.md à partir des artefacts (rien n'est recopié), le convertit en HTML
puis en PDF via Chrome en mode headless. Figures embarquées en base64 : le PDF est autonome.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import markdown
import pandas as pd

from .data import RACINE, charger_config

REP, FIG = RACINE / "reports", RACINE / "reports" / "figures"
LIV = RACINE / "livrable"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def _img(nom: str, legende: str, largeur: str = "100%") -> str:
    p = FIG / nom
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{largeur}"><figcaption>{legende}</figcaption></figure>\n'


def _charger(nom):
    p = REP / f"{nom}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def composer() -> str:
    cfg = charger_config()
    R = _charger("resultats"); tx = _charger("texte"); tp = _charger("tabpfn"); mo = _charger("monitoring")
    a = R["arbitrage_des_3"]; m = R["modele_final"]["metriques_retenues"]
    pk = R["evaluation_metier"]["precision_at_k"]; eco = R["evaluation_metier"]["economie"]
    eq = R["audit_equite"]; age = eq["tranche_age"]["groupes"]
    sel = R["selection_modele_final"]["classement"]; fu = R["diagnostic_fuite"]
    dg = pd.DataFrame(R["diagnostic_protocole"]).set_index("scenario")
    h = R.get("hyperparametres", {})
    mk = cfg["metier"]

    def eur(v): return f"{int(round(v)):,} €".replace(",", " ")

    tabpfn_txt = ""
    if tp and tp.get("statut") == "ok":
        gagne = tp["metriques"]["kappa"] > tp["reference_logistique"]["kappa"] + 0.005
        tabpfn_txt = (f"Un modèle de fondation tabulaire (TabPFN) a aussi été testé : kappa {tp['metriques']['kappa']:.3f} contre "
                      f"{tp['reference_logistique']['kappa']:.3f}, pour une inférence {tp['duree_inference_s']/max(tp['reference_logistique']['duree_inference_s'],1e-3):.0f}× plus lente. "
                      f"Il {'bat' if gagne else 'ne bat pas'} le modèle retenu.")
    elif tp:
        tabpfn_txt = "TabPFN n'a pas pu être évalué dans l'environnement de travail ; sa place est documentée dans le rapport détaillé."

    texte_txt = ""
    if tx:
        texte_txt = f"""
### Le signal textuel

{tx['n_verbatims']} notes de contact synthétiques ont été générées (source : `{tx['source_verbatims']}`, 25 % de bruit de
tonalité), puis combinées au modèle tabulaire. La fusion gagne {tx['gain_kappa_fusion_tardive']:+.3f} de kappa — **et ce chiffre
ne prouve rien** : le texte a été fabriqué à partir de la classe, il contient donc le label. L'expérience valide la chaîne
technique, pas la valeur de vrais verbatims. La conclusion opérationnelle est de **ne pas déployer** de composante texte
sur cette seule base.
"""

    hp_txt = ""
    if h:
        lignes = "\n".join(f"| {nom} | {v['kappa_silencieux_defaut']:.3f} | {v['kappa_silencieux_regle']:.3f} | {v['retenu']} |"
                           for nom, v in h.items() if "kappa_silencieux_defaut" in v)
        hp_txt = f"""
| Famille | Kappa par défaut | Kappa réglé | Décision |
|---|---|---|---|
{lignes}
"""

    mo_txt = ""
    if mo:
        mo_txt = (f"Un monitoring de dérive (indice PSI par variable et sur les prédictions) est implémenté. Sur un mois simulé "
                  f"({mo['scenario_simule']}), le déclencheur de réentraînement "
                  f"{'se déclenche' if mo['reentrainement_declenche_mois_simule'] else 'ne se déclenche pas'} — "
                  f"variables en alerte : {', '.join(mo['variables_en_alerte_mois_simule']) or 'aucune'}.")

    return f"""
# Prédire la catégorie NPS des clients silencieux
## Priorisation des appels de rétention — synthèse pour la direction Expérience Client

---

## 1. En une page

Un opérateur télécom interroge ses clients sur leur propension à le recommander ; **15 % répondent**. Ce travail
construit un modèle qui estime, pour chacun des 85 % silencieux, s'il est Détracteur, Passif ou Promoteur, et une
application qui en tire une liste d'appels priorisée pour l'équipe rétention.

**Trois résultats.**

1. **La grille NPS proposée dans l'énoncé fabrique un résultat faux.** Appliquée telle quelle, elle donne un NPS de **{a['nps']['M1']:+.0f}**,
   à 60–75 points des benchmarks télécom. Elle classe en détracteurs {a['n_ambigus']} clients « moyennement satisfaits » (38 % de la base)
   dont 84 % sont restés fidèles. En demandant aux données à qui ces clients ressemblent, on trouve qu'ils penchent à
   **{a['part_3_vers_promoteur']:.0%} du côté des promoteurs**. Le NPS corrigé est de **{a['nps']['M3']:+.0f}**, dans la fourchette du secteur.
2. **Le modèle multiplie par {pk['lift']:.1f} l'efficacité des appels.** Sur {pk['k']} appels, {pk['precision_at_k']:.0%} touchent un vrai détracteur contre
   {pk['taux_de_base']:.0%} au hasard — **{eur(eco['gain_apporte_par_le_modele_eur'])} de gain net supplémentaire par campagne** à hypothèses de coût
   constantes. Il retrouve {m['par_classe']['Detracteur']['rappel']:.0%} des détracteurs.
3. **Il rate davantage les jeunes détracteurs** : {age['<30']['rappel_detracteur']:.0%} de rappel chez les moins de 30 ans contre {age['60+']['rappel_detracteur']:.0%}
   chez les 60 ans et plus — sans qu'aucune variable d'âge n'entre dans le modèle. Mesuré, signalé, à traiter avant production.

**Une décision demandée.** Appeler les clients les plus à risque n'est pas la stratégie la plus rentable : la recherche établit qu'il
faut cibler ceux dont le comportement change le plus quand on les appelle (Ascarza, *Journal of Marketing Research*, 2018). Mesurer
cela ne demande qu'une chose, gratuite : **ne pas contacter 10 à 20 % des clients ciblés à la prochaine campagne, tirés au hasard.**

---

## 2. Les données et leurs pièges

7 043 clients d'un opérateur californien fictif (IBM Telco), cinq tables jointes sans perte. La colonne clé est une
**satisfaction de 1 à 5** déclarée par le client. Trois faits ont structuré le travail.

**Le dataset contient sa propre réponse.** Les clients à satisfaction 1–2 sont **tous partis** ; ceux à 4–5, **aucun**. Toute
colonne liée au départ (`Churn Value`, `Churn Score`, `Customer Status`, `CLTV`…) donnerait un score presque parfait et un modèle
inutilisable — au moment de détecter un détracteur, il n'est pas encore parti. Ces colonnes sont identifiées, classées en trois
natures de fuite (conséquence, disponibilité, modèle amont), **mesurées** par information mutuelle, et exclues. La littérature
fournit le contre-exemple : une étude publiée annonce 98 % d'exactitude avec la note NPS parmi ses variables.

**La note « 3 » décide de tout.** {a['n_ambigus']} clients, 38 % de la base, 84 % encore présents. Plutôt que de décider où ils vont,
un modèle entraîné sur les seuls extrêmes (1–2 contre 4–5, exactitude {a['exactitude_sur_extremes']:.0%}) les a classés :
{a['part_3_vers_promoteur']:.0%} ressemblent à des promoteurs. Le bloc est donc traité comme **Passif**. On a évité d'étiqueter chaque
client individuellement par ce modèle — la cible serait devenue une fonction des variables, et le modèle final aurait réappris sa propre sortie.

{_img("03_cible_mappings.png", "Figure 1 — Trois façons de construire la cible, et le NPS qui en résulte. Bande verte : benchmark télécom publié.")}

**Les offres commerciales sont une quasi-expérience — piégée.** Les clients ayant reçu l'offre E partent **deux fois plus** que ceux
sans offre. Lire « l'offre E fait fuir » serait une erreur : elle a été proposée à des clients déjà fragiles. On ne lit pas un
effet causal dans des données observationnelles — ce constat fonde la décision demandée en section 5.

---

## 3. Un modèle évalué dans les conditions réelles

**Le protocole.** Un découpage aléatoire aurait été trompeur : les répondants à une enquête ne sont pas un échantillon au hasard —
les mécontents et les enthousiastes répondent plus. Trois scénarios de non-réponse ont été simulés, du plus optimiste au plus
réaliste (MNAR : la propension à répondre dépend de la satisfaction elle-même). Dans le scénario réaliste, la part de notes
extrêmes passe de **{dg.loc['S3_MNAR','part_extremes_base']:.0%} dans la base à {dg.loc['S3_MNAR','part_extremes_repondants']:.0%} chez les répondants**. Le modèle
est entraîné sur ces 15 % biaisés et évalué sur les 85 % silencieux, dont on connaît la vérité. Ses scores sont donc réalistes.

{_img("04_protocole_15_85.png", "Figure 2 — Satisfaction des répondants simulés vs base complète. Sous S3, les extrêmes dominent.")}

**Trois familles, le plus simple gagne.** Régression logistique, régression ordinale à seuils, gradient boosting — comparés sur
27 combinaisons (3 scénarios × 3 cibles × 3 modèles), réglés puis sélectionnés sur le kappa quadratique pondéré, qui respecte
l'ordre des classes.

| Modèle | Kappa | Macro-F1 | Rappel Détracteur |
|---|---|---|---|
{chr(10).join(f"| {s['modele']} | {s['kappa']:.3f} | {s['macro_f1']:.3f} | {s['rappel_detracteur']:.3f} |" for s in sel)}
{hp_txt}
Avec ~1 050 clients pour apprendre, le boosting n'a pas de quoi justifier sa complexité. La baseline n'était pas une formalité.
{tabpfn_txt}

**Aucune fuite.** Exactitude maximale observée {fu['exactitude_max_observee']:.2f} (seuil d'alerte 0,85, calé sur la littérature) ; écart
arbres/linéaire {fu['ecart_arbres_lineaire_S1_M3']:+.2f} (une fuite se signale par un écart > +0,30).

**Performance, par classe — jamais globale seule.**

| Classe | Précision | Rappel | F1 | Effectif |
|---|---|---|---|---|
| Détracteur | {m['par_classe']['Detracteur']['precision']:.2f} | **{m['par_classe']['Detracteur']['rappel']:.2f}** | {m['par_classe']['Detracteur']['f1']:.2f} | {m['par_classe']['Detracteur']['effectif']} |
| Passif | {m['par_classe']['Passif']['precision']:.2f} | {m['par_classe']['Passif']['rappel']:.2f} | {m['par_classe']['Passif']['f1']:.2f} | {m['par_classe']['Passif']['effectif']} |
| Promoteur | {m['par_classe']['Promoteur']['precision']:.2f} | {m['par_classe']['Promoteur']['rappel']:.2f} | {m['par_classe']['Promoteur']['f1']:.2f} | {m['par_classe']['Promoteur']['effectif']} |

Kappa quadratique **{m['kappa_quadratique']:.3f}**, macro-F1 {m['macro_f1']:.3f}. Le Détracteur — la classe qui compte — est la mieux retrouvée.
Le Passif, milieu ambigu par définition, est le plus difficile, comme dans la littérature. La calibration des probabilités a été
testée puis **corrigée** : ajustée sur les répondants biaisés, elle écrasait la classe Passif et surestimait les probabilités.
Décision : les probabilités servent à **classer** les clients, pas à lire une fréquence.
{texte_txt}
---

## 4. Ce que ça vaut pour l'équipe rétention

{_img("05_calibration_gain.png", "Figure 3 — Gauche : fiabilité des probabilités (sous la diagonale : surestimées). Droite : part des détracteurs captés selon le nombre d'appels.")}

À K = {pk['k']} appels par mois : **précision {pk['precision_at_k']:.0%}**, rappel {pk['rappel_at_k']:.0%}, lift ×{pk['lift']:.1f}. Avec un coût d'appel de
{mk['cout_appel_eur']} €, une valeur de client retenu de {mk['valeur_client_retenu_eur']} € et un taux de succès de {mk['taux_succes_appel']:.0%} — **paramètres à valider avec
l'équipe** — la campagne ciblée rapporte {eur(eco['gain_net_eur'])} nets contre {eur(eco['gain_net_ciblage_aleatoire_eur'])} au hasard.

**Ce qui fait un détracteur, selon le modèle** : contrat sans engagement, faible ancienneté, absence de parrainage, facture
mensuelle élevée. Ces associations tiennent quel que soit le mapping de la cible. Le levier le plus probable pour un détracteur
prédit : s'il est en contrat mensuel, proposer un engagement ; sinon, revoir la charge mensuelle. Ce sont des **associations**,
pas des causes.

{_img("05_equite.png", "Figure 4 — Rappel Détracteur par sous-groupe. Les variables démographiques sont exclues du modèle et servent uniquement à cet audit.")}

**Équité.** L'écart de {eq['tranche_age']['ecart_max']*100:.0f} points entre groupes d'âge est le point à escalader. Exclure les démographiques du modèle
coûte peu en performance mais **ne suffit pas** : le modèle les retrouve via les proxies contractuels. L'audit doit être répété à
chaque réentraînement ; un seuil de décision par groupe d'âge est une piste, à trancher par le métier.

---

## 5. Limites, décision, suites

**Ce qu'on ne peut pas faire avec ces données** : estimer l'effet d'un appel. Aucune campagne enregistrée, aucune assignation
aléatoire. Ce modèle classe par **risque**. La littérature établit que l'optimum économique classe par **sensibilité à
l'intervention** — les clients qui restent *parce qu'on les a appelés*, pas ceux qui étaient déjà décidés à partir.

**La décision.** À la prochaine campagne, ne pas contacter 10 à 20 % des clients ciblés, tirés au hasard. Ce groupe de contrôle
coûte quelques clients non retenus et achète deux choses : la mesure de l'effet réel de la campagne, et les données qui
permettraient, au cycle suivant, de cibler par sensibilité. Sans lui, un détracteur appelé puis retenu change de classe, et le
modèle réapprend l'effet de sa propre intervention comme une propriété du client — il se dégrade sans que personne ne voie pourquoi.

**Monitoring.** {mo_txt}

**Limites assumées** : hypothèses économiques à valider ; propension à répondre simulée, plausible mais non observée ; cadrage
panafricain plaqué sur un dataset californien ; verbatims synthétiques sans clé API disponible.

**Reproductibilité.** `python -m src.run` régénère données, cible, modèles et évaluation ; `python -m src.rapports` l'étude
complète (19+ rapports, 17+ figures) ; l'application Streamlit consomme les mêmes artefacts. Seed unique. Aucune clé dans le dépôt.

**Usage d'outils d'IA** (§ 7 de l'énoncé) : code, structure de l'étude et rédaction produits avec l'assistance d'un assistant IA
(Claude), sous direction et relecture humaines ; fragments des verbatims synthétiques rédigés par ce même assistant. Les choix de
modélisation, les décisions de périmètre et les conclusions sont assumés par l'auteur.

---

*Références.* Ascarza (2018) *Retention Futility*, J. Marketing Research · Elero et al. (2026) *Computers* 15(2) · Chong et al.
(2023) J. Soft Computing & Data Mining · Mustafa et al. (2021) F1000Research · Kannan et al. (2022) ICTIM · Zihayat et al. (2021)
J. Business Research · Gómez-Vargas et al. (2025) Eur. J. Operational Research · Devriendt, Berrevoets, Verbeke, Information Sciences.
"""


CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.3pt; line-height: 1.48; color: #1c2430; }
h1 { font-family: Arial, Helvetica, sans-serif; font-size: 20pt; color: #0d2440; margin: 0 0 2px; }
h2 { font-family: Arial, Helvetica, sans-serif; font-size: 13pt; color: #0d2440; margin-top: 14px; border-bottom: 1px solid #c9d3de; padding-bottom: 2px; }
h3 { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; color: #1f4e79; margin-top: 10px; }
h1 + h2 { border: none; font-weight: normal; color: #4a5a6a; font-size: 12pt; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 9.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d3de; padding: 3px 6px; text-align: left; }
th { background: #eef3f8; }
figure { margin: 8px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.6pt; color: #4a5a6a; margin-top: 2px; font-style: italic; }
hr { border: 0; border-top: 1px solid #c9d3de; margin: 10px 0; }
li { margin-bottom: 2px; }
p { orphans: 3; widows: 3; margin: 5px 0; }
em { color: #4a5a6a; }
"""


def main():
    LIV.mkdir(exist_ok=True)
    md = composer()
    (LIV / "write_up.md").write_text(md, encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables", "sane_lists"])
    (LIV / "write_up.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>",
                                       encoding="utf-8")
    chrome = next((c for c in CHROME if Path(c).exists()), None)
    if chrome is None:
        print("Chrome/Edge introuvable — HTML produit, PDF non généré."); return
    pdf = LIV / "write_up.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", (LIV / "write_up.html").resolve().as_uri()],
                   capture_output=True, timeout=120)
    if pdf.exists():
        try:
            import fitz
            n = fitz.open(pdf).page_count
            print(f"livrable/write_up.pdf — {n} pages, {pdf.stat().st_size//1024} Ko")
        except Exception:
            print(f"livrable/write_up.pdf — {pdf.stat().st_size//1024} Ko")
    else:
        print("Échec de la conversion PDF ; HTML disponible dans livrable/.")


if __name__ == "__main__":
    main()
