"""Application Streamlit — outil de priorisation pour l'équipe rétention.

    streamlit run app/app.py

Sept pages, adressables par ?page=. Toutes les valeurs affichées sont lues dans les artefacts
produits par le pipeline (reports/*.json, models/, data/processed/) — rien n'est recopié.
Chaque page : les chiffres d'abord, la lecture ensuite, le détail dans des dépliants.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from src.explication import agreger_par_variable, contributions_detracteur, libelle_variable  # noqa: E402
from src.run import levier, leviers_ordonnes  # noqa: E402

REP, FIG = RACINE / "reports", RACINE / "reports" / "figures"
LIB = {"Detracteur": "Détracteur", "Passif": "Passif", "Promoteur": "Promoteur"}
COUL = {"Detracteur": "#c0392b", "Passif": "#e67e22", "Promoteur": "#27ae60"}

st.set_page_config(page_title="NPS · Rétention", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .block-container {padding-top: 1.1rem; max-width: 1280px;}
  .kpi {border:1px solid #e3e6ea; border-radius:10px; padding:13px 15px; background:#fafbfc; height:100%;}
  .kpi .lab {font-size:11.5px; color:#5f6b7a; text-transform:uppercase; letter-spacing:.04em;}
  .kpi .val {font-size:25px; font-weight:700; color:#1f2d3d; margin-top:2px;}
  .kpi .sub {font-size:12px; color:#7a8794; margin-top:2px;}
  .verdict {display:inline-block; padding:10px 16px; border-radius:8px; color:#fff; font-weight:700; font-size:20px;}
  .note {border-left:4px solid #2c7fb8; background:#f3f8fc; padding:10px 14px; border-radius:6px; font-size:14px; margin-bottom:8px;}
  .warn {border-left:4px solid #e67e22; background:#fdf6ee; padding:10px 14px; border-radius:6px; font-size:14px; margin-bottom:8px;}
  .bad  {border-left:4px solid #c0392b; background:#fbf0ee; padding:10px 14px; border-radius:6px; font-size:14px; margin-bottom:8px;}
  .ok   {border-left:4px solid #27ae60; background:#eefaf2; padding:10px 14px; border-radius:6px; font-size:14px; margin-bottom:8px;}
  h2 {margin-top:.6rem;} h3 {margin-top:.9rem;}
  section[data-testid="stSidebar"] .stRadio label {font-size:15px;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ chargement
@st.cache_resource(show_spinner="Chargement du modèle et des résultats…")
def charger():
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    clients = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    res = json.load(open(REP / "resultats.json", encoding="utf-8"))
    extra = {}
    for nom in ("texte", "fondation", "monitoring", "reproductibilite"):
        p = REP / f"{nom}.json"
        extra[nom] = json.load(open(p, encoding="utf-8")) if p.exists() else None
    pv = RACINE / "data" / "processed" / "verbatims.parquet"
    verbatims = pd.read_parquet(pv).set_index("Customer ID") if pv.exists() else None
    pt = RACINE / "models" / "modele_texte.joblib"
    texte = joblib.load(pt) if pt.exists() else None
    cfg = __import__("yaml").safe_load(open(RACINE / "config.yaml", encoding="utf-8"))
    return art, clients, res, extra, verbatims, texte, cfg


art, clients, res, extra, verbatims, modele_texte, cfg = charger()
pipe_cal, pipe_brut, pipe_dec = art["pipeline"], art["pipeline_brut"], art.get("pipeline_decision", art["pipeline_brut"])
classes, num, cat = art["classes"], art["num"], art["cat"]
i_det = classes.index("Detracteur")
mf = res["modele_final"]; sel = res["selection_modele_final"]
K = res["evaluation_metier"]["precision_at_k"]["k"]
silencieux = clients[~clients["repondant_S3"]]
NOM_MODELE = mf["modele"].replace("_", " ")


# ============================================================================ utilitaires
def kpi(col, label, valeur, sous=""):
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{valeur}</div><div class="sub">{sous}</div></div>', unsafe_allow_html=True)


def image(nom: str, legende: str = ""):
    p = FIG / nom
    if p.exists():
        st.image(str(p), caption=legende, width="stretch")
    else:
        st.info(f"Figure absente : {nom} — lancer `python -m src.rapports`.")


def encart(texte: str, genre: str = "note"):
    st.markdown(f'<div class="{genre}">{texte}</div>', unsafe_allow_html=True)


def eur(v) -> str:
    return f"{int(round(v)):,} €".replace(",", " ")


def pct(v, d=0) -> str:
    return f"{v*100:.{d}f} %"


def _degrade(styler, colonne: str):
    """Dégradé de couleur sur une colonne. Il passe par matplotlib : si le paquet manque, on rend
    le tableau sans dégradé plutôt que de faire tomber la page."""
    try:
        return styler.background_gradient(subset=[colonne], cmap="Reds")
    except ImportError:
        return styler


def tableau(df: pd.DataFrame, fmt: dict | None = None, hauteur: int | None = None):
    kw = {"height": hauteur} if hauteur else {}
    st.dataframe(df.style.format(fmt or {}), hide_index=True, width="stretch", **kw)


def resume_vers_df(lignes: list[dict], cle_libelle: str, libelle: str) -> pd.DataFrame:
    d = pd.DataFrame([l for l in lignes if "kappa" in l])
    cols = {cle_libelle: libelle, "kappa": "Kappa", "macro_f1": "Macro-F1", "rappel_detracteur": "Rappel Dét.", "rappel_passif": "Rappel Passif",
            "rappel_promoteur": "Rappel Prom.", "precision_detracteur": "Précision Dét.", "auc_detracteur": "AUC Dét."}
    d = d[[c for c in cols if c in d]].rename(columns=cols)
    return d


FMT3 = {c: "{:.3f}" for c in ("Kappa", "Macro-F1", "Rappel Dét.", "Rappel Passif", "Rappel Prom.", "Précision Dét.", "AUC Dét.")}


# ============================================================================ filtres croisés
# Un seul jeu de filtres, dans la barre latérale, appliqué à toutes les pages montrant des clients.
CHAMPS_CAT = [("Contract", "Contrat"), ("Internet Type", "Type d'internet"), ("Offer", "Offre reçue"),
              ("prediction", "Classe prédite"), ("tranche_age", "Tranche d'âge"),
              ("levier_recommande", "Levier recommandé"), ("levier_secondaire", "Second levier"),
              ("Payment Method", "Moyen de paiement")]
CHAMPS_NUM = [("Tenure in Months", "Ancienneté (mois)", 0), ("Monthly Charge", "Facture mensuelle ($)", 0),
              ("Number of Referrals", "Parrainages", 0)]


def _defauts() -> dict:
    d = {f"f_{c}": [] for c, _ in CHAMPS_CAT}
    d["f_proba"] = (0.0, 1.0)
    d["f_recherche"] = ""
    d["f_perimetre"] = "Clients silencieux (les 85 % à prédire)"
    for c, _, _ in CHAMPS_NUM:
        d[f"f_{c}"] = None
    return d


def reinitialiser_filtres():
    for k, v in _defauts().items():
        st.session_state[k] = v


def panneau_filtres(base_complete: pd.DataFrame, silencieux_: pd.DataFrame):
    """Dessine les filtres dans la barre latérale et renvoie le sous-ensemble sélectionné."""
    for k, v in _defauts().items():
        st.session_state.setdefault(k, v)

    st.markdown("### Filtres")
    st.caption("Ils s'appliquent à toutes les pages qui montrent des clients. Chaque filtre se croise avec les autres.")
    perim = st.radio("Périmètre", ["Clients silencieux (les 85 % à prédire)", "Base complète (7 043 clients)"],
                     key="f_perimetre", label_visibility="collapsed")
    d = silencieux_ if perim.startswith("Clients silencieux") else base_complete

    sel = d
    for col, lib in CHAMPS_CAT:
        if col not in d.columns:
            continue
        opts = sorted(str(v) for v in d[col].dropna().unique())
        if len(opts) < 2:
            continue
        aff = [LIB.get(o, o) for o in opts] if col == "prediction" else opts
        corresp = dict(zip(aff, opts))
        choix = st.multiselect(lib, aff, key=f"f_{col}", placeholder="tous")
        if choix:
            sel = sel[sel[col].astype(str).isin([corresp[c] for c in choix])]

    with st.expander("Plages de valeurs", expanded=False):
        for col, lib, dec in CHAMPS_NUM:
            if col not in d.columns:
                continue
            lo, hi = float(d[col].min()), float(d[col].max())
            if lo == hi:
                continue
            # L'état est initialisé avant la création du widget : avec une `key`, passer `value`
            # fait lire l'état existant, et un état à None fait échouer la sérialisation du
            # curseur. La valeur mémorisée est bornée au périmètre courant.
            cle = f"f_{col}"
            cur = st.session_state.get(cle)
            if not isinstance(cur, (tuple, list)) or len(cur) != 2 or cur[0] is None or cur[1] is None:
                st.session_state[cle] = (lo, hi)
            else:
                st.session_state[cle] = (min(max(lo, float(cur[0])), hi), max(min(hi, float(cur[1])), lo))
            bornes = st.slider(lib, lo, hi, key=cle)
            sel = sel[sel[col].between(*bornes)]
        if not isinstance(st.session_state.get("f_proba"), (tuple, list)):
            st.session_state["f_proba"] = (0.0, 1.0)
        pmin, pmax = st.slider("P(détracteur)", 0.0, 1.0, step=0.05, key="f_proba")
        sel = sel[sel["proba_detracteur"].between(pmin, pmax)]

    q = st.text_input("Rechercher un identifiant client", key="f_recherche", placeholder="ex. 0002-ORFBO")
    if q.strip():
        sel = sel[sel["Customer ID"].astype(str).str.contains(q.strip(), case=False, na=False)]

    n, tot = len(sel), len(d)
    st.markdown(f"**{n} client{'s' if n > 1 else ''}** sélectionné{'s' if n > 1 else ''} sur {tot} "
                f"({n / max(tot, 1):.0%} du périmètre)")
    if n == 0:
        st.warning("Aucun client ne satisfait ces filtres. Élargissez-en un.")
    st.button("Réinitialiser les filtres", on_click=reinitialiser_filtres, width="stretch")
    return sel, d, perim


def nps_de_serie(colonne) -> float:
    """NPS d'un ensemble de classes : % promoteurs − % détracteurs, en points."""
    v = pd.Series(colonne).astype(str)
    if not len(v):
        return float("nan")
    return 100 * float((v == "Promoteur").mean() - (v == "Detracteur").mean())


# ============================================================================ navigation
PAGES = ["Synthèse", "Explorer les clients", "Prioriser les appels", "Analyser un client", "Données et cible",
         "Modèles et performance", "Drivers et équité", "Suivi et méthode"]
with st.sidebar:
    st.markdown("## NPS · Rétention")
    demande = st.query_params.get("page", PAGES[0])
    page = st.radio("Pages", PAGES, index=PAGES.index(demande) if demande in PAGES else 0, label_visibility="collapsed")
    if page != demande:
        st.query_params["page"] = page
    st.divider()
    filtre, perimetre_df, nom_perimetre = panneau_filtres(clients, silencieux)
    st.divider()
    st.caption(f"**Modèle retenu** {NOM_MODELE} · {mf['version']}  \n**Cible** M3 · **Scénario** S3 (MNAR)  \n"
               f"Entraîné sur {mf['n_entrainement']} répondants · évalué sur {mf['n_evaluation']} silencieux  \n"
               f"Kappa {mf['metriques_retenues']['kappa_quadratique']:.3f} · seed {res['seed']}")


# ============================================================================ 1. Synthèse
def page_synthese():
    st.title("Synthèse")
    a = res["arbitrage_des_3"]; pk = res["evaluation_metier"]["precision_at_k"]; eco = res["evaluation_metier"]["economie"]
    m = mf["metriques_retenues"]; ns = res["nps_simule"]; eq = res["audit_equite"]["tranche_age"]["groupes"]
    c = st.columns(5)
    kpi(c[0], "NPS estimé — base complète", f"{a['nps']['M3']:+.0f}", f"grille de l'énoncé : {a['nps']['M1']:+.0f} · secteur +19 à +34")
    kpi(c[1], "NPS des 85 % silencieux", f"{ns['nps_silencieux_predit']:+.0f}", f"IC 95 % {ns['nps_silencieux_predit_ic95'][0]:+.0f} à {ns['nps_silencieux_predit_ic95'][1]:+.0f} · répondants seuls {ns['nps_repondants_observe']:+.0f}")
    kpi(c[2], f"Précision @ {pk['k']} appels", pct(pk["precision_at_k"]), f"taux de base {pct(pk['taux_de_base'])} · lift ×{pk['lift']:.1f}")
    kpi(c[3], "Gain net apporté / campagne", eur(eco["gain_apporte_par_le_modele_eur"]), f"{pk['k']} appels · hypothèses à valider")
    kpi(c[4], "Détracteurs retrouvés", pct(m["par_classe"]["Detracteur"]["rappel"]), f"{pct(eq['<30']['rappel_detracteur'])} chez les moins de 30 ans")

    st.markdown("### Trois résultats à retenir")
    g1, g2, g3 = st.columns(3)
    with g1:
        encart(f"<b>La grille de l'énoncé tire le NPS vers le bas.</b> Appliquée telle quelle : <b>{a['nps']['M1']:+.0f}</b>. Elle classe en détracteurs "
               f"{a['n_ambigus']} clients « moyens » dont 84 % sont restés. Un modèle entraîné sur les seuls extrêmes montre qu'ils penchent à "
               f"<b>{pct(a['part_3_vers_promoteur'])} côté promoteur</b>. NPS corrigé : <b>{a['nps']['M3']:+.0f}</b>.")
    with g2:
        encart(f"<b>Le modèle rend les appels {pk['lift']:.1f} fois plus efficaces.</b> {len(sel['classement'])} modèles de "
               f"7 familles ont été comparés ; le choix repose sur la validation croisée des répondants, sans utiliser les clients d'évaluation, "
               f"avec une règle de parcimonie. Retenu : <b>{NOM_MODELE}</b>. Gain net en plus : <b>{eur(eco['gain_apporte_par_le_modele_eur'])}</b> par campagne.")
    with g3:
        encart(f"<b>Il repère moins bien les détracteurs jeunes.</b> Rappel {pct(eq['<30']['rappel_detracteur'])} chez les moins de 30 ans contre "
               f"{pct(eq['60+']['rappel_detracteur'])} chez les 60+, sans variable d'âge : les features contractuelles <b>reconstruisent l'âge</b> "
               f"(AUC {res['audit_proxies'].get('Moins de 30 ans', {}).get('auc', float('nan')):.2f}). Une correction est chiffrée plus bas ; la décision revient au métier.", "warn")

    c1, c2 = st.columns(2)
    with c1:
        image("05_nps_simule.png", "Le NPS que l'opérateur voit (répondants), celui que le modèle estime pour les silencieux, et la vérité — connue ici seulement")
    with c2:
        image("05_calibration_gain.png", "Gauche : fiabilité des probabilités (sous la diagonale = surestimées). Droite : part des détracteurs captés selon le nombre d'appels")

    st.markdown("### La décision demandée à la direction")
    encart("<b>Appeler les clients les plus à risque n'est pas la stratégie la plus rentable</b> (Ascarza, <i>Journal of Marketing Research</i>, 2018 : "
           "jusqu'à 7 points de churn en moins en ciblant par <i>sensibilité à l'appel</i>). Pour le mesurer, il suffit de <b>ne pas "
           "contacter 10 à 20 % des clients ciblés à la prochaine campagne, tirés au hasard</b>. Sans ce groupe de contrôle, on ne pourra pas savoir si la "
           "campagne a servi, et le modèle risque de se dégrader en réapprenant ses propres effets.", "bad")


# ============================================================================ 2. Prioriser
def page_prioriser():
    st.title("Prioriser les appels")
    st.caption("Clients classés par probabilité calibrée d'être détracteur, dans le périmètre que vous avez filtré à gauche. "
               "Capacité d'appel, levier recommandé, export.")
    base = filtre
    if not len(base):
        st.stop()
    st.info(f"Périmètre courant : **{len(base)} clients** — {nom_perimetre.lower()}, après les filtres de la barre latérale. "
            "Modifiez-les à gauche pour changer la liste d'appels.")
    c1, c2 = st.columns([2, 1])
    k = c1.slider("Capacité d'appel (K)", 50, max(50, min(2000, len(base))), min(K, max(50, len(base))), step=50)
    pond = c2.checkbox("Pondérer par la valeur client (CLTV)", help="CLTV est interdite comme variable du modèle (sortie de modèle : elle fuit) mais légitime pour "
                       "arbitrer QUI appeler parmi les détracteurs prédits — couche de décision, pas de modélisation.")
    score = base["proba_detracteur"].copy()
    if pond and "CLTV" in base:
        score = score * (0.5 + 0.5 * base["CLTV"] / base["CLTV"].max())
    liste = base.assign(priorite=score).sort_values("priorite", ascending=False).head(k)
    vrais = (liste["cible_M3"] == "Detracteur").mean() if len(liste) else np.nan
    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "Clients dans le périmètre", f"{len(base)}", "après filtres")
    kpi(m2, "Appels planifiés", f"{len(liste)}", f"P(détracteur) min. {liste['proba_detracteur'].min():.0%}" if len(liste) else "")
    kpi(m3, "Vrais détracteurs dans la liste", pct(vrais) if pd.notna(vrais) else "—", "connu ici parce que le dataset est complet")
    kpi(m4, "Détracteurs du périmètre captés", pct((liste['cible_M3'] == 'Detracteur').sum() / max((base['cible_M3'] == 'Detracteur').sum(), 1)), "rappel sur le périmètre filtré")

    cols = ["Customer ID", "proba_detracteur", "prediction", "levier_recommande"] \
        + (["levier_secondaire"] if "levier_secondaire" in base.columns else []) \
        + ["Contract", "Tenure in Months", "Monthly Charge", "Internet Type", "Offer", "Number of Referrals"] \
        + (["CLTV"] if pond and "CLTV" in base else [])
    vue = liste[cols].rename(columns={"proba_detracteur": "P(détracteur)", "prediction": "Classe prédite",
                                      "levier_recommande": "Levier recommandé", "levier_secondaire": "Second levier",
                                      "Tenure in Months": "Ancienneté (mois)", "Monthly Charge": "Facture mens. ($)", "Number of Referrals": "Parrainages"})
    vue["Classe prédite"] = vue["Classe prédite"].map(LIB)
    st.dataframe(_degrade(vue.style.format({"P(détracteur)": "{:.1%}", "Facture mens. ($)": "{:.2f}"}), "P(détracteur)"),
                 hide_index=True, width="stretch", height=430)
    st.download_button("Exporter la liste d'appels (CSV)", liste[cols].to_csv(index=False), file_name=f"appels_prioritaires_top{len(liste)}.csv",
                       mime="text/csv", type="primary")

    st.markdown("### Économie de la campagne")
    mk = cfg["metier"]; eco = res["evaluation_metier"]["economie"]
    e1, e2, e3, e4 = st.columns(4)
    kpi(e1, "Coût", eur(eco["cout_campagne_eur"]), f"{K} appels × {mk['cout_appel_eur']} €")
    kpi(e2, "Gain net (modèle)", eur(eco["gain_net_eur"]), f"ROI {eco['roi']:.2f}")
    kpi(e3, "Gain net (au hasard)", eur(eco["gain_net_ciblage_aleatoire_eur"]), "même taille de campagne")
    kpi(e4, "Apporté par le modèle", eur(eco["gain_apporte_par_le_modele_eur"]), "écart des deux")
    st.caption(f"Hypothèses (`config.yaml`) : coût d'un appel {mk['cout_appel_eur']} €, valeur d'un client retenu {mk['valeur_client_retenu_eur']} €, "
               f"taux de succès {mk['taux_succes_appel']:.0%}. À valider avec l'équipe rétention.")
    with st.expander("Précision et rappel selon la capacité d'appel"):
        ck = pd.DataFrame(res["evaluation_metier"]["courbe_k"])[["k", "precision_at_k", "rappel_at_k", "lift", "detracteurs_captes"]]
        ck.columns = ["K appels", "Précision@K", "Rappel@K", "Lift", "Détracteurs captés"]
        tableau(ck, {"Précision@K": "{:.1%}", "Rappel@K": "{:.1%}", "Lift": "×{:.2f}"})
    encart("<b>Deux limites.</b> La liste classe par <i>risque</i>, comme demandé ; la littérature établit qu'il est plus rentable de cibler "
           "par <i>sensibilité à l'appel</i> (voir Synthèse). Les probabilités affichées sont surestimées en valeur absolue (calibrateur ajusté sur des "
           "répondants biaisés) : elles servent à <b>classer</b>, pas à lire une fréquence.", "warn")


# ============================================================================ 2 bis. Explorer
COLONNES_VUE = {
    "Customer ID": "Client", "prediction": "Classe prédite", "proba_detracteur": "P(détracteur)",
    "levier_recommande": "Levier recommandé", "levier_secondaire": "Second levier",
    "Contract": "Contrat", "Tenure in Months": "Ancienneté (mois)",
    "Monthly Charge": "Facture mens. ($)", "Internet Type": "Type d'internet", "Offer": "Offre",
    "Number of Referrals": "Parrainages", "tranche_age": "Tranche d'âge", "Payment Method": "Paiement",
    "cible_M3": "Classe réelle", "CLTV": "CLTV",
}


def page_explorer():
    st.title("Explorer les clients")
    st.caption("Croisez les filtres de la barre latérale : tous les chiffres de cette page se recalculent sur la sélection.")
    if not len(filtre):
        st.stop()

    # --- les chiffres de la sélection, comparés au périmètre de référence
    nps_sel, nps_ref = nps_de_serie(filtre["prediction"]), nps_de_serie(perimetre_df["prediction"])
    det_sel = float((filtre["prediction"] == "Detracteur").mean())
    det_ref = float((perimetre_df["prediction"] == "Detracteur").mean())
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Clients", f"{len(filtre)}", f"{len(filtre)/max(len(perimetre_df),1):.0%} du périmètre")
    kpi(k2, "NPS prédit de la sélection", f"{nps_sel:+.0f}", f"périmètre entier : {nps_ref:+.0f}")
    kpi(k3, "Détracteurs prédits", pct(det_sel), f"périmètre : {pct(det_ref)}")
    kpi(k4, "Facture mensuelle moyenne", f"{filtre['Monthly Charge'].mean():.0f} $",
        f"périmètre : {perimetre_df['Monthly Charge'].mean():.0f} $")
    kpi(k5, "Ancienneté médiane", f"{filtre['Tenure in Months'].median():.0f} mois",
        f"périmètre : {perimetre_df['Tenure in Months'].median():.0f} mois")

    if "cible_M3" in filtre:
        nps_vrai = nps_de_serie(filtre["cible_M3"])
        encart(f"<b>Contrôle possible ici seulement.</b> Le NPS <i>réel</i> de cette sélection est <b>{nps_vrai:+.0f}</b>, "
               f"contre <b>{nps_sel:+.0f}</b> prédit. En production, seule la colonne prédite existerait — "
               "cette ligne sert à juger la fiabilité du filtre que vous venez de composer.", "note")

    # --- répartition des classes, et croisement avec une dimension au choix
    g1, g2 = st.columns([1, 1.4])
    with g1:
        st.markdown("**Répartition des classes prédites**")
        rep_cls = (filtre["prediction"].value_counts(normalize=True).reindex(["Detracteur", "Passif", "Promoteur"])
                   .fillna(0).rename(index=LIB))
        st.bar_chart(rep_cls, color="#2c7fb8", height=260)
    with g2:
        dims = [c for c, _ in CHAMPS_CAT if c in filtre.columns and c != "prediction"]
        libs = dict(CHAMPS_CAT)
        dim = st.selectbox("Croiser la classe prédite avec", dims, format_func=lambda c: libs.get(c, c))
        ct = pd.crosstab(filtre[dim].astype(str), filtre["prediction"]).reindex(
            columns=["Detracteur", "Passif", "Promoteur"]).fillna(0).astype(int)
        ct.columns = [LIB[c] for c in ct.columns]
        ct["Total"] = ct.sum(axis=1)
        ct["% détracteurs"] = (ct["Détracteur"] / ct["Total"].replace(0, np.nan)).fillna(0)
        ct["NPS prédit"] = [nps_de_serie(filtre.loc[filtre[dim].astype(str) == i, "prediction"]) for i in ct.index]
        st.markdown(f"**Classe prédite par {libs.get(dim, dim).lower()}**")
        tableau(ct.reset_index().rename(columns={dim: libs.get(dim, dim)}),
                {"% détracteurs": "{:.0%}", "NPS prédit": "{:+.0f}"}, hauteur=260)

    # --- la liste, triable, exportable
    st.markdown("### Les clients de la sélection")
    tri = st.radio("Trier par", ["Risque décroissant", "Facture décroissante", "Ancienneté croissante", "Identifiant"],
                   horizontal=True, label_visibility="collapsed")
    cle = {"Risque décroissant": ("proba_detracteur", False), "Facture décroissante": ("Monthly Charge", False),
           "Ancienneté croissante": ("Tenure in Months", True), "Identifiant": ("Customer ID", True)}[tri]
    dispo = [c for c in COLONNES_VUE if c in filtre.columns]
    vue = filtre.sort_values(cle[0], ascending=cle[1])[dispo].rename(columns=COLONNES_VUE)
    for c in ("Classe prédite", "Classe réelle"):
        if c in vue:
            vue[c] = vue[c].map(lambda v: LIB.get(v, v))
    st.dataframe(_degrade(vue.style.format({"P(détracteur)": "{:.1%}", "Facture mens. ($)": "{:.2f}", "CLTV": "{:.0f}"}), "P(détracteur)"),
                 hide_index=True, width="stretch", height=430)
    st.download_button(f"Exporter la sélection ({len(vue)} clients, CSV)", vue.to_csv(index=False),
                       file_name=f"selection_{len(vue)}_clients.csv", mime="text/csv", type="primary")

    # --- ce qui distingue la sélection du reste du périmètre
    with st.expander("Ce qui distingue cette sélection du reste du périmètre", expanded=False):
        reste = perimetre_df.drop(index=filtre.index, errors="ignore")
        if len(reste) < 30 or len(filtre) < 30:
            st.info("Sélection ou complément trop petits pour une comparaison lisible (moins de 30 clients).")
        else:
            lignes = []
            for col, lib in [("Monthly Charge", "Facture mensuelle ($)"), ("Tenure in Months", "Ancienneté (mois)"),
                             ("Number of Referrals", "Parrainages"), ("proba_detracteur", "P(détracteur)")]:
                if col in filtre:
                    lignes.append({"Indicateur": lib, "Sélection": filtre[col].mean(),
                                   "Reste du périmètre": reste[col].mean(),
                                   "Écart": filtre[col].mean() - reste[col].mean()})
            for col, lib in [("Contract", "Part en contrat mensuel"), ("Internet Type", "Part en fibre")]:
                if col in filtre:
                    val = "Month-to-Month" if col == "Contract" else "Fiber Optic"
                    lignes.append({"Indicateur": lib, "Sélection": float((filtre[col] == val).mean()),
                                   "Reste du périmètre": float((reste[col] == val).mean()),
                                   "Écart": float((filtre[col] == val).mean() - (reste[col] == val).mean())})
            tableau(pd.DataFrame(lignes), {"Sélection": "{:.2f}", "Reste du périmètre": "{:.2f}", "Écart": "{:+.2f}"})
            st.caption("Écarts descriptifs sur la sélection que vous avez composée — des associations, pas des causes.")


# ============================================================================ 3. Analyser un client
def afficher_prediction(X_row: pd.DataFrame, cid: str | None, ligne: pd.Series | None):
    proba = pipe_cal.predict_proba(X_row)[0]
    pred = str(pipe_dec.predict(X_row)[0])
    c1, c2, c3 = st.columns([1.1, 1.3, 1.6])
    with c1:
        st.markdown(f'<span class="verdict" style="background:{COUL[pred]}">{LIB[pred]}</span>', unsafe_allow_html=True)
        st.caption("classe prédite (pipeline de décision)")
        if cid is not None:
            vrai = clients.loc[clients["Customer ID"] == cid, "cible_M3"].iloc[0]
            st.markdown(f"Classe réelle (connue ici) : **{LIB.get(vrai, vrai)}**")
        med_ = art.get("mediane_charge_par_service", float(clients["charge_par_service"].median()))
        _row = ligne if ligne is not None else X_row.iloc[0]
        try:
            _C = agreger_par_variable(contributions_detracteur(pipe_brut, X_row, res["seed"])[0], cat)
            _rangs = leviers_ordonnes(_row, med_, _C.iloc[0])
        except Exception:
            _rangs = leviers_ordonnes(_row, med_)
        encart(f"<b>Levier recommandé</b><br>{_rangs[0]}"
               + (f"<br><span style='color:#5f6b7a'><b>Second levier</b> — {_rangs[1]}</span>" if len(_rangs) > 1 else ""),
               "ok" if pred == "Detracteur" else "note")
        st.caption("Les leviers sont classés par contribution décroissante au risque **de ce client**. Le premier est souvent "
                   "« engagement » parce que la quasi-totalité des détracteurs prédits sont en contrat mensuel : c'est le second "
                   "qui distingue deux clients par ailleurs semblables.")
    with c2:
        st.markdown("**Probabilités calibrées**")
        for cl, p in zip(classes, proba):
            st.progress(float(p), text=f"{LIB[cl]} — {p:.1%}")
        st.caption("Servent à classer les clients entre eux ; surestimées en valeur absolue.")
    with c3:
        st.markdown("**Ce qui pèse pour ce client**")
        C, methode = contributions_detracteur(pipe_brut, X_row, res["seed"])
        contrib = C.iloc[0]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(7)
        st.dataframe(pd.DataFrame({"Facteur": [libelle_variable(n, cat) for n in top.index], "Effet": ["▲ risque" if v > 0 else "▼ risque" for v in top.values],
                                   "Poids": top.abs().round(3).values}), hide_index=True, width="stretch")
        st.caption(f"Méthode : {methode}. Associations, pas causes.")


def page_client():
    st.title("Analyser un client")
    mode = st.radio("Mode", ["Client existant", "Saisie manuelle"], horizontal=True, label_visibility="collapsed")
    X_row, ligne, cid, valeurs = None, None, None, {}
    if mode == "Client existant":
        # Le client se choisit par critères, pas par identifiant.
        if not len(filtre):
            st.warning("Aucun client ne correspond aux filtres. Élargissez-en un dans le panneau de gauche.")
            st.stop()
        st.caption(f"**{len(filtre)} clients** répondent aux critères choisis à gauche, sur {len(perimetre_df)} du périmètre. "
                   "Sélectionnez une ligne pour analyser ce client.")
        cand = filtre.sort_values("proba_detracteur", ascending=False)
        vue_c = pd.DataFrame({
            "Client": cand["Customer ID"].values,
            "Classe prédite": [LIB.get(v, v) for v in cand["prediction"]],
            "P(détracteur)": cand["proba_detracteur"].values,
            "Contrat": cand["Contract"].values,
            "Ancienneté (mois)": cand["Tenure in Months"].values,
            "Facture ($)": cand["Monthly Charge"].values,
            "Levier recommandé": cand["levier_recommande"].values,
        })
        evt = st.dataframe(
            _degrade(vue_c.style.format({"P(détracteur)": "{:.1%}", "Facture ($)": "{:.0f}"}), "P(détracteur)"),
            hide_index=True, width="stretch", height=260,
            on_select="rerun", selection_mode="single-row", key="choix_client")
        lignes_sel = evt.selection.rows if hasattr(evt, "selection") else []
        i_sel = lignes_sel[0] if lignes_sel else 0
        cid = str(vue_c.iloc[i_sel]["Client"])
        if not lignes_sel:
            st.caption(f"Par défaut, le client le plus à risque de la sélection est analysé : **{cid}**.")
        ligne = clients[clients["Customer ID"] == cid].iloc[0]
        X_row = clients[clients["Customer ID"] == cid][num + cat]
        st.markdown(f"### Client {cid}")
        c1, c2 = st.columns([1, 2])
        with c2:
            p1, p2, p3, p4 = st.columns(4)
            kpi(p1, "Contrat", str(ligne["Contract"]), f"ancienneté {int(ligne['Tenure in Months'])} mois")
            kpi(p2, "Facture mensuelle", f"{ligne['Monthly Charge']:.0f} $", f"{int(ligne.get('nb_services', 0))} services")
            kpi(p3, "Internet", str(ligne["Internet Type"]), f"{ligne['Avg Monthly GB Download']:.0f} Go/mois")
            kpi(p4, "Offre · parrainages", str(ligne["Offer"]), f"{int(ligne['Number of Referrals'])} parrainage(s)")
        with c1:
            kpi(c1, "Rang dans la sélection", f"{i_sel + 1} / {len(vue_c)}", "classée par risque décroissant")
        if pd.notna(ligne.get("cible_profil_detracteur_si_ambigu", np.nan)):
            encart(f"Satisfaction déclarée = 3 (zone ambiguë). L'arbitrage empirique attribue à ce client un <b>profil de détracteur à "
                   f"{ligne['cible_profil_detracteur_si_ambigu']:.0%}</b>. Indicateur métier, non utilisé à l'entraînement.")
        afficher_prediction(X_row, cid, ligne)
        seg = res["nps_simule"]["par_segment"]
        seg_c = next((s for s in seg if s["segment"] == ("Contrat mensuel" if ligne["Contract"] == "Month-to-Month" else "Contrat 1–2 ans")), None)
        if seg_c:
            st.caption(f"Contexte : dans le segment « {seg_c['segment']} », {seg_c['part_detracteurs_vraie']:.0%} des clients sont réellement détracteurs "
                       f"(NPS du segment {seg_c['nps_vrai']:+.0f}).")
        with st.expander("Toutes les données du client"):
            groupes = {"Contrat et engagement": ["Contract", "Tenure in Months", "Offer", "Paperless Billing", "Payment Method"],
                       "Services": ["Phone Service", "Multiple Lines", "Internet Service", "Internet Type", "Online Security", "Online Backup",
                                    "Device Protection Plan", "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data"],
                       "Facturation": ["Monthly Charge", "Total Charges", "Total Refunds", "Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue"],
                       "Engagement": ["Referred a Friend", "Number of Referrals", "Avg Monthly GB Download", "Avg Monthly Long Distance Charges"]}
            cols = st.columns(len(groupes))
            for col, (titre, champs) in zip(cols, groupes.items()):
                col.markdown(f"**{titre}**")
                col.dataframe(pd.DataFrame({"": [str(ligne[c]) for c in champs if c in ligne]}, index=[c for c in champs if c in ligne]), width="stretch")
        if verbatims is not None and cid in verbatims.index:
            st.markdown("**Dernière note de contact (verbatim synthétique)**")
            st.info(verbatims.loc[cid, "verbatim"])
    else:
        st.caption("Laissez un champ sur « (inconnu) » : le pipeline impute la valeur et tolère les modalités jamais vues.")
        with st.form("saisie"):
            cols = st.columns(3)
            for i, c in enumerate(cat):
                opts = ["(inconnu)"] + sorted(clients[c].dropna().astype(str).unique().tolist())
                v = cols[i % 3].selectbox(c, opts); valeurs[c] = np.nan if v == "(inconnu)" else v
            for i, c in enumerate(num):
                med = float(clients[c].median()) if clients[c].notna().any() else 0.0
                valeurs[c] = cols[i % 3].number_input(c, value=med, format="%.2f")
            ok = st.form_submit_button("Prédire", type="primary")
        if ok:
            X_row = pd.DataFrame([valeurs])[num + cat]
            ligne = X_row.iloc[0].copy()
            afficher_prediction(X_row, None, ligne)

    if modele_texte is not None and X_row is not None:
        st.markdown("### Coller un verbatim client")
        txt = st.text_area("Note de contact, extrait de chat ou avis", height=90,
                           placeholder="Ex. : Troisième appel ce mois-ci pour la même coupure, j'attends un retour écrit.")
        if txt.strip():
            p_tab = pipe_cal.predict_proba(X_row)[0]
            pt = modele_texte["pipeline_texte"]
            p_txt = pt.predict_proba([txt])[0][[list(pt.classes_).index(c) for c in classes]]
            w = modele_texte["poids_texte"]; p_fus = (1 - w) * p_tab + w * p_txt
            t = pd.DataFrame({"Classe": [LIB[c] for c in classes], "Tabulaire seul": p_tab, "Texte seul": p_txt, f"Fusion (poids texte {w:.1f})": p_fus})
            tableau(t, {c: "{:.1%}" for c in t.columns[1:]})
            encart("Le modèle texte est entraîné sur des verbatims <b>synthétiques</b>, générés à partir de la classe. Sa réaction est une "
                   "démonstration de la chaîne, pas une mesure de la valeur de vrais verbatims.", "warn")


# ============================================================================ 4. Données et cible
def page_donnees():
    st.title("Données et cible")
    q = res["qualite"]; a = res["arbitrage_des_3"]
    c = st.columns(5)
    kpi(c[0], "Clients", f"{q['lignes']}", "5 tables jointes sur Customer ID")
    kpi(c[1], "Colonnes", f"{q['colonnes']}", "après jointure")
    kpi(c[2], "Variables du modèle", f"{len(num) + len(cat)}", f"{len(num)} numériques · {len(cat)} catégorielles")
    kpi(c[3], "Colonnes exclues (fuites)", f"{sum(len(v) for k, v in res['registre_fuites'].items() if k.startswith('fuite'))}", "churn, statut, score, CLTV")
    kpi(c[4], "Note « 3 » (ambiguë)", f"{a['n_ambigus']}", f"{a['n_ambigus']/q['lignes']:.0%} de la base")
    t1, t2, t3, t4 = st.tabs(["Exploration", "Qualité", "Registre de fuites", "Construction de la cible"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            image("02_eda_satisfaction.png", "La cible : distribution de la satisfaction, et lien avec le départ")
            encart("Satisfaction 1–2 → 100 % de départs, 4–5 → 0 %. La note 3 est la seule zone d'incertitude — et elle pèse 38 % de la base.")
            image("02_eda_cramer.png", "Force d'association des catégorielles avec la satisfaction (gris : démographie, exclue du modèle)")
        with c2:
            image("02_eda_offer.png", "La quasi-expérience : taux de départ par offre reçue")
            off = pd.DataFrame(res["analyse_offer"]).rename(columns={"Offer": "Offre", "n": "Clients", "taux_depart": "Taux de départ", "satisfaction": "Satisfaction moy."})
            tableau(off, {"Taux de départ": "{:.1%}", "Satisfaction moy.": "{:.2f}"})
            encart("Les clients ayant reçu l'<b>offre E</b> partent deux fois plus. Lire « l'offre E fait fuir » serait une erreur : elle a été "
                   "proposée à des clients déjà fragiles (biais d'indication). Aucune lecture causale possible.", "warn")
        with st.expander("Numériques par niveau de satisfaction · corrélations"):
            image("02_eda_numeriques.png"); image("02_eda_correlations.png")
    with t2:
        st.markdown("**Cinq contrôles de qualité**")
        c1, c2 = st.columns(2)
        with c1:
            manq = pd.DataFrame([{"Colonne": k, "Manquants": v, "Part": f"{v/q['lignes']:.1%}", "Nature": "structurel",
                                  "Traitement": {"Offer": "catégorie « Aucune offre »", "Internet Type": "catégorie « Pas d'internet »"}.get(k, "colonne exclue (fuite)")}
                                 for k, v in q["manquants"].items()])
            st.markdown("Valeurs manquantes — toutes structurelles, aucune imputation"); tableau(manq)
            st.markdown(f"Doublons : **{q['doublons_id']}** sur l'identifiant, **{q['doublons_lignes']}** sur les lignes · colonnes constantes supprimées : {', '.join(q['constantes'])} · "
                        f"ancienneté nulle : {q['anciennete_zero']} · incohérences de facturation : {q['incoherence_facturation']} (documentées, non corrigées)")
        with c2:
            image("02_boites_aberrantes.png", "Aberrantes conservées : profils réels, modèles robustes")
            tableau(pd.DataFrame([{"Variable": k, "Hors IQR × 1,5": v} for k, v in q["aberrantes"].items()]))
    with t3:
        encart("<b>Principe</b> : connaître le churn revient à connaître la cible, alors qu'au moment de détecter un détracteur, il n'est pas "
               "encore parti. Toute colonne liée au churn est donc une fuite, de trois natures.", "bad")
        reg = res["registre_fuites"]
        nat = {"fuite_consequence": ("Conséquence", "la colonne est un effet de la cible"), "fuite_disponibilite": ("Disponibilité", "n'existe que pour les partis : sa présence trahit la réponse"),
               "fuite_modele_amont": ("Modèle amont", "sortie d'un autre modèle (IBM SPSS) ; CLTV admise en couche de décision seulement"),
               "audit_seulement": ("Audit seulement", "démographie et géographie : hors modèle, utilisées pour l'audit d'équité"), "techniques": ("Techniques", "identifiants, constantes")}
        tableau(pd.DataFrame([{"Nature": nat[k][0], "Définition": nat[k][1], "Colonnes": ", ".join(v)} for k, v in reg.items() if k in nat]))
        c1, c2 = st.columns(2)
        with c1:
            image("02_information_mutuelle.png", "Information mutuelle avec la cible — rouge : fuites exclues, bleu : features légitimes")
        with c2:
            d = res["diagnostic_fuite"]
            st.markdown("**Porte de contrôle après modélisation**")
            tableau(pd.DataFrame([{"Diagnostic": "Exactitude maximale observée (toute la grille)", "Valeur": f"{d['exactitude_max_observee']:.3f}", "Seuil": "0,85", "Verdict": "alerte" if d["alarme"] else "OK"},
                                  {"Diagnostic": "Écart arbres − linéaire (signature d'une fuite)", "Valeur": f"{d['ecart_arbres_lineaire_S1_M3']:+.3f}", "Seuil": "+0,30", "Verdict": "alerte" if d["ecart_arbres_lineaire_S1_M3"] > 0.3 else "OK"}]))
            encart("Contre-exemple publié : Mustafa et al. (2021) annoncent 98 % d'exactitude avec la note NPS parmi les features — linéaires à 41 %, arbres à 98 %. "
                   "Cet écart est la signature d'une variable qui fuit ; ici il est nul.")
    with t4:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            image("03_cible_mappings.png", "Trois mappings, et le NPS qui en résulte (bande verte : benchmark télécom)")
        with c2:
            tableau(pd.DataFrame([{"Mapping": "M1 (énoncé)", "Détracteur": "1, 2, 3", "Passif": "4", "Promoteur": "5", "NPS": f"{a['nps']['M1']:+.1f}"},
                                  {"Mapping": "M2 (recentré)", "Détracteur": "1, 2", "Passif": "3, 4", "Promoteur": "5", "NPS": f"{a['nps']['M2']:+.1f}"},
                                  {"Mapping": "M3 (arbitré)", "Détracteur": "1, 2", "Passif": "3", "Promoteur": "4, 5", "NPS": f"{a['nps']['M3']:+.1f}"}]))
            encart(f"<b>Arbitrage empirique des « 3 »</b> : un modèle entraîné sur les seuls extrêmes (1–2 vs 4–5, exactitude {a['exactitude_sur_extremes']:.0%}) "
                   f"classe les {a['n_ambigus']} clients à 3 : <b>{a['part_3_vers_detracteur']:.0%}</b> penchent détracteur, <b>{a['part_3_vers_promoteur']:.0%}</b> promoteur. "
                   f"Le bloc rejoint les {a['destination_des_3']}s.")
            encart("<b>Choix de méthode</b> : étiqueter chaque « 3 » individuellement par ce modèle rendrait la cible fonction des features (circularité). "
                   "On tranche le bloc ; la probabilité individuelle reste un indicateur métier, jamais une étiquette.", "warn")
        image("03_arbitrage_des_3.png", "Distribution de la probabilité « profil détracteur » attribuée aux clients à satisfaction 3")
        image("03_desequilibre.png", "Déséquilibre des classes dans la base, chez les répondants (entraînement) et chez les silencieux (évaluation)")
        with st.expander("Stratégies de déséquilibre comparées (baseline logistique)"):
            tableau(resume_vers_df(res["desequilibre"]["strategies"], "strategie", "Stratégie"), FMT3)


# ============================================================================ 5. Modèles et performance
def page_modele():
    st.title("Modèles et performance")
    m = mf["metriques_retenues"]; cl = pd.DataFrame(sel["classement"])
    c = st.columns(5)
    kpi(c[0], "Modèles comparés", f"{len(cl)}", "7 familles · 3 formulations")
    kpi(c[1], "Retenu", NOM_MODELE, f"{mf['version']} · rang CV {int(cl[cl.modele == mf['modele']]['rang_cv'].iloc[0])} · rang test {int(cl[cl.modele == mf['modele']]['rang_silencieux'].iloc[0])}")
    kpi(c[2], "Kappa quadratique (test)", f"{m['kappa_quadratique']:.3f}", f"CV répondants {cl[cl.modele == mf['modele']]['kappa_cv'].iloc[0]:.3f}")
    kpi(c[3], "Macro-F1 · exactitude équilibrée", f"{m['macro_f1']:.3f} · {m['exactitude_equilibree']:.3f}", f"exactitude brute {m['exactitude']:.3f} (trompeuse)")
    kpi(c[4], "AUC · AP Détracteur", f"{m.get('auc_detracteur', float('nan')):.3f} · {m.get('ap_detracteur', float('nan')):.3f}", "qualité du classement")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Catalogue justifié", "Comparaison et sélection", "Hyperparamètres", "Par classe et calibration", "Robustesse", "Fondation et texte"])
    with t1:
        st.markdown("Chaque modèle correspond à une **hypothèse** sur les données que l'on cherche à tester.")
        catd = pd.DataFrame(res["catalogue"])[["famille", "nom", "formulation", "pourquoi", "hypothese", "reference"]]
        catd.columns = ["Famille", "Modèle", "Formulation", "Pourquoi il est là", "Hypothèse testée", "Référence"]
        st.dataframe(catd, hide_index=True, width="stretch", height=560)
        encart("Écartés avec argument : CORAL / CORN (apprentissage profond ordinal, sans objet sur ~1 000 lignes). Modèles de fondation évalués à part (onglet dédié).")
    with t2:
        st.markdown("**Règle de sélection, fixée a priori** : kappa en validation croisée sur les répondants, pondéré par l'inverse de la propension estimée ; "
                    "parmi les modèles à moins d'un écart-type du meilleur, le **plus simple** l'emporte (règle du « one standard error », ESL § 7.10). "
                    "Les 85 % silencieux sont le test : rapportés pour tous, utilisés pour aucun choix.")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            image("04_cv_vs_silencieux.png", "Ce que la validation croisée promet (x) contre ce que le test donne (y). Rouge : retenu ; bleu : dans la marge de parcimonie")
        with c2:
            kpi(st, "Meilleur kappa CV brut", sel["meilleur_cv_brut"].replace("_", " "), f"marge de parcimonie ≥ {sel['seuil_parcimonie']:.3f}")
            st.write("")
            kpi(st, "Meilleur sur le test", sel["gagnant_silencieux"].replace("_", " "), "connu seulement parce que le dataset est complet")
            st.write("")
            kpi(st, "ρ de Spearman CV ↔ test", f"{sel['spearman_cv_ipw_vs_silencieux']:.2f}", f"{sel['spearman_cv_vs_silencieux']:.2f} sans pondération IPW")
            st.write("")
            kpi(st, "Écart CV → test du modèle retenu", f"{sel['ecart_cv_silencieux_du_retenu']:+.3f}", "le coût du biais de réponse, invisible en production")
        clt = cl[["rang_cv", "modele", "famille", "formulation", "version", "kappa_cv_ipw", "kappa_cv_et", "macro_f1_cv", "dans_la_marge",
                  "kappa_silencieux", "macro_f1_silencieux", "rappel_detracteur_silencieux", "rappel_passif_silencieux", "rang_silencieux"]].copy()
        clt["dans_la_marge"] = clt["dans_la_marge"].map({True: "✓", False: ""})
        clt.columns = ["Rang CV", "Modèle", "Famille", "Formulation", "Version", "Kappa CV IPW", "± plis", "Macro-F1 CV", "Marge", "Kappa test", "Macro-F1 test",
                       "Rappel Dét. test", "Rappel Passif test", "Rang test"]
        tableau(clt, {c_: "{:.3f}" for c_ in ("Kappa CV IPW", "± plis", "Macro-F1 CV", "Kappa test", "Macro-F1 test", "Rappel Dét. test", "Rappel Passif test")}, 560)
        image("04_grille_kappa.png", "Gauche : effet du mapping de la cible (S3). Droite : effet du biais de non-réponse (M3). Kappa sur les silencieux")
        with st.expander("Les trois formulations de l'énoncé — meilleur modèle de chacune"):
            fo_ = pd.DataFrame([{"Formulation": k, "Modèles": v["n_modeles"], "Meilleur (CV)": v["meilleur_modele"].replace("_", " "), "Kappa CV IPW": v["kappa_cv_ipw"],
                                 "Kappa test": v["kappa_silencieux"], "Macro-F1 test": v["macro_f1_silencieux"]} for k, v in res["formulations"].items()])
            tableau(fo_, {"Kappa CV IPW": "{:.3f}", "Kappa test": "{:.3f}", "Macro-F1 test": "{:.3f}"})
        with st.expander("Protocole 15 / 85 et repondération IPW"):
            image("04_protocole_15_85.png", "Satisfaction des répondants simulés vs base : sous S3 (MNAR), les extrêmes dominent")
            dg = pd.DataFrame(res["diagnostic_protocole"])
            tableau(dg, {"taux_reponse": "{:.1%}", "satisfaction_moyenne_base": "{:.3f}", "satisfaction_moyenne_repondants": "{:.3f}", "part_extremes_base": "{:.1%}", "part_extremes_repondants": "{:.1%}", "ecart": "{:+.3f}"})
            e = res["effet_ipw"]
            if "sans_ipw" in e:
                st.caption(f"IPW estimée (MAR) comme poids d'entraînement : kappa {e['sans_ipw']['kappa']:.3f} sans → {e['avec_ipw']['kappa']:.3f} avec · retenu : {e['retenu']}. {e['note']}.")
    with t3:
        image("04_reglage.png", "Kappa CV pondéré, par défaut vs réglé (recherche aléatoire, 5 plis). Réglé retenu si gain > 0,005 en CV — jamais sur le test")
        rows = []
        for nom, l in res["comparatif_s3_m3"].items():
            r = l.get("regle", {}); d = l["defaut"]
            rows.append({"Modèle": nom.replace("_", " "), "Configurations": str(r.get("n_configurations", "—")), "Kappa CV défaut": d["cv"]["kappa_cv_ipw"],
                         "Kappa CV réglé": r.get("cv", {}).get("kappa_cv_ipw", np.nan), "Gain": l.get("gain_reglage_cv_ipw", np.nan),
                         "Test défaut": d["silencieux"]["kappa"], "Test réglé": r.get("silencieux", {}).get("kappa", np.nan), "Version retenue": l["version_retenue"],
                         "Meilleurs paramètres": ", ".join(f"{k}={v}" for k, v in r.get("meilleurs_params", {}).items()) or "—"})
        tableau(pd.DataFrame(rows), {"Kappa CV défaut": "{:.3f}", "Kappa CV réglé": "{:.3f}", "Gain": "{:+.3f}", "Test défaut": "{:.3f}", "Test réglé": "{:.3f}"}, 560)
        st.caption("Espaces de recherche modestes à dessein : sur ~1 000 lignes, le risque de surajuster la recherche dépasse vite le gain.")
    with t4:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            pc = pd.DataFrame(m["par_classe"]).T.rename(index=LIB); pc["effectif"] = pc["effectif"].astype(int)
            pc.columns = ["Précision", "Rappel", "F1", "Effectif"]
            st.dataframe(pc.style.format({"Précision": "{:.3f}", "Rappel": "{:.3f}", "F1": "{:.3f}", "Effectif": "{:d}"}), width="stretch")
            cal = mf["calibration"]
            tableau(pd.DataFrame([{"": "Non calibré", "Brier": cal["non_calibre"]["brier"], "ECE": cal["non_calibre"]["ece"], "Rappel Passif": mf["metriques_non_calibre"]["par_classe"]["Passif"]["rappel"]},
                                  {"": "Calibré (Platt)", "Brier": cal["calibre_platt"]["brier"], "ECE": cal["calibre_platt"]["ece"], "Rappel Passif": mf["metriques_calibre"]["par_classe"]["Passif"]["rappel"]}]),
                    {"Brier": "{:.3f}", "ECE": "{:.3f}", "Rappel Passif": "{:.2f}"})
            encart(f"<b>Stratégie de décision</b> : {mf['strategie_decision']}.", "warn" if cal["passif_ecrase"] else "note")
        with c2:
            image("05_confusion_pr.png", "Matrice de confusion et courbes précision-rappel sur les 85 % silencieux")
        image("05_calibration_gain.png", "Fiabilité (sous la diagonale : le calibrateur, ajusté sur des répondants biaisés, surestime P(Détracteur)) et gain cumulé")
        encart("Le <b>classement</b> par P(Détracteur) est valide (l'ordre est préservé) ; la <b>valeur absolue</b> ne doit pas être lue comme une fréquence. "
               "En production : recalibrer sur les premières vraies réponses de la population cible.")
    with t5:
        image("05_robustesse.png", "Gauche : dégradation quand on corrompt les étiquettes d'entraînement. Droite : ce que chaque bloc de variables coûte ou rapporte")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Bruit d'étiquettes**")
            br = resume_vers_df(res["bruit_etiquettes"], "taux_bruit", "Bruit"); br["Bruit"] = br["Bruit"].map(lambda v: f"{v:.0%}")
            tableau(br, FMT3)
        with c2:
            st.markdown("**Ablation de features**")
            ab = resume_vers_df(res["ablation_features"], "configuration", "Configuration"); ab.insert(1, "Variables", [a_["n_variables"] for a_ in res["ablation_features"] if "kappa" in a_])
            tableau(ab, FMT3)
        encart("Ajouter la démographie ou la géographie est le <b>coût mesuré du choix d'équité</b> : il est assumé, et l'onglet Équité montre que le modèle "
               "retrouve de toute façon ces attributs par proxies.")
        with st.expander("Sensibilité au mapping de la cible"):
            image("05_sensibilite_mapping.png", "Les mêmes drivers sous M1, M2, M3 — contributions moyennes du modèle retenu : le sens tient, l'amplitude varie")
    with t6:
        fo = extra.get("fondation")
        st.markdown("**Modèles de fondation tabulaires**")
        if not fo:
            st.info("Non exécuté (`python -m src.fondation_eval`).")
        else:
            ref = fo["reference"]
            rows = []
            for mm in fo["modeles"]:
                if mm["statut"] == "ok":
                    rows.append({"Modèle": mm["nom"], "Kappa": mm["metriques"]["kappa"], "Macro-F1": mm["metriques"]["macro_f1"], "Rappel Dét.": mm["metriques"]["rappel_detracteur"],
                                 "Rappel Passif": mm["metriques"]["rappel_passif"], "Ajustement (s)": mm["duree_fit_s"], "Inférence (s)": mm["duree_inference_s"], "ms / client": mm["latence_par_client_ms"], "Statut": "évalué"})
                else:
                    # `cause` (diagnostic court) remplace `erreur` depuis la refonte ; on tolère
                    # les deux, et un modèle écarté du périmètre n'est pas un modèle en échec.
                    _c = mm.get("cause") or mm.get("erreur") or ""
                    rows.append({"Modèle": mm["nom"], "Statut": f"{mm['statut']} — {_c[:90]}"})
            rows.append({"Modèle": f"{NOM_MODELE} (retenu)", "Kappa": ref["kappa"], "Macro-F1": ref["macro_f1"], "Rappel Dét.": ref["rappel_detracteur"], "Inférence (s)": ref["duree_inference_s"], "Statut": "référence"})
            tableau(pd.DataFrame(rows), {"Kappa": "{:.3f}", "Macro-F1": "{:.3f}", "Rappel Dét.": "{:.3f}", "Rappel Passif": "{:.3f}", "Ajustement (s)": "{:.1f}", "Inférence (s)": "{:.2f}", "ms / client": "{:.2f}"})
            for mm in [x for x in fo["modeles"] if x["statut"] == "ok"]:
                gagne = mm["metriques"]["kappa"] > ref["kappa"] + 0.005
                encart(f"<b>{mm['nom']}</b> {'bat' if gagne else 'ne bat pas'} le modèle retenu (kappa {mm['metriques']['kappa']:.3f} vs {ref['kappa']:.3f}) pour une inférence "
                       f"{mm['duree_inference_s']/max(ref['duree_inference_s'],1e-3):.0f}× plus lente et sans explication native.", "ok" if gagne else "note")
        st.markdown("**Verbatims synthétiques et fusion texte**")
        tx = extra.get("texte")
        if not tx:
            st.info("Non exécuté (`python -m src.verbatims` puis `python -m src.texte`).")
        else:
            a_, b_, c_, d_ = st.columns(4)
            kpi(a_, "Verbatims", f"{tx['n_verbatims']}", tx["source_verbatims"].replace("_", " "))
            kpi(b_, "Kappa tabulaire", f"{tx['tabulaire_seul']['kappa']:.3f}", "référence")
            kpi(c_, "Kappa texte seul", f"{tx['texte_seul']['kappa']:.3f}", f"concordance ton/classe {tx['concordance_tonalite_classe']:.0%}")
            kpi(d_, "Kappa fusion tardive", f"{tx['fusion_tardive']['kappa']:.3f}", f"poids texte {tx['fusion_tardive']['poids_texte']:.1f} · gain {tx['gain_kappa_fusion_tardive']:+.3f}")
            encart(f"<b>À lire avant d'interpréter :</b> {tx['lecture']} Conclusion : pas de composante texte en production sur cette base.", "warn")


# ============================================================================ 6. Drivers et équité
def page_drivers():
    st.title("Drivers et équité")
    dr = res["drivers"]; eq = res["audit_equite"]; px = res["audit_proxies"]; mit = res["mitigation_seuils_age"]
    pire = max(eq.items(), key=lambda kv: kv[1]["ecart_max"])
    c = st.columns(4)
    kpi(c[0], "Méthode d'explication", dr.get("methode", "—").split(" (")[0], "identique dans l'étude et pour chaque client")
    kpi(c[1], "Driver n°1", list(dr["importance_detracteur"])[0] if "importance_detracteur" in dr else "—", "importance moyenne la plus forte")
    kpi(c[2], "Écart d'équité maximal", f"{pire[1]['ecart_max']*100:.0f} pts", f"rappel Détracteur · dimension : {pire[0]}")
    kpi(c[3], "Attribut le mieux reconstruit", max(px.items(), key=lambda kv: kv[1]["auc"])[0], f"AUC de reconstruction {max(v['auc'] for v in px.values()):.2f}")
    t1, t2, t3, t4, t5 = st.tabs(["Drivers globaux", "Par segment", "Leviers", "Équité par sous-groupe", "Proxies et mitigation"])
    with t1:
        image("05_interpretabilite.png", "Les 15 variables qui pèsent le plus (rouge : une valeur élevée augmente le risque, vert : le diminue)")
        encart("<b>Ce que le modèle dit</b> : contrat sans engagement, faible ancienneté, absence de parrainage, charge mensuelle élevée vont avec la détraction ; "
               "l'offre E ressort comme marqueur de clients déjà fragiles. <b>Associations, pas causes.</b>")
        with st.expander("Table des importances"):
            _ds = dr.get("detail_sens", {})
            imp = pd.DataFrame({"Variable": list(dr["importance_detracteur"]),
                                "Importance": list(dr["importance_detracteur"].values()),
                                "Sens de l'effet": [dr.get("sens", {}).get(v, "—") for v in dr["importance_detracteur"]],
                                "Nature": [_ds.get(v, {}).get("nature", "") for v in dr["importance_detracteur"]]})
            tableau(imp, {"Importance": "{:.4f}"})
            st.caption(dr.get("legende_sens", ""))
    with t2:
        image("05_drivers_segments.png", "Importance moyenne de chaque variable dans chaque segment")
        segs = dr.get("par_segment", {})
        if segs:
            tableau(pd.DataFrame([{"Segment": s, "Clients": v["clients"],
                                   **{f"Driver {i+1}": f"{k} ({d['importance']:.2f})"
                                      for i, (k, d) in enumerate(list(v["drivers"].items())[:5])}} for s, v in segs.items()]))
            st.caption("Chaque cellule donne la variable et son **poids** dans le segment. Le modèle étant additif, le sens "
                       "d'une variable ne change pas d'un segment à l'autre : il se lit dans la table des importances. "
                       "Ce qui varie ici est l'importance relative — un effet de composition, pas un effet propre au segment.")
        segn = pd.DataFrame(res["nps_simule"]["par_segment"]).rename(columns={"segment": "Segment", "clients": "Clients", "nps_predit": "NPS prédit", "nps_vrai": "NPS vrai",
                                                                            "part_detracteurs_predite": "Part Dét. prédite", "part_detracteurs_vraie": "Part Dét. vraie"})
        st.markdown("**NPS par segment — prédit vs vrai (silencieux)**"); tableau(segn, {"Part Dét. prédite": "{:.1%}", "Part Dét. vraie": "{:.1%}"})
    with t3:
        lev = res["leviers"]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Actionnable ou non**")
            tableau(pd.DataFrame([("Contrat mensuel", "oui", "proposer un engagement 12 mois avec avantage"), ("Faible ancienneté", "partiellement", "parcours d'accueil renforcé"),
                                  ("Charge mensuelle / par service élevée", "oui", "revue tarifaire, bundle"), ("Absence de parrainage", "indirectement", "programme de parrainage"),
                                  ("Offre E reçue", "signal, pas levier", "comprendre à qui elle a été proposée"), ("Type d'accès (fibre)", "non à court terme", "réseau, support premium")],
                                 columns=["Driver", "Actionnable ?", "Levier"]))
        with c2:
            st.markdown("**Règle de recommandation individuelle** (affichée pour chaque client)")
            st.markdown("\n".join(f"- {r}" for r in lev["regle"]))
            tableau(pd.DataFrame([{"Levier": k, "Détracteurs prédits concernés": v} for k, v in lev["distribution_detracteurs_predits"].items()]).sort_values("Détracteurs prédits concernés", ascending=False))
        encart(f"{lev['avertissement'].capitalize()}.", "warn")
    with t4:
        image("05_equite.png", "Par sous-groupe : rappel Détracteur (bleu) et taux de sélection (gris). Variables exclues du modèle, utilisées uniquement ici")
        rows = [{"Dimension": d, "Groupe": g, "Clients": r["effectif"], "Détracteurs réels": r["taux_detracteurs_reel"], "Taux de sélection": r["taux_selection"],
                 "Rappel Dét.": r["rappel_detracteur"], "Précision Dét.": r["precision_detracteur"]} for d, v in eq.items() for g, r in v["groupes"].items()]
        tableau(pd.DataFrame(rows), {"Détracteurs réels": "{:.1%}", "Taux de sélection": "{:.1%}", "Rappel Dét.": "{:.1%}", "Précision Dét.": "{:.1%}"}, 460)
        g30 = eq["tranche_age"]["groupes"]
        encart(f"<b>Écart le plus fort : {pire[0]}, {pire[1]['ecart_max']*100:.0f} points.</b> Le modèle rappelle {g30['<30']['rappel_detracteur']:.0%} des détracteurs de moins de 30 ans contre "
               f"{g30['60+']['rappel_detracteur']:.0%} des 60 ans et plus — le cas que l'énoncé décrit comme inacceptable, en sens inverse. Genre et densité de zone : écarts faibles.", "bad")
    with t5:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            image("05_proxies.png", "À quel point les features du modèle permettent de reconstruire chaque attribut exclu (AUC en validation croisée)")
        with c2:
            tableau(pd.DataFrame([{"Attribut exclu": k, "Prévalence": v["prevalence"], "AUC de reconstruction": v["auc"]} for k, v in px.items()]), {"Prévalence": "{:.1%}", "AUC de reconstruction": "{:.3f}"})
            encart("<b>Exclure une variable ne suffit pas pour que le modèle l'ignore.</b> L'âge et la situation familiale sont reconstructibles à partir du contrat, de "
                   "l'ancienneté et des services. C'est la cause structurelle de l'écart d'âge. L'exclusion reste justifiée (pas de décision explicite sur l'attribut), "
                   "mais elle ne dispense pas de l'audit.", "warn")
        st.markdown(f"**Mitigation testée** : un seuil par tranche d'âge, calé sur le rappel global ({mit['rappel_cible']:.0%}) — ce que ça coûte")
        rows = []
        for g, av in mit["avant"].items():
            ap = mit["apres"].get(g, {})
            rows.append({"Tranche d'âge": g, "Rappel avant": av["rappel_detracteur"], "Rappel après": ap.get("rappel", np.nan), "Précision avant": av["precision_detracteur"],
                         "Précision après": ap.get("precision", np.nan), "Sélection avant": av["taux_selection"], "Sélection après": ap.get("taux_selection", np.nan)})
        tableau(pd.DataFrame(rows), {c_: "{:.1%}" for c_ in ("Rappel avant", "Rappel après", "Précision avant", "Précision après", "Sélection avant", "Sélection après")})
        ga, gp = mit["global_avant"], mit["global_apres"]
        a_, b_, c_, d_ = st.columns(4)
        kpi(a_, "Kappa", f"{ga['kappa']:.3f} → {gp['kappa']:.3f}", "avant → après")
        kpi(b_, "Rappel Détracteur", f"{ga['rappel_detracteur']:.2f} → {gp['rappel_detracteur']:.2f}", "")
        kpi(c_, "Précision Détracteur", f"{ga['precision_detracteur']:.2f} → {gp['precision_detracteur']:.2f}", "")
        kpi(d_, "Appels (prédits Dét.)", f"{mit['appels_avant']} → {mit['appels_apres']}", "")
        encart("Égaliser le rappel se paie en précision chez les jeunes et en volume. Traiter différemment selon l'âge est un <b>arbitrage métier et juridique</b>, pas "
               "technique : à escalader à l'Expérience Client et au juridique avant toute production. Non appliqué par défaut.", "warn")


# ============================================================================ 7. Suivi et méthode
def page_suivi():
    st.title("Suivi et méthode")
    t1, t2 = st.tabs(["Monitoring", "Carte du modèle"])
    with t1:
        mo = extra.get("monitoring")
        if not mo:
            st.info("Monitoring non exécuté (`python -m src.monitoring`)."); return
        pd_ = mo["part_detracteurs_predits"]; su = mo.get("suivi_nouvelles_reponses")
        c = st.columns(4)
        kpi(c[0], "PSI prédictions — courant", f"{mo['psi_predictions_courant']:.3f}", "stable < 0,10 · alerte > 0,20")
        kpi(c[1], "PSI prédictions — mois simulé", f"{mo['psi_predictions_mois_simule']:.3f}", mo["scenario_simule"][:42] + "…")
        kpi(c[2], "Part détracteurs prédits", f"{pd_['courant']:.0%} → {pd_['mois_simule']:.0%}", f"référence {pd_['reference']:.0%}")
        kpi(c[3], "Réentraînement (mois simulé)", "DÉCLENCHÉ" if mo["reentrainement_declenche_mois_simule"] else "non", ", ".join(mo["variables_en_alerte_mois_simule"][:3]))
        image("06_monitoring.png", "PSI par variable et distribution de P(Détracteur) — référence, courant, mois simulé")
        if su:
            st.markdown("**Performance sur les nouvelles réponses d'enquête** (simulation : ~150 réponses / mois)")
            c1, c2 = st.columns([1.2, 1])
            with c1:
                image("06_nouvelles_reponses.png")
            with c2:
                lots = pd.DataFrame(su["lots"])[["mois", "cumul", "kappa_cumul", "rappel_detracteur_cumul", "chute_performance", "volume_atteint"]]
                lots.columns = ["Mois", "Labels cumulés", "Kappa cumulé", "Rappel Dét. cumulé", "Chute", "Volume atteint"]
                tableau(lots, {"Kappa cumulé": "{:.3f}", "Rappel Dét. cumulé": "{:.3f}"})
                st.caption(f"Seuils : chute kappa > {su['seuils']['chute_kappa']}, chute rappel > {su['seuils']['chute_rappel_detracteur']}, {su['seuils']['min_nouveaux_labels']} nouveaux labels "
                           f"(atteints au mois {su['premier_mois_volume_atteint']}), échéance {su['seuils']['echeance_max_mois']} mois.")
        tableau(pd.DataFrame([("Dérive des entrées", "PSI par variable, mensuel", "PSI > 0,20 sur une variable majeure"), ("Dérive des prédictions", "PSI de P(Détracteur), part de Détracteurs", "PSI > 0,20 ou > 5 pts"),
                              ("Performance réelle", "kappa, rappel Détracteur sur nouvelles réponses", "chute > 0,05 kappa ou > 10 pts rappel"), ("Équité", "rappel Détracteur par tranche d’âge (cumul, implémenté)", "écart > 10 pts"),
                              ("Volume de labels", "nouvelles réponses reçues", "≥ 500, au plus tard tous les 6 mois"), ("Calibration", "ECE sur nouvelles réponses", "recalibrer dès 300 labels")],
                             columns=["Signal", "Mesure", "Déclencheur"]))
        encart("<b>Boucle de rétroaction.</b> Un détracteur appelé puis retenu change de classe ; réentraîner dessus fait apprendre au modèle l'effet de sa propre intervention. "
               "Parades : journaliser chaque contact, <b>groupe de contrôle non contacté</b> (10–20 %), réentraîner sur non-contactés et contrôles.", "bad")
        with st.expander("Dérive des entrées — détail"):
            a_, b_ = st.columns(2)
            a_.markdown("Courant vs référence"); a_.dataframe(pd.DataFrame(mo["derive_entrees_courant"]).style.format({"PSI": "{:.4f}"}), hide_index=True, width="stretch", height=360)
            b_.markdown("Mois simulé vs référence"); b_.dataframe(pd.DataFrame(mo["derive_entrees_mois_simule"]).style.format({"PSI": "{:.4f}"}), hide_index=True, width="stretch", height=360)
    with t2:
        m = mf["metriques_retenues"]
        tableau(pd.DataFrame([("Nom", "Prédiction de catégorie NPS — clients silencieux"), ("Version", f"1.1 · seed {res['seed']} · `python -m src.run`"),
                              ("Modèle", f"{NOM_MODELE} ({mf['famille']}, {mf['formulation']}), {mf['version']}, Pipeline scikit-learn"),
                              ("Cible", "M3 — Détracteur (1–2) · Passif (3) · Promoteur (4–5)"), ("Entraînement", f"{mf['n_entrainement']} répondants simulés (S3, MNAR)"),
                              ("Évaluation", f"{mf['n_evaluation']} silencieux, vérité connue"), ("Kappa · macro-F1", f"{m['kappa_quadratique']:.3f} · {m['macro_f1']:.3f}"),
                              ("Rappel / précision Détracteur", f"{m['par_classe']['Detracteur']['rappel']:.2f} / {m['par_classe']['Detracteur']['precision']:.2f}"),
                              ("Décision", mf["strategie_decision"]), ("Explication", res["drivers"].get("methode", "")),
                              ("Variables exclues", "churn, CLTV, démographie, géographie"), ("Usage prévu", "prioriser une liste d'appels mensuelle ; comprendre les drivers"),
                              ("Usage interdit", "décision d'offre ou de tarif individuel ; décision sur un attribut protégé ; lire P(Détracteur) comme une fréquence"),
                              ("Limites connues", f"écart de rappel {res['audit_equite']['tranche_age']['ecart_max']*100:.0f} pts entre âges (proxies) ; Passif mal rappelé ; probabilités surestimées ; propension simulée"),
                              ("Fichiers", "models/modele_final.joblib · data/processed/clients_scores.parquet")], columns=["", "Valeur"]))


{"Synthèse": page_synthese, "Explorer les clients": page_explorer, "Prioriser les appels": page_prioriser,
 "Analyser un client": page_client, "Données et cible": page_donnees,
 "Modèles et performance": page_modele, "Drivers et équité": page_drivers, "Suivi et méthode": page_suivi}[page]()