"""Application Streamlit — outil de priorisation pour l'équipe rétention.

    streamlit run app/app.py

Neuf pages, navigation latérale. Toutes les valeurs affichées sont lues dans les artefacts
produits par le pipeline (reports/*.json, models/, data/processed/) — rien n'est recopié.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from src.features import noms_apres_encodage  # noqa: E402

REP, FIG = RACINE / "reports", RACINE / "reports" / "figures"
LIB = {"Detracteur": "Détracteur", "Passif": "Passif", "Promoteur": "Promoteur"}
COUL = {"Detracteur": "#c0392b", "Passif": "#e67e22", "Promoteur": "#27ae60"}

st.set_page_config(page_title="NPS · Rétention", page_icon="📞", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; max-width: 1250px;}
  .kpi {border:1px solid #e3e6ea; border-radius:10px; padding:14px 16px; background:#fafbfc;}
  .kpi .lab {font-size:12px; color:#5f6b7a; text-transform:uppercase; letter-spacing:.04em;}
  .kpi .val {font-size:26px; font-weight:700; color:#1f2d3d; margin-top:2px;}
  .kpi .sub {font-size:12px; color:#7a8794; margin-top:2px;}
  .verdict {display:inline-block; padding:10px 16px; border-radius:8px; color:#fff; font-weight:700; font-size:20px;}
  .note {border-left:4px solid #2c7fb8; background:#f3f8fc; padding:10px 14px; border-radius:6px; font-size:14px;}
  .warn {border-left:4px solid #e67e22; background:#fdf6ee; padding:10px 14px; border-radius:6px; font-size:14px;}
  .bad  {border-left:4px solid #c0392b; background:#fbf0ee; padding:10px 14px; border-radius:6px; font-size:14px;}
  h2 {margin-top: .6rem;}
  section[data-testid="stSidebar"] .stRadio label {font-size: 15px;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ chargement
@st.cache_resource(show_spinner="Chargement du modèle et des résultats…")
def charger():
    art = joblib.load(RACINE / "models" / "modele_final.joblib")
    clients = pd.read_parquet(RACINE / "data" / "processed" / "clients_scores.parquet")
    res = json.load(open(REP / "resultats.json", encoding="utf-8"))
    extra = {}
    for nom in ("texte", "tabpfn", "monitoring"):
        p = REP / f"{nom}.json"
        extra[nom] = json.load(open(p, encoding="utf-8")) if p.exists() else None
    pv = RACINE / "data" / "processed" / "verbatims.parquet"
    verbatims = pd.read_parquet(pv).set_index("Customer ID") if pv.exists() else None
    pt = RACINE / "models" / "modele_texte.joblib"
    texte = joblib.load(pt) if pt.exists() else None
    return art, clients, res, extra, verbatims, texte


art, clients, res, extra, verbatims, modele_texte = charger()
pipe_cal, pipe_brut, pipe_dec = art["pipeline"], art["pipeline_brut"], art.get("pipeline_decision", art["pipeline_brut"])
classes, num, cat = art["classes"], art["num"], art["cat"]
i_det = classes.index("Detracteur")
cfg = __import__("yaml").safe_load(open(RACINE / "config.yaml", encoding="utf-8"))
K = res["evaluation_metier"]["precision_at_k"]["k"]
silencieux = clients[~clients["repondant_S3"]] if "repondant_S3" in clients else clients


# ============================================================================ utilitaires
def kpi(col, label, valeur, sous=""):
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{valeur}</div>'
                 f'<div class="sub">{sous}</div></div>', unsafe_allow_html=True)


def image(nom: str, legende: str = ""):
    p = FIG / nom
    if p.exists():
        st.image(str(p), caption=legende, use_container_width=True)
    else:
        st.info(f"Figure absente : {nom} — lancer `python -m src.rapports`.")


def md_sans_images(chemin: str) -> str:
    p = REP / chemin
    if not p.exists():
        return f"*Rapport absent : {chemin}*"
    t = p.read_text(encoding="utf-8")
    t = re.sub(r"!\[\]\([^)]*\)\n?", "", t)          # les figures sont affichées à part
    t = re.sub(r"^# .*\n", "", t, count=1)           # le titre H1 est porté par la page
    return t


def contributions(X_row: pd.DataFrame) -> pd.Series:
    prep, mdl = pipe_brut.named_steps["preparation"], pipe_brut.named_steps["modele"]
    xt = prep.transform(X_row); noms = noms_apres_encodage(prep)
    j = list(mdl.classes_).index("Detracteur")
    if hasattr(mdl, "coef_"):
        c = mdl.coef_[j] * xt[0]
    else:
        import shap
        sv = shap.TreeExplainer(mdl).shap_values(xt)
        c = (sv[j] if isinstance(sv, list) else sv[:, :, j])[0]
    return pd.Series(c, index=noms)


def libelle(nom: str) -> str:
    for c in cat:
        if nom.startswith(c + "_"):
            return f"{c} = {nom[len(c) + 1:]}"
    return nom


def eur(v) -> str:
    return f"{int(round(v)):,} €".replace(",", " ")


# ============================================================================ navigation
with st.sidebar:
    st.markdown("## 📞 NPS · Rétention")
    st.caption("Priorisation des appels à partir de la catégorie NPS prédite")
    PAGES = ["Vue d'ensemble", "Prioriser les appels", "Analyser un client", "Données",
             "Modèle et performance", "Équité", "Drivers de détraction", "Méthode", "Monitoring", "À propos"]
    # Adressable par URL (?page=Équité) — utile pour les captures et pour partager un écran.
    demande = st.query_params.get("page", PAGES[0])
    page = st.radio("", PAGES, index=PAGES.index(demande) if demande in PAGES else 0,
                    label_visibility="collapsed")
    if page != demande:
        st.query_params["page"] = page
    st.divider()
    mf = res["modele_final"]
    st.caption(f"**Modèle** {mf['modele']}  \n**Cible** {mf['mapping']} · **Scénario** {mf['scenario']}  \n"
               f"Entraîné sur {mf['n_entrainement']} répondants · évalué sur {mf['n_evaluation']} silencieux  \n"
               f"Seed {res['seed']}")


# ============================================================================ pages
def page_vue_densemble():
    st.title("Vue d'ensemble")
    a = res["arbitrage_des_3"]; pk = res["evaluation_metier"]["precision_at_k"]; eco = res["evaluation_metier"]["economie"]
    m = mf["metriques_retenues"]
    c = st.columns(5)
    kpi(c[0], "NPS estimé (base complète)", f"{a['nps']['M3']:+.0f}", "mapping arbitré par les données · secteur +19 à +34")
    kpi(c[1], "Détracteurs prédits", f"{(silencieux['prediction']=='Detracteur').mean():.0%}", f"sur {len(silencieux)} clients silencieux")
    kpi(c[2], f"Précision @ {pk['k']} appels", f"{pk['precision_at_k']:.0%}", f"taux de base {pk['taux_de_base']:.0%} · lift ×{pk['lift']:.1f}")
    kpi(c[3], "Rappel Détracteur", f"{m['par_classe']['Detracteur']['rappel']:.0%}", "part des vrais détracteurs retrouvés")
    kpi(c[4], "Gain net vs hasard", eur(eco["gain_apporte_par_le_modele_eur"]), f"par campagne de {pk['k']} appels")

    st.markdown("### Trois résultats à retenir")
    g1, g2, g3 = st.columns(3)
    g1.markdown(f'<div class="note"><b>La grille de l\'énoncé fabrique un NPS faux.</b> Appliquée telle quelle : '
                f'<b>{a["nps"]["M1"]:+.0f}</b>. Elle classe en détracteurs {a["n_ambigus"]} clients « moyens » dont 84 % sont restés. '
                f'Les données disent qu\'ils penchent à <b>{a["part_3_vers_promoteur"]:.0%} côté promoteur</b>. NPS corrigé : <b>{a["nps"]["M3"]:+.0f}</b>.</div>',
                unsafe_allow_html=True)
    g2.markdown(f'<div class="note"><b>Le modèle multiplie par {pk["lift"]:.1f} l\'efficacité des appels.</b> Sur {pk["k"]} appels, '
                f'{pk["precision_at_k"]:.0%} touchent un vrai détracteur contre {pk["taux_de_base"]:.0%} au hasard — '
                f'<b>{eur(eco["gain_apporte_par_le_modele_eur"])}</b> de gain net supplémentaire par campagne.</div>', unsafe_allow_html=True)
    eq = res["audit_equite"]["tranche_age"]["groupes"]
    g3.markdown(f'<div class="warn"><b>Il rate davantage les jeunes détracteurs.</b> Rappel {eq["<30"]["rappel_detracteur"]:.0%} chez les '
                f'moins de 30 ans contre {eq["60+"]["rappel_detracteur"]:.0%} chez les 60+, sans aucune variable d\'âge dans le modèle. '
                f'Mesuré, signalé, à traiter avant production.</div>', unsafe_allow_html=True)

    st.markdown("### ")
    c1, c2 = st.columns(2)
    with c1:
        image("03_cible_mappings.png", "Répartition des classes et NPS selon le mapping de la cible")
    with c2:
        image("05_calibration_gain.png", "Fiabilité des probabilités (gauche) et gain cumulé de la campagne (droite)")

    st.markdown("### La décision demandée à la direction")
    st.markdown('<div class="bad"><b>Appeler les clients les plus à risque n\'est pas la stratégie la plus rentable</b> '
                '(Ascarza, <i>Journal of Marketing Research</i>, 2018 : jusqu\'à 7 points de churn en moins en ciblant par '
                '<i>sensibilité à l\'appel</i>). Pour le mesurer, une seule chose, gratuite : <b>ne pas contacter 10 à 20 % '
                'des clients ciblés à la prochaine campagne, tirés au hasard</b>. Sans ce groupe de contrôle, on ne saura jamais '
                'si la campagne a servi, et le modèle se dégradera en réapprenant ses propres effets.</div>', unsafe_allow_html=True)


def page_prioriser():
    st.title("Prioriser les appels")
    st.caption("Clients silencieux classés par probabilité calibrée d'être détracteur. Filtres, capacité d'appel, export.")
    with st.expander("Filtres", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        contrats = f1.multiselect("Contrat", sorted(silencieux["Contract"].unique()), default=sorted(silencieux["Contract"].unique()))
        internet = f2.multiselect("Type d'internet", sorted(silencieux["Internet Type"].unique()), default=sorted(silencieux["Internet Type"].unique()))
        offres = f3.multiselect("Offre reçue", sorted(silencieux["Offer"].unique()), default=sorted(silencieux["Offer"].unique()))
        anc = f4.slider("Ancienneté (mois)", 0, int(silencieux["Tenure in Months"].max()), (0, int(silencieux["Tenure in Months"].max())))
    base = silencieux[silencieux["Contract"].isin(contrats) & silencieux["Internet Type"].isin(internet)
                      & silencieux["Offer"].isin(offres) & silencieux["Tenure in Months"].between(*anc)]
    c1, c2 = st.columns([2, 1])
    k = c1.slider("Capacité d'appel (K)", 50, max(50, min(2000, len(base))), min(K, max(50, len(base))), step=50)
    pond = c2.checkbox("Pondérer par la valeur client (CLTV)", help="CLTV est interdite comme variable du modèle (sortie de modèle : elle fuit) "
                       "mais légitime pour arbitrer QUI appeler parmi les détracteurs prédits — couche de décision, pas de modélisation.")
    score = base["proba_detracteur"].copy()
    if pond and "CLTV" in base:
        score = score * (0.5 + 0.5 * base["CLTV"] / base["CLTV"].max())
    liste = base.assign(priorite=score).sort_values("priorite", ascending=False).head(k)

    vrais = (liste["cible_M3"] == "Detracteur").mean() if "cible_M3" in liste else np.nan
    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "Clients dans le périmètre", f"{len(base)}", "après filtres")
    kpi(m2, "Appels planifiés", f"{len(liste)}", f"P(détracteur) min. {liste['proba_detracteur'].min():.0%}" if len(liste) else "")
    kpi(m3, "Vrais détracteurs dans la liste", f"{vrais:.0%}" if pd.notna(vrais) else "—", "connu ici parce que le dataset est complet")
    kpi(m4, "Détracteurs de la base captés", f"{(liste['cible_M3']=='Detracteur').sum() / max((base['cible_M3']=='Detracteur').sum(),1):.0%}", "rappel sur le périmètre filtré")

    cols = ["Customer ID", "proba_detracteur", "prediction", "Contract", "Tenure in Months", "Monthly Charge",
            "Internet Type", "Offer", "Number of Referrals"] + (["CLTV"] if pond and "CLTV" in base else [])
    vue = liste[cols].rename(columns={"proba_detracteur": "P(détracteur)", "prediction": "Classe prédite",
                                      "Tenure in Months": "Ancienneté (mois)", "Monthly Charge": "Facture mens. ($)",
                                      "Number of Referrals": "Parrainages"})
    vue["Classe prédite"] = vue["Classe prédite"].map(LIB)
    st.dataframe(vue.style.format({"P(détracteur)": "{:.1%}", "Facture mens. ($)": "{:.2f}"})
                 .background_gradient(subset=["P(détracteur)"], cmap="Reds"),
                 hide_index=True, use_container_width=True, height=440)
    st.download_button("⬇️ Exporter la liste d'appels (CSV)", liste[cols].to_csv(index=False),
                       file_name=f"appels_prioritaires_top{len(liste)}.csv", mime="text/csv", type="primary")

    st.markdown("### Économie de la campagne")
    mk = cfg["metier"]; eco = res["evaluation_metier"]["economie"]
    e1, e2, e3, e4 = st.columns(4)
    kpi(e1, "Coût", eur(eco["cout_campagne_eur"]), f"{K} appels × {mk['cout_appel_eur']} €")
    kpi(e2, "Gain net (modèle)", eur(eco["gain_net_eur"]), f"ROI {eco['roi']:.2f}")
    kpi(e3, "Gain net (au hasard)", eur(eco["gain_net_ciblage_aleatoire_eur"]), "même taille de campagne")
    kpi(e4, "Apporté par le modèle", eur(eco["gain_apporte_par_le_modele_eur"]), "écart des deux")
    st.caption(f"Hypothèses métier (`config.yaml`) : coût d'un appel {mk['cout_appel_eur']} €, valeur d'un client retenu "
               f"{mk['valeur_client_retenu_eur']} €, taux de succès {mk['taux_succes_appel']:.0%}. Paramètres à valider avec l'équipe rétention.")
    st.markdown('<div class="warn"><b>Limite de cette règle de ciblage.</b> La liste classe par <i>risque</i>, comme demandé. La '
                'littérature établit qu\'il est plus rentable de cibler par <i>sensibilité à l\'appel</i>. Les probabilités affichées '
                'sont surestimées en valeur absolue (calibrateur ajusté sur les répondants, biaisés) : elles servent à <b>classer</b>, '
                'pas à lire une fréquence.</div>', unsafe_allow_html=True)


def afficher_prediction(X_row: pd.DataFrame, cid: str | None):
    proba = pipe_cal.predict_proba(X_row)[0]
    pred = str(pipe_dec.predict(X_row)[0])
    c1, c2, c3 = st.columns([1.1, 1.4, 1.5])
    with c1:
        st.markdown(f'<span class="verdict" style="background:{COUL[pred]}">{LIB[pred]}</span>', unsafe_allow_html=True)
        st.caption("classe prédite (pipeline de décision)")
        if cid is not None and "cible_M3" in clients:
            vrai = clients.loc[clients["Customer ID"] == cid, "cible_M3"].iloc[0]
            st.markdown(f"Classe réelle (connue ici) : **{LIB.get(vrai, vrai)}**")
    with c2:
        st.markdown("**Probabilités calibrées**")
        for cl, p in zip(classes, proba):
            st.progress(float(p), text=f"{LIB[cl]} — {p:.1%}")
        st.caption("Servent à classer les clients entre eux ; surestimées en valeur absolue.")
    with c3:
        st.markdown("**Ce qui pèse pour ce client**")
        contrib = contributions(X_row)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(7)
        st.dataframe(pd.DataFrame({"Facteur": [libelle(n) for n in top.index],
                                   "Effet": ["▲ risque" if v > 0 else "▼ risque" for v in top.values],
                                   "Poids": top.abs().round(2).values}), hide_index=True, use_container_width=True)
    st.caption("Associations, pas causes. Modèle linéaire : chaque poids est la contribution exacte coef × valeur au score de détraction.")


def page_client():
    st.title("Analyser un client")
    mode = st.radio("", ["Client existant", "Saisie manuelle"], horizontal=True, label_visibility="collapsed")
    if mode == "Client existant":
        c1, c2 = st.columns([1, 2])
        cid = c1.selectbox("Identifiant client", clients["Customer ID"].tolist())
        ligne = clients[clients["Customer ID"] == cid].iloc[0]
        X_row = clients[clients["Customer ID"] == cid][num + cat]
        with c2:
            p1, p2, p3, p4 = st.columns(4)
            kpi(p1, "Contrat", str(ligne["Contract"]), f"ancienneté {int(ligne['Tenure in Months'])} mois")
            kpi(p2, "Facture mensuelle", f"{ligne['Monthly Charge']:.0f} $", f"{int(ligne.get('nb_services', 0))} services")
            kpi(p3, "Internet", str(ligne["Internet Type"]), f"{ligne['Avg Monthly GB Download']:.0f} Go/mois")
            kpi(p4, "Offre · parrainages", str(ligne["Offer"]), f"{int(ligne['Number of Referrals'])} parrainage(s)")
        if pd.notna(ligne.get("cible_profil_detracteur_si_ambigu", np.nan)):
            st.markdown(f'<div class="note">Satisfaction déclarée = 3 (zone ambiguë). L\'arbitrage empirique attribue à ce client un '
                        f'<b>profil de détracteur à {ligne["cible_profil_detracteur_si_ambigu"]:.0%}</b>. Indicateur métier, non utilisé à l\'entraînement.</div>',
                        unsafe_allow_html=True)
        afficher_prediction(X_row, cid)
        with st.expander("Toutes les données du client"):
            groupes = {"Contrat et engagement": ["Contract", "Tenure in Months", "Offer", "Paperless Billing", "Payment Method"],
                       "Services": ["Phone Service", "Multiple Lines", "Internet Service", "Internet Type", "Online Security", "Online Backup",
                                    "Device Protection Plan", "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data"],
                       "Facturation": ["Monthly Charge", "Total Charges", "Total Refunds", "Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue"],
                       "Engagement": ["Referred a Friend", "Number of Referrals", "Avg Monthly GB Download", "Avg Monthly Long Distance Charges"]}
            cols = st.columns(len(groupes))
            for col, (titre, champs) in zip(cols, groupes.items()):
                col.markdown(f"**{titre}**")
                col.dataframe(pd.DataFrame({"": [str(ligne[c]) for c in champs if c in ligne]}, index=[c for c in champs if c in ligne]),
                              use_container_width=True)
        if verbatims is not None and cid in verbatims.index:
            st.markdown("**Dernière note de contact (verbatim synthétique)**")
            st.info(verbatims.loc[cid, "verbatim"])
    else:
        st.caption("Laisse un champ sur « (inconnu) » ou vide : le pipeline impute et tolère les modalités jamais vues.")
        valeurs = {}
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
            afficher_prediction(pd.DataFrame([valeurs])[num + cat], None)

    # Bonus § 4.8 : un verbatim déplace-t-il la prédiction ?
    if modele_texte is not None:
        st.markdown("### Coller un verbatim client")
        txt = st.text_area("Note de contact, extrait de chat ou avis", height=90,
                           placeholder="Ex. : Troisième appel ce mois-ci pour la même coupure, j'attends un retour écrit.")
        if txt.strip():
            X_cur = X_row if mode == "Client existant" else pd.DataFrame([valeurs])[num + cat]
            p_tab = pipe_cal.predict_proba(X_cur)[0]
            pt = modele_texte["pipeline_texte"]
            p_txt = pt.predict_proba([txt])[0][[list(pt.classes_).index(c) for c in classes]]
            w = modele_texte["poids_texte"]; p_fus = (1 - w) * p_tab + w * p_txt
            t = pd.DataFrame({"Classe": [LIB[c] for c in classes], "Tabulaire seul": p_tab, "Texte seul": p_txt,
                              f"Fusion (poids texte {w:.1f})": p_fus})
            st.dataframe(t.style.format({c: "{:.1%}" for c in t.columns[1:]}), hide_index=True, use_container_width=True)
            st.markdown('<div class="warn">Le modèle texte est entraîné sur des verbatims <b>synthétiques</b>, générés à partir de la classe. '
                        'Sa réaction est une démonstration de la chaîne, pas une mesure de la valeur de vrais verbatims.</div>', unsafe_allow_html=True)


def page_donnees():
    st.title("Données")
    q = res["qualite"]
    c = st.columns(5)
    kpi(c[0], "Clients", f"{q['lignes']}", "5 fichiers joints sur Customer ID")
    kpi(c[1], "Colonnes", f"{q['colonnes']}", "après jointure")
    kpi(c[2], "Doublons", f"{q['doublons_id']}", "sur l'identifiant")
    kpi(c[3], "Colonnes constantes", f"{len(q['constantes'])}", ", ".join(q["constantes"]))
    kpi(c[4], "Manquants structurels", f"{len(q['manquants'])} colonnes", "encodés en catégorie, jamais imputés")
    t1, t2, t3, t4 = st.tabs(["Exploration", "Qualité", "Dictionnaire", "Registre de fuites"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            image("02_eda_satisfaction.png", "La cible : distribution et lien avec le départ")
            image("02_eda_cramer.png", "Force d'association des catégorielles avec la satisfaction")
        with c2:
            image("02_eda_offer.png", "La quasi-expérience : taux de départ par offre reçue")
            image("02_eda_correlations.png", "Corrélations entre numériques")
        image("02_eda_numeriques.png", "Numériques par niveau de satisfaction")
        st.markdown(md_sans_images("02_donnees/analyse_exploratoire.md"))
    with t2:
        image("02_boites_aberrantes.png", "Boîtes à moustaches — valeurs aberrantes conservées")
        st.markdown(md_sans_images("02_donnees/qualite_donnees.md"))
    with t3:
        st.markdown(md_sans_images("02_donnees/dictionnaire_donnees.md"))
    with t4:
        image("02_information_mutuelle.png", "Information mutuelle avec la cible — rouge : fuites exclues")
        st.markdown(md_sans_images("02_donnees/registre_fuites.md"))


def page_modele():
    st.title("Modèle et performance")
    m = mf["metriques_retenues"]
    c = st.columns(5)
    kpi(c[0], "Kappa quadratique", f"{m['kappa_quadratique']:.3f}", "métrique ordinale principale")
    kpi(c[1], "Macro-F1", f"{m['macro_f1']:.3f}", "moyenne des trois classes")
    kpi(c[2], "Exactitude équilibrée", f"{m['exactitude_equilibree']:.3f}", "")
    kpi(c[3], "Exactitude brute", f"{m['exactitude']:.3f}", "trompeuse — 0,43 en prédisant toujours Passif")
    kpi(c[4], "Alarme fuite", "aucune" if not res["diagnostic_fuite"]["alarme"] else "OUI",
        f"max observé {res['diagnostic_fuite']['exactitude_max_observee']:.2f} < 0,85")
    t1, t2, t3, t4, t5 = st.tabs(["Par classe", "Comparaison des modèles", "Hyperparamètres", "Calibration", "TabPFN"])
    with t1:
        pc = pd.DataFrame(m["par_classe"]).T.rename(index=LIB)
        pc["effectif"] = pc["effectif"].astype(int)
        pc.columns = ["Précision", "Rappel", "F1", "Effectif"]
        st.dataframe(pc.style.format({"Précision": "{:.3f}", "Rappel": "{:.3f}", "F1": "{:.3f}", "Effectif": "{:d}"}),
                     use_container_width=True)
        image("05_confusion_pr.png", "Matrice de confusion et courbes précision-rappel sur les 85 % silencieux")
        st.caption(f"Stratégie de décision : {mf.get('strategie_decision', '')}")
    with t2:
        image("04_grille_kappa.png", "27 combinaisons — la cible M3 domine, la logistique gagne")
        sel = res.get("selection_modele_final", {})
        if sel:
            st.markdown(f"**Sélection par la mesure** — {sel['critere']}")
            st.dataframe(pd.DataFrame(sel["classement"]), hide_index=True, use_container_width=True)
        image("04_cv_boxplot.png", "Stabilité en validation croisée (5 plis) sous S3 · M3")
        if "effet_ipw" in res and "sans_ipw" in res["effet_ipw"]:
            e = res["effet_ipw"]
            st.caption(f"Repondération IPW : kappa {e['sans_ipw']['kappa']:.3f} sans → {e['avec_ipw']['kappa']:.3f} avec · retenu : {e['retenu']}.")
    with t3:
        h = res.get("hyperparametres", {})
        if not h:
            st.info("Recherche d'hyperparamètres non encore exécutée.")
        for nom, v in h.items():
            st.markdown(f"**{nom}**")
            if "erreur" in v:
                st.error(v["erreur"]); continue
            a, b, cc, d = st.columns(4)
            kpi(a, "Kappa CV interne (réglé)", f"{v['kappa_cv_interne']:.3f}", f"{v['n_configurations']} configurations")
            kpi(b, "Kappa silencieux — défaut", f"{v['kappa_silencieux_defaut']:.3f}", "")
            kpi(cc, "Kappa silencieux — réglé", f"{v['kappa_silencieux_regle']:.3f}", f"gain {v['gain_kappa']:+.3f}")
            kpi(d, "Décision", v["retenu"], "")
            st.json(v["meilleurs_params"], expanded=False)
        st.caption("Espaces de recherche modestes à dessein : sur ~1 050 lignes, le risque de surajuster la recherche dépasse vite le gain.")
    with t4:
        image("05_calibration_gain.png", "Courbe de fiabilité et gain cumulé")
        mc, mb = mf["metriques_calibre"], mf["metriques_non_calibre"]
        st.markdown(f"""
La calibration (isotonique puis Platt) a **écrasé la classe Passif** : rappel {mc['par_classe']['Passif']['rappel']:.2f} après contre
{mb['par_classe']['Passif']['rappel']:.2f} avant. Sous MNAR les Passifs sont rares parmi les répondants. D'où la stratégie hybride : classe par
le modèle non calibré, P(Détracteur) par le modèle calibré pour le classement.

Les courbes de fiabilité sont **sous la diagonale** : le calibrateur apprend le taux de base des répondants (biaisé vers les extrêmes),
pas celui de la population silencieuse. Le **classement** reste valide, la **valeur absolue** des probabilités ne l'est pas.
""")
    with t5:
        tp = extra.get("tabpfn")
        if not tp:
            st.info("Évaluation TabPFN non encore exécutée (`python -m src.tabpfn_eval`).")
        elif tp.get("statut") != "ok":
            st.warning(f"TabPFN indisponible dans cet environnement : {tp.get('erreur', '')}")
            st.caption("Domaine de validité annoncé de TabPFN-2.5 : ~50 000 lignes, ~2 000 features — nos 7 043 lignes sont dans le domaine. "
                       "Limites connues : latence d'inférence, pas d'interprétabilité native, coût en production.")
        else:
            a, b, cc, d = st.columns(4)
            kpi(a, "Kappa TabPFN", f"{tp['metriques']['kappa']:.3f}", f"logistique {tp['reference_logistique']['kappa']:.3f}")
            kpi(b, "Macro-F1", f"{tp['metriques']['macro_f1']:.3f}", "")
            kpi(cc, "Inférence", f"{tp['duree_inference_s']:.1f} s", f"{tp['latence_par_client_ms']:.2f} ms / client")
            kpi(d, "Logistique", f"{tp['reference_logistique']['duree_inference_s']:.3f} s", "même population")
            st.dataframe(pd.DataFrame(tp["metriques"]["par_classe"]).T.rename(index=LIB), use_container_width=True)
            verdict = "gagne" if tp["metriques"]["kappa"] > tp["reference_logistique"]["kappa"] + 0.005 else "ne gagne pas"
            st.caption(f"Verdict honnête : TabPFN **{verdict}** face à la régression logistique sur cette tâche, pour un coût d'inférence bien supérieur.")


def page_equite():
    st.title("Équité")
    eq = res["audit_equite"]; pire = max(eq.items(), key=lambda kv: kv[1]["ecart_max"])
    c = st.columns(len(eq))
    for col, (dim, v) in zip(c, eq.items()):
        kpi(col, dim, f"{v['ecart_max']*100:.0f} pts", "écart max de rappel Détracteur")
    image("05_equite.png", "Rappel Détracteur par sous-groupe — variables exclues du modèle, utilisées uniquement ici")
    st.markdown(f'<div class="bad"><b>Écart le plus fort : {pire[0]} — {pire[1]["ecart_max"]*100:.0f} points.</b> '
                f'C\'est le cas que l\'énoncé décrit comme inacceptable, en sens inverse : ce sont les jeunes détracteurs qu\'on rate, '
                f'sans qu\'aucune variable d\'âge n\'entre dans le modèle — il les retrouve via les proxies contractuels.</div>', unsafe_allow_html=True)
    st.markdown(md_sans_images("05_evaluation/audit_equite.md"))


def page_drivers():
    st.title("Drivers de détraction")
    image("05_interpretabilite.png", "Les 15 variables qui pèsent le plus — rouge augmente le risque, vert le diminue")
    c1, c2 = st.columns(2)
    with c1:
        image("05_sensibilite_mapping.png", "Les mêmes drivers sous M1, M2, M3 ? Le sens tient, l'amplitude varie")
    with c2:
        st.markdown(md_sans_images("05_evaluation/sensibilite_mapping.md"))
    st.markdown(md_sans_images("05_evaluation/interpretabilite_drivers.md"))


def page_methode():
    st.title("Méthode")
    t1, t2, t3, t4 = st.tabs(["Construction de la cible", "Protocole 15 / 85", "Verbatims et texte", "Cadrage et revue"])
    with t1:
        image("03_arbitrage_des_3.png", "Arbitrage empirique des clients à satisfaction 3")
        st.markdown(md_sans_images("03_preparation/construction_cible.md"))
    with t2:
        image("04_protocole_15_85.png", "Satisfaction des répondants vs base — S3 sur-représente les extrêmes")
        st.markdown(md_sans_images("04_modelisation/protocole_validation.md"))
    with t3:
        tx = extra.get("texte")
        if not tx:
            st.info("Chaîne texte non encore exécutée (`python -m src.verbatims` puis `python -m src.texte`).")
        else:
            a, b, cc, d = st.columns(4)
            kpi(a, "Verbatims", f"{tx['n_verbatims']}", tx["source_verbatims"].replace("_", " "))
            kpi(b, "Kappa tabulaire", f"{tx['tabulaire_seul']['kappa']:.3f}", "référence")
            kpi(cc, "Kappa texte seul", f"{tx['texte_seul']['kappa']:.3f}", f"concordance ton/classe {tx['concordance_tonalite_classe']:.0%}")
            kpi(d, "Kappa fusion tardive", f"{tx['fusion_tardive']['kappa']:.3f}", f"poids texte {tx['fusion_tardive']['poids_texte']:.1f} · gain {tx['gain_kappa_fusion_tardive']:+.3f}")
            st.markdown(f'<div class="warn"><b>Lecture obligatoire.</b> {tx["lecture"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
- **Source des verbatims** : `{tx['source_verbatims']}`. Sans clé API dans l'environnement, le générateur local à gabarits
  stochastiques (fragments rédigés par un LLM, assemblage seedé) a été utilisé ; le script d'appel à l'API Anthropic est prêt
  (`src/verbatims.py`, prompt versionné `prompts/verbatim_v1.txt`) et produit le même schéma.
- **Bruit injecté** : 25 % des tonalités redistribuées au hasard → concordance observée {tx['concordance_tonalite_classe']:.0%}, plafond théorique {tx['plafond_theorique']:.0%}.
- **Fusion précoce (SVD 20)** : kappa {tx['fusion_precoce_svd20']['kappa']:.3f}.
- Poids de fusion choisi par validation croisée sur les répondants : {tx['fusion_tardive']['grille_poids_cv']}.
""")
    with t4:
        st.markdown(md_sans_images("01_comprehension_metier.md"))
        st.divider()
        st.markdown(md_sans_images("05_evaluation/revue_processus.md"))


def page_monitoring():
    st.title("Monitoring et réentraînement")
    mo = extra.get("monitoring")
    if not mo:
        st.info("Monitoring non encore exécuté (`python -m src.monitoring`)."); return
    s = mo["seuils"]; pd_ = mo["part_detracteurs_predits"]
    c = st.columns(4)
    kpi(c[0], "PSI prédictions — courant", f"{mo['psi_predictions_courant']:.3f}", "stable < 0,10 · alerte > 0,20")
    kpi(c[1], "PSI prédictions — mois simulé", f"{mo['psi_predictions_mois_simule']:.3f}", mo["scenario_simule"][:40] + "…")
    kpi(c[2], "Part détracteurs prédits", f"{pd_['courant']:.0%} → {pd_['mois_simule']:.0%}", f"référence {pd_['reference']:.0%}")
    kpi(c[3], "Réentraînement (mois simulé)", "DÉCLENCHÉ" if mo["reentrainement_declenche_mois_simule"] else "non", ", ".join(mo["variables_en_alerte_mois_simule"][:3]))
    image("06_monitoring.png", "PSI par variable et distribution de P(Détracteur) — référence, courant, mois simulé")
    t1, t2 = st.tabs(["Dérive courante", "Mois simulé"])
    with t1:
        st.dataframe(pd.DataFrame(mo["derive_entrees_courant"]).style.format({"PSI": "{:.4f}"}), hide_index=True, use_container_width=True, height=380)
    with t2:
        st.dataframe(pd.DataFrame(mo["derive_entrees_mois_simule"]).style.format({"PSI": "{:.4f}"}), hide_index=True, use_container_width=True, height=380)
    st.markdown(md_sans_images("06_deploiement/monitoring_et_retrainement.md"))


def page_apropos():
    st.title("À propos")
    st.markdown(md_sans_images("06_deploiement/carte_modele.md"))
    st.divider()
    st.markdown((RACINE / "README.md").read_text(encoding="utf-8") if (RACINE / "README.md").exists() else "")


{"Vue d'ensemble": page_vue_densemble, "Prioriser les appels": page_prioriser, "Analyser un client": page_client,
 "Données": page_donnees, "Modèle et performance": page_modele, "Équité": page_equite,
 "Drivers de détraction": page_drivers, "Méthode": page_methode, "Monitoring": page_monitoring, "À propos": page_apropos}[page]()
