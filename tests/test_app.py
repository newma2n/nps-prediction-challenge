"""Tests de l'application Streamlit — chaque page doit s'exécuter sans exception.

Pourquoi ce fichier existe : une requête HTTP sur l'application renvoie 200 même quand une page
plante, parce que Streamlit rend le contenu après coup, par websocket. Un contrôle « le serveur
répond » ne prouve donc rien. `AppTest` exécute réellement le script de chaque page et expose les
exceptions — c'est ainsi qu'a été trouvé le défaut des curseurs de filtre, qui cassait les huit
pages d'un coup puisque la barre latérale leur est commune.

Ces tests sont ignorés si les artefacts du pipeline ne sont pas présents.
"""
from __future__ import annotations

from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
APP = RACINE / "app" / "app.py"
ARTEFACTS = [RACINE / "models" / "modele_final.joblib",
             RACINE / "data" / "processed" / "clients_scores.parquet",
             RACINE / "reports" / "resultats.json"]

PAGES = ["Synthèse", "Explorer les clients", "Prioriser les appels", "Analyser un client",
         "Données et cible", "Modèles et performance", "Drivers et équité", "Suivi et méthode"]

besoin_artefacts = pytest.mark.skipif(not all(p.exists() for p in ARTEFACTS),
                                      reason="pipeline non exécuté")


def _lancer(page: str | None = None, **etat):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=300)
    if page:
        at.query_params["page"] = page
    for k, v in etat.items():
        at.session_state[k] = v
    at.run()
    return at


@besoin_artefacts
@pytest.mark.parametrize("page", PAGES)
def test_chaque_page_s_execute_sans_exception(page):
    at = _lancer(page)
    assert not at.exception, f"{page} : {[str(e.value)[:300] for e in at.exception]}"
    assert len(at.markdown) > 5, f"{page} n'a produit presque aucun contenu"


@besoin_artefacts
def test_les_filtres_croises_reduisent_la_selection_sans_casser_les_pages():
    """Le cœur de la promesse métier : croiser deux filtres restreint la population, et la page
    reste lisible — y compris quand le croisement ne laisse aucun client."""
    large = _lancer("Explorer les clients")
    assert not large.exception

    etroit = _lancer("Explorer les clients", f_Contract=["Month-to-Month"],
                     **{"f_Internet Type": ["Fiber Optic"]})
    assert not etroit.exception

    # Un filtre impossible ne doit pas faire planter la page : elle doit le dire.
    vide = _lancer("Explorer les clients", f_recherche="IDENTIFIANT-QUI-N-EXISTE-PAS")
    assert not vide.exception
    assert any("Aucun client" in w.value for w in vide.warning), "aucun avertissement sur sélection vide"


@besoin_artefacts
def test_la_reinitialisation_des_filtres_restaure_un_etat_valide():
    at = _lancer("Explorer les clients", f_Contract=["Month-to-Month"])
    assert not at.exception
    boutons = [b for b in at.button if "éinitialiser" in b.label]
    assert boutons, "bouton de réinitialisation absent"
    apres = boutons[0].click().run()
    assert not apres.exception
    assert apres.session_state["f_Contract"] == []
