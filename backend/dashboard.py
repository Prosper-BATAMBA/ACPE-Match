"""
dashboard.py — ACPE Match Dashboard (Streamlit)

Lancement:
    cd backend
    streamlit run dashboard.py --server.port 8501
"""

import os
import time

import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ACPE Match — IndabaX Congo 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1a1a2e; font-weight: 700; }
    h2, h3 { color: #2d2d44; }
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label { color: #6c757d; font-size: 0.85rem; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("ACPE Match")
st.sidebar.caption("IndabaX Congo 2026")
page = st.sidebar.radio(
    "Navigation",
    ["Vue d'ensemble", "Matching", "Recommandations", "Offre > Candidats", "Recherche NL", "Rapport"],
)
st.sidebar.divider()
st.sidebar.caption("API: " + API_URL)


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}min {s:02d}s"


def _api_get(path, params=None, timeout=10):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


def _render_profile_table(info: dict, fields: list[tuple[str, str]]):
    rows = []
    for label, key in fields:
        val = info.get(key)
        if val is None:
            val = "-"
        rows.append({"Champ": label, "Valeur": str(val)})
    df = pd.DataFrame(rows)
    st.dataframe(df.set_index("Champ"), use_container_width=True, hide_index=True)


def _render_recommendations_table(recs: list, show_gap_details: bool = True):
    if not recs:
        st.info("Aucune recommandation.")
        return

    rows = []
    for i, r in enumerate(recs, 1):
        gap = r.get("skill_gap", {})
        rows.append({
            "Rang": i,
            "Intitule": r.get("intitule") or r.get("candidate_name") or r.get("offer_id", "N/A"),
            "Entreprise": r.get("entreprise") or r.get("metier_vise") or "-",
            "Lieu": r.get("lieu") or "-",
            "Score": round(r["score"], 3),
            "Couverture": f"{gap.get('gap_score', 0):.0%}",
            "Acquis": len(gap.get("acquired", [])),
            "Manquants": len(gap.get("missing", [])),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if show_gap_details and recs:
        st.markdown("**Details du skill gap (Top 3)**")
        for i, r in enumerate(recs[:3], 1):
            gap = r.get("skill_gap", {})
            acquired = gap.get("acquired", [])
            missing = gap.get("missing", [])
            label = r.get("intitule") or r.get("candidate_name") or r.get("offer_id", "")
            with st.expander(f"#{i} — {label}"):
                col_acq, col_miss = st.columns(2)
                with col_acq:
                    st.markdown("**Competences acquises**")
                    if acquired:
                        for s in acquired:
                            st.markdown(f"- `{s}`")
                    else:
                        st.caption("Aucune")
                with col_miss:
                    st.markdown("**Competences manquantes**")
                    if missing:
                        for s in missing:
                            st.markdown(f"- `{s}`")
                    else:
                        st.caption("Aucune — profil complet")


# ─────────────────────────────────────────────
# PAGE 1 : VUE D'ENSEMBLE
# ─────────────────────────────────────────────
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble")

    @st.cache_data(ttl=300)
    def load_stats():
        return _api_get("/api/v1/stats")

    stats = load_stats()

    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Candidats", f"{stats['total_candidates']:,}")
        col2.metric("Offres", f"{stats['total_offers']:,}")
        col3.metric(
            "Encodes",
            f"{stats['encoded_candidates']:,} / {stats['encoded_offers']:,}",
        )
        col4.metric("Taux encodage", f"{stats['encoding_rate']:.1%}")

        avg_score = st.session_state.get("avg_recommendation_score")
        col5.metric("Score moyen", f"{avg_score:.2%}" if avg_score else "N/A")

        st.divider()

        dist = stats.get("distributions", {})

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Offres par secteur")
            sect_data = dist.get("offers_by_sector", {})
            if sect_data:
                df = pd.DataFrame(
                    list(sect_data.items()), columns=["Secteur", "Nombre"]
                ).sort_values("Nombre", ascending=False).head(15)
                st.bar_chart(df.set_index("Secteur"))
            else:
                st.info("Aucune donnee")

        with col_b:
            st.subheader("Metiers les plus demandes")
            top_f = stats.get("top_familles_offers", [])
            if top_f:
                df_top = pd.DataFrame(top_f)
                if "label" in df_top.columns:
                    df_top = df_top.rename(columns={"label": "Metier"})
                elif "id" in df_top.columns:
                    df_top = df_top.rename(columns={"id": "Metier"})
                st.bar_chart(df_top.set_index("Metier")["count"])
            else:
                st.info("Aucune donnee")

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Repartition geographique")
            dept_data = dist.get("candidates_by_department", {})
            if dept_data:
                df = pd.DataFrame(
                    list(dept_data.items()), columns=["Departement", "Nombre"]
                ).sort_values("Nombre", ascending=False)
                st.bar_chart(df.set_index("Departement"))
            else:
                st.info("Aucune donnee")

        with col_d:
            st.subheader("Niveau d'etudes")
            edu_data = dist.get("candidates_by_education", {})
            if edu_data:
                df = pd.DataFrame(
                    list(edu_data.items()), columns=["Niveau", "Nombre"]
                ).sort_values("Nombre", ascending=False)
                st.bar_chart(df.set_index("Niveau"))
            else:
                st.info("Aucune donnee")

        col_e, col_f = st.columns(2)
        with col_e:
            st.subheader("Offres par localisation")
            loc_data = dist.get("offers_by_localisation", {})
            if loc_data:
                df = pd.DataFrame(
                    list(loc_data.items()), columns=["Localisation", "Nombre"]
                ).sort_values("Nombre", ascending=False).head(12)
                st.bar_chart(df.set_index("Localisation"))
            else:
                st.info("Aucune donnee")

        with col_f:
            st.subheader("Statistiques recommandations")
            total_recs = st.session_state.get("total_recommendations_generated", 0)
            avg_score = st.session_state.get("avg_recommendation_score")
            rec_count = st.session_state.get("recommendations_count", 0)

            stat_rows = [
                {"Metrique": "Total recommandations generees", "Valeur": f"{total_recs:,}"},
                {"Metrique": "Score moyen", "Valeur": f"{avg_score:.2%}" if avg_score else "N/A"},
                {"Metrique": "Derniere session", "Valeur": f"{rec_count} recommandations"},
            ]
            df_stats = pd.DataFrame(stat_rows)
            st.dataframe(df_stats.set_index("Metrique"), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Types de contrat")
        contrat_data = dist.get("offers_by_contract", {})
        if contrat_data:
            cols = st.columns(len(contrat_data))
            for i, (contrat, count) in enumerate(sorted(contrat_data.items(), key=lambda x: -x[1])):
                cols[i].metric(contrat, f"{count:,}")

        st.caption(
            f"Derniere mise a jour : {stats.get('last_updated', 'N/A')[:16]} "
            f"· Refresh automatique toutes les 5 min"
        )


# ─────────────────────────────────────────────
# PAGE 2 : MATCHING
# ─────────────────────────────────────────────
elif page == "Matching":
    st.title("Matching Candidat > Offres")

    search_q = st.text_input(
        "Rechercher un candidat (nom, metier, secteur, lieu ou ID)",
        placeholder="Ex: Comptable, Infirmier, Brazzaville, ou ID PPBZV...",
    )

    if search_q:
        data = _api_get("/api/v1/candidates/search", params={"q": search_q, "limit": 20})
        if data is None:
            data = {"total": 0, "results": []}

        candidates = data["results"]
        st.info(f"{data['total']} candidat(s) trouve(s)")

        if candidates:
            options = []
            for c in candidates:
                name = f"{c.get('prenom') or ''} {c.get('nom') or ''}".strip() or c["id"]
                metier = c.get("metier_vise") or "N/A"
                lieu = c.get("lieu") or "N/A"
                options.append(f"{name} — {metier} ({lieu})")

            selected_label = st.selectbox("Selectionner un candidat", options)
            selected_idx = options.index(selected_label)
            candidate_id = candidates[selected_idx]["id"]
            candidate_info = candidates[selected_idx]

            with st.expander("Profil du candidat", expanded=False):
                fields = [
                    ("Nom complet", None),
                    ("ID", "id"),
                    ("Genre", "genre"),
                    ("Age", "age"),
                    ("Lieu", "lieu"),
                    ("Etudes", "etudes"),
                    ("Specialite", "specialite"),
                    ("Qualification", "qualification"),
                    ("Metier vise", "metier_vise"),
                    ("Secteur demande", "secteur_demande"),
                ]
                rows = []
                for label, key in fields:
                    if key is None:
                        val = f"{candidate_info.get('prenom') or ''} {candidate_info.get('nom') or ''}".strip()
                    else:
                        val = candidate_info.get(key)
                    rows.append({"Champ": label, "Valeur": str(val) if val else "-"})
                df = pd.DataFrame(rows)
                st.dataframe(df.set_index("Champ"), use_container_width=True, hide_index=True)

            top_k = st.slider("Nombre de recommandations", 1, 20, 10)

            if st.button("Lancer le matching", type="primary"):
                with st.spinner("Calcul des recommandations en cours..."):
                    results = _api_get(
                        f"/api/v1/matching/candidate/{candidate_id}",
                        params={"top_k": top_k},
                        timeout=30,
                    )

                if results:
                    recs = results["recommendations"]
                    st.success(
                        f"{len(recs)} offre(s) trouvee(s) pour "
                        f"**{results['candidate_name']}**"
                    )
                    _render_recommendations_table(recs)

                    if recs:
                        scores = [r["score"] for r in recs]
                        st.session_state["avg_recommendation_score"] = sum(scores) / len(scores)
                        st.session_state["total_recommendations_generated"] = (
                            st.session_state.get("total_recommendations_generated", 0) + len(recs)
                        )


# ─────────────────────────────────────────────
# PAGE 3 : RECOMMANDATIONS
# ─────────────────────────────────────────────
elif page == "Recommandations":
    st.title("Recommandations")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_export = st.text_input(
            "Rechercher des candidats",
            placeholder="Nom, metier, secteur, ou ID...",
        )
    with col2:
        top_k_export = st.number_input("Top-K", min_value=1, max_value=50, value=10)

    selected_ids = []

    if search_export:
        data = _api_get(
            "/api/v1/candidates/search",
            params={"q": search_export, "limit": 50},
        )
        candidates = data["results"] if data else []

        if candidates:
            st.info(f"{len(candidates)} candidat(s) trouve(s)")

            options_map = {}
            options_labels = []
            for c in candidates:
                name = f"{c.get('prenom') or ''} {c.get('nom') or ''}".strip() or c["id"]
                metier = c.get("metier_vise") or "N/A"
                label = f"{name} — {metier}"
                options_map[label] = c["id"]
                options_labels.append(label)

            selected_labels = st.multiselect(
                "Selectionner des candidats",
                options_labels,
            )
            selected_ids = [options_map[l] for l in selected_labels]

    if selected_ids:
        st.divider()
        st.info(f"{len(selected_ids)} candidat(s) selectionne(s)")

        if st.button("Generer les recommandations", type="primary"):
            all_recs = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, cand_id in enumerate(selected_ids, 1):
                status_text.text(f"Traitement {idx}/{len(selected_ids)} — {cand_id}")
                progress_bar.progress(idx / len(selected_ids))

                result = _api_get(
                    f"/api/v1/matching/candidate/{cand_id}",
                    params={"top_k": top_k_export},
                    timeout=30,
                )
                if result and result.get("recommendations"):
                    cand_name = result.get("candidate_name", cand_id)
                    for r in result["recommendations"]:
                        all_recs.append({
                            "Candidat": cand_name,
                            "ID Candidat": cand_id,
                            "Rang": len([x for x in all_recs if x["ID Candidat"] == cand_id]) + 1,
                            "Offre": r.get("intitule", "N/A"),
                            "Entreprise": r.get("entreprise", "N/A"),
                            "Score": round(r["score"], 3),
                            "Couverture": f"{r.get('skill_gap', {}).get('gap_score', 0):.0%}",
                        })

            progress_bar.progress(1.0)
            status_text.text("Termine")

            if all_recs:
                df_recs = pd.DataFrame(all_recs)
                st.subheader(f"{len(all_recs)} recommandation(s) pour {len(selected_ids)} candidat(s)")
                st.dataframe(df_recs, use_container_width=True, hide_index=True)

                if "Score" in df_recs.columns:
                    st.session_state["avg_recommendation_score"] = df_recs["Score"].mean()
                    st.session_state["total_recommendations_generated"] = (
                        st.session_state.get("total_recommendations_generated", 0) + len(all_recs)
                    )
                    st.session_state["recommendations_count"] = len(all_recs)

                st.divider()
                st.subheader("Exporter les resultats")

                csv_data = df_recs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Telecharger CSV ({len(all_recs)} lignes)",
                    data=csv_data,
                    file_name="acpe_recommandations.csv",
                    mime="text/csv",
                    type="primary",
                )
            else:
                st.warning("Aucune recommandation generee.")


# ─────────────────────────────────────────────
# PAGE 4 : OFFRE → CANDIDATS
# ─────────────────────────────────────────────
elif page == "Offre > Candidats":
    st.title("Offre > Meilleurs candidats")

    offer_q = st.text_input(
        "Rechercher une offre (reference, intitule, secteur ou entreprise)",
        placeholder="Ex: JOB250002487, Comptable, Finance, NEEDLEWORK...",
    )

    if offer_q:
        data = _api_get(
            "/api/v1/job-offers/search",
            params={"q": offer_q, "limit": 20},
        )
        offers = data["results"] if data else []

        if offers:
            st.info(f"{len(offers)} offre(s) trouvee(s)")
            options = []
            for o in offers:
                intitule = o.get("intitule") or "N/A"
                entreprise = o.get("entreprise") or "N/A"
                secteur = o.get("secteur") or "N/A"
                options.append(f"{o['id']} — {intitule} ({entreprise} · {secteur})")

            selected_label = st.selectbox("Selectionner une offre", options)
            selected_idx = options.index(selected_label)
            offer_id = offers[selected_idx]["id"]
            offer_info = offers[selected_idx]

            with st.expander("Detail de l'offre", expanded=False):
                fields = [
                    ("Reference", "id"),
                    ("Intitule", "intitule"),
                    ("Poste", "poste"),
                    ("Entreprise", "entreprise"),
                    ("Type contrat", "type_contrat"),
                    ("Type entreprise", "type_entreprise"),
                    ("Secteur", "secteur"),
                    ("Localisation", "localisation"),
                    ("Date publication", "date_publication"),
                    ("Description", "description"),
                    ("Competences recherchees", "competences_recherchees"),
                ]
                rows = []
                for label, key in fields:
                    val = offer_info.get(key)
                    rows.append({"Champ": label, "Valeur": str(val) if val else "-"})
                df = pd.DataFrame(rows)
                st.dataframe(df.set_index("Champ"), use_container_width=True, hide_index=True)

            top_k_offer = st.slider("Nombre de candidats (top-K)", 1, 20, 5)

            if st.button("Trouver les meilleurs candidats", type="primary"):
                with st.spinner("Calcul des recommandations en cours..."):
                    results = _api_get(
                        f"/api/v1/matching/offer/{offer_id}",
                        params={"top_k": top_k_offer},
                        timeout=30,
                    )

                if results:
                    recs = results["recommendations"]
                    st.success(
                        f"{len(recs)} candidat(s) pour "
                        f"**{results['offer_intitule']}** "
                        f"@ {results.get('offer_entreprise') or 'N/A'}"
                    )
                    _render_recommendations_table(recs)

            st.divider()
            if st.button("Generer le CSV (offre > candidats)", type="primary"):
                with st.spinner("Generation du fichier CSV en cours..."):
                    try:
                        csv_resp = requests.get(
                            f"{API_URL}/api/v1/matching/export-csv-by-offer",
                            params={"offer_ids": offer_id, "top_k": top_k_offer},
                            timeout=60,
                            stream=True,
                        )
                        csv_resp.raise_for_status()
                        st.download_button(
                            label="Telecharger CSV",
                            data=csv_resp.content,
                            file_name="acpe_offre_candidats_export.csv",
                            mime="text/csv",
                            type="primary",
                        )
                        preview = csv_resp.content.decode("utf-8").split("\n")
                        st.success(f"CSV genere — {len(preview) - 1} lignes")
                        with st.expander("Apercu du CSV"):
                            st.code("\n".join(preview[:20]))
                    except Exception as e:
                        st.error(f"Erreur generation CSV : {e}")

    st.divider()
    st.caption(
        "Astuce : collez directement une reference d'offre "
        "(ex: JOB250002487) dans la recherche ci-dessus."
    )


# ─────────────────────────────────────────────
# PAGE 5 : RECHERCHE NL OFFRES
# ─────────────────────────────────────────────
elif page == "Recherche NL":
    st.title("Recherche d'offres en langage naturel")
    st.caption(
        "Decrivez librement ce que vous cherchez — competences, secteur, "
        "metier — et le moteur classe les offres pertinentes."
    )

    nl_query = st.text_area(
        "Votre recherche libre",
        placeholder=(
            "Ex: Je cherche une offre dans la restauration, je sais faire "
            "la cuisine et manager une equipe..."
        ),
        height=120,
    )
    nl_k = st.slider("Nombre d'offres (top-K)", 1, 20, 10)

    if st.button("Rechercher", type="primary", disabled=not nl_query.strip()):
        with st.spinner("Recherche et classement des offres..."):
            nl_results = _api_get(
                "/api/v1/matching/nl-offer-search",
                params=None,
                timeout=30,
            )
            # POST request needed
            try:
                resp = requests.post(
                    f"{API_URL}/api/v1/matching/nl-offer-search",
                    json={"query": nl_query, "top_k": nl_k},
                    timeout=30,
                )
                resp.raise_for_status()
                nl_results = resp.json()
            except Exception as e:
                st.error(f"Erreur API : {e}")
                nl_results = None

        if nl_results:
            if nl_results.get("error"):
                st.error(nl_results["error"])
            else:
                skills = nl_results.get("extracted_skills", [])
                secteur = nl_results.get("target_secteur")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric(
                        "Competences detectees",
                        ", ".join(skills) if skills else "Aucune",
                    )
                with col_s2:
                    st.metric(
                        "Secteur vise",
                        secteur.get("secteur_canonique") if secteur else "Non detecte",
                    )

                recs = nl_results.get("recommendations", [])
                if not recs:
                    st.warning("Aucune offre ne correspond suffisamment a votre recherche.")
                else:
                    rows = []
                    for i, rec in enumerate(recs, 1):
                        matched = rec.get("matched_skills", [])
                        rows.append({
                            "Rang": i,
                            "Intitule": rec.get("intitule", "N/A"),
                            "Entreprise": rec.get("entreprise", "N/A"),
                            "Secteur": rec.get("secteur", "N/A"),
                            "Score": round(rec["score"], 3),
                            "Match competences": len(matched),
                            "Match secteur": "Oui" if rec.get("sector_match") else "Non",
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    with st.expander("Details des competences (Top 3)"):
                        for i, rec in enumerate(recs[:3], 1):
                            matched = rec.get("matched_skills", [])
                            st.markdown(f"**#{i} — {rec.get('intitule', 'N/A')}**")
                            if matched:
                                for s in matched:
                                    st.markdown(f"- `{s}`")
                            else:
                                st.caption("Aucune competence en commun")
                            col_m, col_s = st.columns(2)
                            with col_m:
                                st.metric("Score de pertinence", f"{rec['score']:.2f}")
                            with col_s:
                                st.metric(
                                    "Similarite semantique",
                                    f"{rec.get('semantic_score', 0):.2f}",
                                )


# ─────────────────────────────────────────────
# PAGE 6 : RAPPORT TECHNIQUE
# ─────────────────────────────────────────────
elif page == "Rapport":
    st.title("Rapport Technique")
    st.caption("ACPE Match — IndabaX Congo 2026")

    rapport_path = os.path.join(os.path.dirname(__file__), "rapport_technique.md")
    if os.path.exists(rapport_path):
        with open(rapport_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning(
            "Fichier rapport_technique.md introuvable. "
            "Creez-le dans le dossier backend/."
        )
