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
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔗 ACPE Match")
st.sidebar.caption("IndabaX Congo 2026")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Vue d'ensemble", "🎯 Matching", "📥 Export CSV", "🔄 Offre → Candidats", "🔍 Recherche NL Offres", "📋 Rapport"],
)
st.sidebar.divider()
st.sidebar.caption("API: " + API_URL)


# ─────────────────────────────────────────────
# PAGE 1 : VUE D'ENSEMBLE
# ─────────────────────────────────────────────
if page == "📊 Vue d'ensemble":
    st.title("📊 Vue d'ensemble")

    @st.cache_data(ttl=300)
    def load_stats():
        try:
            r = requests.get(f"{API_URL}/api/v1/stats", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    stats = load_stats()

    if isinstance(stats, dict) and "error" in stats:
        st.error(f"Erreur de connexion à l'API : {stats['error']}")
        stats = None

    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👤 Candidats", f"{stats['total_candidates']:,}")
        col2.metric("💼 Offres", f"{stats['total_offers']:,}")
        col3.metric(
            "🔗 Encodés",
            f"{stats['encoded_candidates']:,} / {stats['encoded_offers']:,}",
        )
        col4.metric("📈 Taux encodage", f"{stats['encoding_rate']:.1%}")

        st.divider()

        dist = stats["distributions"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Candidats par département")
            dept_data = dist.get("candidates_by_department", {})
            if dept_data:
                df = pd.DataFrame(
                    list(dept_data.items()), columns=["Département", "Nombre"]
                ).sort_values("Nombre", ascending=False)
                st.bar_chart(df.set_index("Département"))
            else:
                st.info("Aucune donnée")

        with col_b:
            st.subheader("Offres par secteur")
            sect_data = dist.get("offers_by_sector", {})
            if sect_data:
                df = pd.DataFrame(
                    list(sect_data.items()), columns=["Secteur", "Nombre"]
                ).sort_values("Nombre", ascending=False).head(15)
                st.bar_chart(df.set_index("Secteur"))
            else:
                st.info("Aucune donnée")

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Offres par type de contrat")
            contrat_data = dist.get("offers_by_contract", {})
            if contrat_data:
                df = pd.DataFrame(
                    list(contrat_data.items()), columns=["Contrat", "Nombre"]
                )
                st.bar_chart(df.set_index("Contrat"))
            else:
                st.info("Aucune donnée")

        with col_d:
            st.subheader("Candidats par niveau d'études")
            edu_data = dist.get("candidates_by_education", {})
            if edu_data:
                df = pd.DataFrame(
                    list(edu_data.items()), columns=["Niveau", "Nombre"]
                ).sort_values("Nombre", ascending=False)
                st.bar_chart(df.set_index("Niveau"))
            else:
                st.info("Aucune donnée")

        st.divider()

        top_f = stats.get("top_familles_offers", [])
        if top_f:
            st.subheader("Top 10 familles d'offres")
            df_top = pd.DataFrame(top_f)
            st.bar_chart(df_top.set_index("id")["count"])

        st.caption(
            f"🕐 Dernière mise à jour : {stats['last_updated'][:16]} "
            f"· Refresh automatique toutes les 5 min"
        )


# ─────────────────────────────────────────────
# PAGE 2 : MATCHING
# ─────────────────────────────────────────────
elif page == "🎯 Matching":
    st.title("🎯 Matching Candidat → Offres")

    search_q = st.text_input(
        "Rechercher un candidat (nom, métier, secteur, lieu ou ID)",
        placeholder="Ex: Comptable, Infirmier, Brazzaville, ou ID PPBZV...",
    )

    if search_q:
        try:
            resp = requests.get(
                f"{API_URL}/api/v1/candidates/search",
                params={"q": search_q, "limit": 20},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Erreur API : {e}")
            data = {"total": 0, "results": []}

        candidates = data["results"]
        st.info(f"{data['total']} candidat(s) trouvé(s)")

        if candidates:
            options = []
            for c in candidates:
                name = f"{c.get('prenom') or ''} {c.get('nom') or ''}".strip() or c["id"]
                metier = c.get("metier_vise") or "N/A"
                lieu = c.get("lieu") or "N/A"
                options.append(f"{name} — {metier} ({lieu})")

            selected_label = st.selectbox("Sélectionner un candidat", options)
            selected_idx = options.index(selected_label)
            candidate_id = candidates[selected_idx]["id"]
            candidate_info = candidates[selected_idx]

            with st.expander("📋 Profil du candidat", expanded=False):
                st.json(candidate_info)

            top_k = st.slider("Nombre de recommandations", 1, 20, 10)

            if st.button("🚀 Lancer le matching", type="primary"):
                with st.spinner("Calcul des recommandations en cours..."):
                    try:
                        match_resp = requests.get(
                            f"{API_URL}/api/v1/matching/candidate/{candidate_id}",
                            params={"top_k": top_k},
                            timeout=30,
                        )
                        match_resp.raise_for_status()
                        results = match_resp.json()
                    except Exception as e:
                        st.error(f"Erreur matching : {e}")
                        results = None

                if results:
                    recs = results["recommendations"]
                    st.success(
                        f"✅ {len(recs)} offre(s) trouvée(s) pour "
                        f"**{results['candidate_name']}**"
                    )

                    max_score = max((r["score"] for r in recs), default=1.0) or 1.0

                    for i, rec in enumerate(recs, 1):
                        norm = min(rec["score"] / max_score, 1.0) if max_score else 0.0
                        gap = rec.get("skill_gap", {})
                        acquired = gap.get("acquired", [])
                        missing = gap.get("missing", [])
                        gap_score = gap.get("gap_score", 0)

                        with st.expander(
                            f"#{i} — {rec.get('intitule', 'N/A')} "
                            f"| {rec.get('entreprise', 'N/A')} "
                            f"| Score relatif: {norm:.0%}",
                            expanded=(i == 1),
                        ):
                            st.progress(norm)

                            col_acq, col_miss = st.columns(2)
                            with col_acq:
                                st.markdown("**🟢 Compétences acquises**")
                                if acquired:
                                    for s in acquired:
                                        st.markdown(f"🟢 `{s}`")
                                else:
                                    st.caption("Aucune")

                            with col_miss:
                                st.markdown("**🔴 Compétences manquantes**")
                                if missing:
                                    for s in missing:
                                        st.markdown(f"🔴 `{s}`")
                                else:
                                    st.caption("Aucune — profil complet !")

                            st.metric(
                                "Taux de couverture",
                                f"{gap_score:.0%}",
                            )
                            st.metric(
                                "Score modèle (brut)",
                                f"{rec['score']:.2f}",
                            )


# ─────────────────────────────────────────────
# PAGE 3 : EXPORT CSV
# ─────────────────────────────────────────────
elif page == "📥 Export CSV":
    st.title("📥 Export CSV des recommandations")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_export = st.text_input(
            "Rechercher des candidats à exporter",
            placeholder="Nom, métier, secteur, ou ID...",
        )
    with col2:
        top_k_export = st.number_input("Top-K", min_value=1, max_value=50, value=10)

    selected_ids = []

    if search_export:
        try:
            resp = requests.get(
                f"{API_URL}/api/v1/candidates/search",
                params={"q": search_export, "limit": 50},
                timeout=10,
            )
            resp.raise_for_status()
            candidates = resp.json()["results"]
        except Exception as e:
            st.error(f"Erreur API : {e}")
            candidates = []

        if candidates:
            st.info(f"{len(candidates)} candidat(s) trouvé(s)")
            select_all = st.checkbox("Tout sélectionner")

            for c in candidates:
                name = (
                    f"{c.get('prenom') or ''} {c.get('nom') or ''}".strip() or c["id"]
                )
                metier = c.get("metier_vise") or "N/A"
                if st.checkbox(f"{name} — {metier}", value=select_all, key=f"exp_{c['id']}"):
                    selected_ids.append(c["id"])

    if selected_ids:
        st.divider()
        st.info(f"📥 {len(selected_ids)} candidat(s) sélectionné(s)")

        if st.button("Générer le CSV", type="primary"):
            with st.spinner("Génération du fichier CSV en cours..."):
                try:
                    csv_resp = requests.get(
                        f"{API_URL}/api/v1/matching/export-csv",
                        params={
                            "candidate_ids": ",".join(selected_ids),
                            "top_k": top_k_export,
                        },
                        timeout=60,
                        stream=True,
                    )
                    csv_resp.raise_for_status()

                    st.download_button(
                        label=f"📥 Télécharger CSV ({len(selected_ids)} candidats)",
                        data=csv_resp.content,
                        file_name="acpe_matching_export.csv",
                        mime="text/csv",
                        type="primary",
                    )

                    preview = csv_resp.content.decode("utf-8").split("\n")
                    st.success(f"CSV généré — {len(preview) - 1} lignes")
                    with st.expander("Aperçu du CSV"):
                        st.code("\n".join(preview[:20]))

                except Exception as e:
                    st.error(f"Erreur génération CSV : {e}")


# ─────────────────────────────────────────────
# PAGE 4 : OFFRE → CANDIDATS
# ─────────────────────────────────────────────
elif page == "🔄 Offre → Candidats":
    st.title("🔄 Offre → Meilleurs candidats")

    offer_q = st.text_input(
        "Rechercher une offre (référence, intitulé, secteur ou entreprise)",
        placeholder="Ex: JOB250002487, Comptable, Finance, NEEDLEWORK...",
    )

    if offer_q:
        try:
            resp = requests.get(
                f"{API_URL}/api/v1/job-offers/search",
                params={"q": offer_q, "limit": 20},
                timeout=10,
            )
            resp.raise_for_status()
            offers = resp.json()["results"]
        except Exception as e:
            st.error(f"Erreur API : {e}")
            offers = []

        if offers:
            st.info(f"{len(offers)} offre(s) trouvée(s)")
            options = []
            for o in offers:
                intitule = o.get("intitule") or "N/A"
                entreprise = o.get("entreprise") or "N/A"
                secteur = o.get("secteur") or "N/A"
                options.append(f"{o['id']} — {intitule} ({entreprise} · {secteur})")

            selected_label = st.selectbox("Sélectionner une offre", options)
            selected_idx = options.index(selected_label)
            offer_id = offers[selected_idx]["id"]
            offer_info = offers[selected_idx]

            with st.expander("📋 Détail de l'offre", expanded=False):
                st.json(offer_info)

            top_k_offer = st.slider("Nombre de candidats (top-K)", 1, 20, 5)

            if st.button("🚀 Trouver les meilleurs candidats", type="primary"):
                with st.spinner("Calcul des recommandations en cours..."):
                    try:
                        match_resp = requests.get(
                            f"{API_URL}/api/v1/matching/offer/{offer_id}",
                            params={"top_k": top_k_offer},
                            timeout=30,
                        )
                        match_resp.raise_for_status()
                        results = match_resp.json()
                    except Exception as e:
                        st.error(f"Erreur matching : {e}")
                        results = None

                if results:
                    recs = results["recommendations"]
                    st.success(
                        f"✅ {len(recs)} candidat(s) pour "
                        f"**{results['offer_intitule']}** "
                        f"@ {results.get('offer_entreprise') or 'N/A'}"
                    )

                    max_score = max((r["score"] for r in recs), default=1.0) or 1.0

                    for i, rec in enumerate(recs, 1):
                        norm = min(rec["score"] / max_score, 1.0) if max_score else 0.0
                        gap = rec.get("skill_gap", {})
                        acquired = gap.get("acquired", [])
                        missing = gap.get("missing", [])
                        gap_score = gap.get("gap_score", 0)

                        with st.expander(
                            f"#{i} — {rec.get('candidate_name') or rec['candidate_id']} "
                            f"| {rec.get('metier_vise') or 'N/A'} "
                            f"| {rec.get('lieu') or 'N/A'} "
                            f"| Score relatif: {norm:.0%}",
                            expanded=(i == 1),
                        ):
                            st.progress(norm)

                            col_acq, col_miss = st.columns(2)
                            with col_acq:
                                st.markdown("**🟢 Compétences acquises**")
                                if acquired:
                                    for s in acquired:
                                        st.markdown(f"🟢 `{s}`")
                                else:
                                    st.caption("Aucune")

                            with col_miss:
                                st.markdown("**🔴 Compétences manquantes**")
                                if missing:
                                    for s in missing:
                                        st.markdown(f"🔴 `{s}`")
                                else:
                                    st.caption("Aucune — profil complet !")

                            st.metric("Taux de couverture", f"{gap_score:.0%}")
                            st.metric("Score modèle (brut)", f"{rec['score']:.2f}")

            st.divider()
            if st.button("📥 Générer le CSV (offre → candidats)", type="primary"):
                with st.spinner("Génération du fichier CSV en cours..."):
                    try:
                        csv_resp = requests.get(
                            f"{API_URL}/api/v1/matching/export-csv-by-offer",
                            params={"offer_ids": offer_id, "top_k": top_k_offer},
                            timeout=60,
                            stream=True,
                        )
                        csv_resp.raise_for_status()
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv_resp.content,
                            file_name="acpe_offre_candidats_export.csv",
                            mime="text/csv",
                            type="primary",
                        )
                        preview = csv_resp.content.decode("utf-8").split("\n")
                        st.success(f"CSV généré — {len(preview) - 1} lignes")
                        with st.expander("Aperçu du CSV"):
                            st.code("\n".join(preview[:20]))
                    except Exception as e:
                        st.error(f"Erreur génération CSV : {e}")

    st.divider()
    st.caption(
        "Astuce : collez directement une référence d'offre "
        "(ex: JOB250002487) dans la recherche ci-dessus."
    )


# ─────────────────────────────────────────────
# PAGE 5 : RECHERCHE NL OFFRES
# ─────────────────────────────────────────────
elif page == "🔍 Recherche NL Offres":
    st.title("🔍 Recherche d'offres en langage naturel")
    st.caption(
        "Décrivez librement ce que vous cherchez — compétences, secteur, "
        "métier — et le moteur classe les offres pertinentes."
    )

    nl_query = st.text_area(
        "Votre recherche libre",
        placeholder=(
            "Ex: Je cherche une offre dans la restauration, je sais faire "
            "la cuisine et manager une équipe..."
        ),
        height=120,
    )
    nl_k = st.slider("Nombre d'offres (top-K)", 1, 20, 10)

    if st.button("🔎 Rechercher", type="primary", disabled=not nl_query.strip()):
        with st.spinner("Recherche et classement des offres..."):
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
                st.info(
                    f"🧩 Compétences détectées : {', '.join(skills) if skills else 'aucune'} "
                    f"· 🏷️ Secteur visé : {secteur['secteur_canonique'] if secteur else 'non détecté'}"
                )

                recs = nl_results.get("recommendations", [])
                if not recs:
                    st.warning("Aucune offre ne correspond suffisamment à votre recherche.")
                else:
                    max_score = max((r["score"] for r in recs), default=1.0) or 1.0
                    for i, rec in enumerate(recs, 1):
                        norm = min(rec["score"] / max_score, 1.0) if max_score else 0.0
                        matched = rec.get("matched_skills", [])
                        with st.expander(
                            f"#{i} — {rec.get('intitule', 'N/A')} "
                            f"| {rec.get('entreprise', 'N/A')} "
                            f"| {rec.get('secteur', 'N/A')} "
                            f"| Score: {norm:.0%}",
                            expanded=(i == 1),
                        ):
                            st.progress(norm)

                            col_m, col_s = st.columns(2)
                            with col_m:
                                st.markdown("**🟢 Compétences matchées**")
                                if matched:
                                    for s in matched:
                                        st.markdown(f"🟢 `{s}`")
                                else:
                                    st.caption("Aucune compétence en commun")
                            with col_s:
                                st.markdown("**🏷️ Secteur**")
                                if rec.get("sector_match"):
                                    st.success(
                                        f"✅ {rec.get('secteur')} correspond au secteur visé"
                                    )
                                else:
                                    st.caption(rec.get("secteur") or "N/A")

                            st.metric("Score de pertinence", f"{rec['score']:.2f}")
                            st.metric(
                                "Similarité sémantique",
                                f"{rec.get('semantic_score', 0):.2f}",
                            )
                            if rec.get("catboost_score") is not None:
                                st.caption(f"Score CatBoost (info) : {rec['catboost_score']:.2f}")


# ─────────────────────────────────────────────
# PAGE 6 : RAPPORT TECHNIQUE
# ─────────────────────────────────────────────
elif page == "📋 Rapport":
    st.title("📋 Rapport Technique")
    st.caption("ACPE Match — IndabaX Congo 2026")

    rapport_path = os.path.join(os.path.dirname(__file__), "rapport_technique.md")
    if os.path.exists(rapport_path):
        with open(rapport_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning(
            "Fichier rapport_technique.md introuvable. "
            "Créez-le dans le dossier backend/."
        )
