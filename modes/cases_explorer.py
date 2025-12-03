import pandas as pd
import streamlit as st

from styles import AGGRID_CUSTOM_CSS  # not used here yet, but fine if you later add AgGrid


def render_cases_explorer(helpers):
    """Cases Explorer (browse Supabase cases, view PPR + FMEA; tables only)."""

    _build_supabase = helpers["_build_supabase"]

    st.title("Cases Explorer")

    sb = _build_supabase()

    # 1) Fetch cases with basic fields
    with st.spinner("Loading cases..."):
        try:
            cases = sb.table("cases").select("id, title, description, created_at").order("id", desc=True).execute().data or []
        except Exception as e:
            st.error(f"Failed to load cases: {e}")
            cases = []

    left, right = st.columns([2, 5], vertical_alignment="top")

    with left:
        st.subheader("All cases")

        # Search over id/title/description
        q = st.text_input("Search (id/title/desc)", value="", placeholder="ex: battery, weld, 1023")
        filtered = cases
        if q.strip():
            ql = q.lower().strip()
            def _match(c):
                return (
                    ql in str(c.get("id", "")).lower() or
                    ql in (c.get("title") or "").lower() or
                    ql in (c.get("description") or "").lower()
                )
            filtered = [c for c in cases if _match(c)]

        # Single select list (no Prev/Next)
        options = [f'#{c["id"]} • {(c.get("title") or "Untitled")[:60]}' for c in filtered]
        idx_map = {options[i]: filtered[i]["id"] for i in range(len(filtered))}
        selected_label = st.selectbox("Select a case", options=options or ["—"], index=0 if options else None, key="cx_case_select")
        selected_case_id = idx_map.get(selected_label)

        # Metadata with description
        if selected_case_id:
            sel = next((c for c in filtered if c["id"] == selected_case_id), None)
            if sel:
                st.caption(f'Case ID: {sel["id"]}')
                st.caption(f'Title: {sel.get("title") or "—"}')
                if sel.get("description"):
                    st.caption(f'Description: {sel["description"]}')
                st.caption(f'Created: {sel.get("created_at") or "—"}')

    with right:
        st.subheader("Case details")

        if not selected_case_id:
            st.info("Select a case on the left to view details.")
            st.stop()

        # 2) Load PPR via links (case_products/case_processes/case_resources/case_inputs)
        def _load_ppr_for_case(case_id: int):
            try:
                # Relationship selects return nested objects with 'name'
                prods = sb.table("case_products").select("product_id, products(name)").eq("case_id", case_id).execute().data or []
                procs = sb.table("case_processes").select("process_id, processes(name)").eq("case_id", case_id).execute().data or []
                ress  = sb.table("case_resources").select("resource_id, resources(name)").eq("case_id", case_id).execute().data or []
                inps  = sb.table("case_inputs").select("input_id, inputs(name)").eq("case_id", case_id).execute().data or []

                # Extract names safely; ensures Input Products are filled if linked
                inputs = sorted({(row.get("inputs") or {}).get("name", "") for row in inps if (row.get("inputs") or {}).get("name")})
                products = sorted({(row.get("products") or {}).get("name", "") for row in prods if (row.get("products") or {}).get("name")})
                processes = sorted({(row.get("processes") or {}).get("name", "") for row in procs if (row.get("processes") or {}).get("name")})
                resources = sorted({(row.get("resources") or {}).get("name", "") for row in ress if (row.get("resources") or {}).get("name")})

                return {
                    "input_products": inputs,
                    "products": products,
                    "processes": processes,
                    "resources": resources,
                }
            except Exception as e:
                st.error(f"Failed to load PPR: {e}")
                return {"input_products": [], "products": [], "processes": [], "resources": []}

        # 3) Load FMEA rows
        def _load_fmea_for_case(case_id: int):
            try:
                rows = sb.table("fmea_rows").select("*").eq("case_id", case_id).order("id", desc=False).execute().data or []
                return rows
            except Exception as e:
                st.error(f"Failed to load FMEA rows: {e}")
                return []

        # Data
        ppr = _load_ppr_for_case(selected_case_id)
        fmea_rows = _load_fmea_for_case(selected_case_id)

        # Tabs for viewing (tables only)
        tabs = st.tabs(["PPR table", "FMEA table"])

        with tabs[0]:
            st.caption("PPR")
            def _list_to_df(name, items):
                return pd.DataFrame({name: sorted({x for x in (items or []) if x})})

            c1, c2 = st.columns(2)
            with c1:
                st.write("Input Products")
                st.dataframe(_list_to_df("Input Products", ppr.get("input_products")), use_container_width=True)
                st.write("Processes")
                st.dataframe(_list_to_df("Processes", ppr.get("processes")), use_container_width=True)
            with c2:
                st.write("Output Products")
                st.dataframe(_list_to_df(" Output Products", ppr.get("products")), use_container_width=True)
                st.write("Resources")
                st.dataframe(_list_to_df("Resources", ppr.get("resources")), use_container_width=True)

        with tabs[1]:
            st.caption("FMEA rows")
            if not fmea_rows:
                st.info("No FMEA rows stored for this case.")
            else:
                df = pd.DataFrame(fmea_rows)

                def _sint(v):
                    try: return int(str(v).strip())
                    except: return 0

                if all(col in df.columns for col in ["s1","o1","d1"]) and "rpn1" not in df.columns:
                    df["rpn1"] = df.apply(lambda r: _sint(r.get("s1")) * _sint(r.get("o1")) * _sint(r.get("d1")), axis=1)
                if all(col in df.columns for col in ["s2","o2","d2"]) and "rpn2" not in df.columns:
                    df["rpn2"] = df.apply(lambda r: _sint(r.get("s2")) * _sint(r.get("o2")) * _sint(r.get("d2")), axis=1)

                hide_cols = [c for c in ["id", "case_id"] if c in df.columns]
                show_df = df[[c for c in df.columns if c not in hide_cols]]

                st.dataframe(show_df, use_container_width=True)

  
