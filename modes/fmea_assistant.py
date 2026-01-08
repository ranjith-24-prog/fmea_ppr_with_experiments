import json
import datetime as dt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
from styles import AGGRID_CUSTOM_CSS
from backend.llm import LLM, LLM_REGISTRY
from backend.export import to_pretty_excel_bytes, to_structured_xml_bytes
import uuid


def render_fmea_assistant(embedder, helpers):
    """
    FMEA Assistant (FMEA first; PPR editor visible after; KB-style PPR gen with input hints).

    embedder: Embeddings instance from app.py
    helpers: dict of helper functions/values from app.py
    """

    # ---- helpers coming from app.py ----
    _normalize_ppr_safe = helpers["_normalize_ppr_safe"]
    _build_supabase = helpers["_build_supabase"]
    _select_kb_rows = helpers["_select_kb_rows"]
    _complete_missing_with_llm = helpers["_complete_missing_with_llm"]
    _normalize_numeric_and_rpn = helpers["_normalize_numeric_and_rpn"]
    _to_plain_list = helpers["_to_plain_list"]
    _get_secret = helpers["_get_secret"]
    ppr_editor_block = helpers["ppr_editor_block"]

    st.title("Case-based FMEA Assistant")

    # Timing slots
    st.session_state.setdefault("fa_fmea_kb_ms", None)    # KB retrieval only
    st.session_state.setdefault("fa_fmea_llm_ms", None)   # LLM gap-fill only
    st.session_state.setdefault("fa_fmea_ms", None)       # Overall (button to ready)
    st.session_state.setdefault("fa_ppr_ms", None)        # PPR LLM time

    # 0) Ensure session defaults exist
    st.session_state.setdefault("fa_user_text", "")
    st.session_state.setdefault(
        "assistant_ppr",
        _normalize_ppr_safe(
            {"input_products": [], "products": [], "processes": [], "resources": []}
        ),
    )

    # 1) Single input box (persisted)
    user_text = st.text_area(
        "Enter the description of FMEA required. (please specify product/process/resources details if possible) ",
        height=120,
        placeholder=(
            "Example: Manual Aluminium Airframe TIG Welding involves joining aluminum airframe components "
            "using TIG welding with suitable filler material and shielding gas. The process includes preparation, "
            "setup, welding execution, post-weld treatments, and inspection. Skilled operators utilize TIG welding "
            "and NDT equipment to ensure high-quality, defect-free welds."
        ),
        key="fa_user_text",
        value=st.session_state.get("fa_user_text", ""),
    )

    # LLM selector
    _model_items = [(mid, cfg["label"]) for mid, cfg in LLM_REGISTRY.items()]
    if "active_model_id" not in st.session_state:
        _default_mid = (
            "perplexity/sonar-pro"
            if "perplexity/sonar-pro" in LLM_REGISTRY
            else _model_items[0][0]
        )
        st.session_state["active_model_id"] = _default_mid
    _current_label = LLM_REGISTRY[st.session_state["active_model_id"]]["label"]

    col1, col2 = st.columns([3, 2])
    with col1:
        selected_label = st.selectbox(
            "Select LLM",
            options=[label for _, label in _model_items],
            index=[label for _, label in _model_items].index(_current_label),
            key="fa_llm_select",
        )
    with col2:
        st.caption(
            "The selected model will be used for Generate FMEA and Generate PPR in this section."
        )

    # Map label back to model id
    for mid, label in _model_items:
        if label == selected_label:
            st.session_state["active_model_id"] = mid
            break

    # Inline status line
    from os import getenv as _getenv

    _cfg = LLM_REGISTRY[st.session_state["active_model_id"]]
    _api_ok = bool(_getenv(_cfg["env"], ""))
    st.markdown(
        f"<small>Using: <code>{_cfg['label']}</code> · API key: "
        f"<span style='color:{'lime' if _api_ok else 'tomato'}'>{'OK' if _api_ok else 'Missing'}</span></small>",
        unsafe_allow_html=True,
    )

    # LLM client
    llm = LLM(model_name=st.session_state["active_model_id"])
    llm.set_model(st.session_state["active_model_id"])

    # Mirror latest description
    user_text = st.session_state.get("fa_user_text", user_text or "")

    # --- Retrieval-only PPR from text (for KB similarity)
    def _derive_ppr_from_text(txt: str) -> dict:
        import re

        tokens = [t.strip() for t in re.split(r"[,\n;]", txt or "") if t.strip()]
        products, processes, resources, inputs = [], [], [], []
        PROC = [
            "weld",
            "bond",
            "cut",
            "drill",
            "mill",
            "assembly",
            "assemble",
            "coating",
            "paint",
            "inspection",
            "inspect",
            "test",
            "testing",
            "grind",
            "polish",
            "form",
            "press",
            "stamp",
            "laser",
            "brazing",
            "solder",
            "adhesive",
            "riveting",
            "clinch",
            "deburr",
            "heat treat",
        ]
        RES = [
            "gun",
            "torch",
            "camera",
            "fixture",
            "jig",
            "robot",
            "laser",
            "sensor",
            "nozzle",
            "clamp",
            "welder",
            "vision",
            "scanner",
            "table",
            "press",
            "furnace",
            "oven",
            "feeder",
            "spindle",
        ]
        INP = [
            "gas",
            "argon",
            "co2",
            "shielding",
            "adhesive",
            "epoxy",
            "glue",
            "filler",
            "wire",
            "flux",
            "powder",
            "rod",
            "solder",
            "base material",
            "workpiece",
            "sheet",
            "plate",
            "bar",
            "stock",
            "fastener",
            "bolt",
            "screw",
            "nut",
            "insert",
            "sealant",
            "primer",
        ]
        for t in tokens:
            low = t.lower()
            if any(k in low for k in PROC):
                processes.append(t)
            elif any(k in low for k in RES):
                resources.append(t)
            elif any(k in low for k in INP):
                inputs.append(t)
            else:
                products.append(t)
        return _normalize_ppr_safe(
            {
                "input_products": inputs,
                "products": products,
                "processes": processes,
                "resources": resources,
            }
        )

    # 2) Generate FMEA
    if st.button("Generate FMEA", key="fa_generate_onebox"):
        import time

        t0 = time.time()
        user_text = st.session_state["fa_user_text"]
        if not user_text or len(user_text.strip()) < 5:
            st.warning("Please enter a brief description.")
        else:
            query_ppr = _derive_ppr_from_text(user_text)

            sb = _build_supabase()
            with st.spinner("Retrieving relevant KB rows..."):
                t_kb0 = time.time()
                kb_rows = _select_kb_rows(
                    sb, embedder, query_ppr, top_cases=8, top_rows=30
                )
                st.session_state["fa_fmea_kb_ms"] = int((time.time() - t_kb0) * 1000)

            with st.spinner("Filling gaps with LLM..."):
                t_llm0 = time.time()
                llm_rows = _complete_missing_with_llm(kb_rows, query_ppr, llm)
                st.session_state["fa_fmea_llm_ms"] = int((time.time() - t_llm0) * 1000)

            merged = _normalize_numeric_and_rpn(kb_rows + llm_rows)

            st.session_state["proposed_rows"] = merged
            st.session_state["_provenance_vec"] = [r.get("_provenance", "kb") for r in merged]

            # Reset PPR
            st.session_state["assistant_ppr"] = _normalize_ppr_safe(
                {"input_products": [], "products": [], "processes": [], "resources": []}
            )

            # --- reset grid state for new generation ---
            st.session_state.pop("fa_grid_df", None)
            st.session_state.pop("fa_selected_rows", None)
            # ------------------------------------------

            st.session_state["fa_fmea_ms"] = int((time.time() - t0) * 1000)

            st.info(f"FMEA generated in {st.session_state['fa_fmea_ms']} ms.")
            st.success(
                f"Prepared {len(merged)} rows: "
                f"{sum(1 for r in merged if r.get('_provenance')=='kb')} from KB, "
                f"{sum(1 for r in merged if r.get('_provenance')=='llm')} from LLM."
            )

    # Input hints to avoid empty inputs
    def _extract_input_hints(text: str) -> list[str]:
        t = (text or "").lower()
        hints: list[str] = []
        add = lambda s: hints.append(s) if s not in hints else None
        if any(k in t for k in ["argon", "co2", "shielding gas", "shield gas"]):
            add("Argon shielding gas")
            add("CO2 shielding gas")
        if any(k in t for k in ["filler", "wire", "rod", "er4043", "er5356"]):
            add("Filler wire ER4043")
            add("Filler wire ER5356")
            add("Filler rod")
        if any(k in t for k in ["adhesive", "glue", "epoxy", "primer"]):
            add("Adhesive")
            add("Epoxy")
            add("Surface primer")
        if any(k in t for k in ["bolt", "screw", "nut", "fastener", "rivet"]):
            add("Fasteners")
            add("Bolts")
            add("Screws")
            add("Nuts")
            add("Rivets")
        if any(k in t for k in ["aluminium", "aluminum", "copper", "steel", "sheet", "profile", "wire"]):
            add("Base material")
        if any(k in t for k in ["clean", "ipa", "isopropyl", "solvent"]):
            add("Cleaning solvent (IPA)")
        return hints

    # 3) KB-style PPR generation (LLM-only)
    def _llm_ppr_same_as_kb(user_txt: str, rows_sample: list[dict]) -> dict:
        sample_rows = (rows_sample or [])[:10]
        input_hints = _extract_input_hints(user_txt)
        prompt = {
            "instruction": (
                "You are a manufacturing PPR extraction assistant. "
                "From the user description and the sample rows, produce four lists only: "
                "input_products, products (outputs), processes, resources. "
                "Extract four lists only: input_products, products (outputs), processes, resources. "
                "Treat Input Products as consumables and base materials fed into the process "
                "(e.g., aluminium extrusions/profiles, sheets/plates, filler wire/rod ER4043, "
                "shielding gas argon/CO2, adhesives/primers, fasteners). "
                "If consumables/base materials are implied, return at least 3 items in input_products "
                "using domain knowledge and the provided input_hints; avoid leaving input_products empty "
                "when evidence exists. Return concise, deduplicated strings. No explanations; only JSON keys with arrays."
            ),
            "user_description": (user_txt or "").strip(),
            "file_name": st.session_state.get("uploaded_file", ""),
            "sample_rows": sample_rows,
            "input_hints": input_hints,
        }
        payload = json.dumps(prompt, ensure_ascii=False)

        with st.expander("PPR generation debug (LLM KB-style)", expanded=False):
            st.write("Description preview (first 280 chars):")
            st.code((user_txt or "")[:280], language="text")
            st.write(f"Sample rows used: {len(sample_rows)} (of {len(rows_sample) if rows_sample else 0})")
            st.write("Payload to LLM (truncated 1,500 chars):")
            st.code(payload[:1500] + ("..." if len(payload) > 1500 else ""), language="json")

        try:
            ppr = llm.generate_ppr_from_text(context_text=payload, ppr_hint=None)
        except Exception as e:
            st.error(f"PPR LLM call failed: {e}")
            return {}

        with st.expander("Raw LLM return (repr)", expanded=False):
            try:
                # keep your structure; avoid NameError on _rows
                st.code(repr(ppr)[:4000], language="text")
            except Exception:
                st.write(ppr)

        ppr = ppr if isinstance(ppr, dict) else {}
        normalized = _normalize_ppr_safe(ppr)

        if not normalized.get("input_products") and input_hints:
            normalized["input_products"] = sorted({h for h in input_hints if h and h.strip()})[:5]

        with st.expander("Normalized PPR (LLM KB-style)", expanded=False):
            st.json(normalized)

        return normalized

    # 4) Review grid (FMEA rows)
    if "proposed_rows" in st.session_state:
        st.subheader("Review FMEA rows")

        if "fa_grid_df" not in st.session_state:
            _df0 = pd.DataFrame(st.session_state["proposed_rows"])
            if "_provenance_vec" in st.session_state and len(st.session_state["_provenance_vec"]) == len(_df0):
                _df0["_provenance"] = st.session_state["_provenance_vec"]
            else:
                if "_provenance" not in _df0.columns:
                    _df0["_provenance"] = "llm"

            if "_row_id" not in _df0.columns:
                _df0["_row_id"] = [str(uuid.uuid4()) for _ in range(len(_df0))]

            st.session_state["fa_grid_df"] = _df0

        df = st.session_state["fa_grid_df"].copy()

        c_del, c_add = st.columns([1, 1])
        with c_del:
            delete_clicked = st.button("Delete selected rows", key="fa_delete_rows")
        with c_add:
            add_clicked = st.button("Add new row", key="fa_add_row")

        if delete_clicked:
            selected = st.session_state.get("fa_selected_rows", None)

            if selected is None:
                selected = []
            elif isinstance(selected, pd.DataFrame):
                selected = selected.to_dict(orient="records")
            elif isinstance(selected, dict):
                selected = [selected]
            elif not isinstance(selected, list):
                selected = []

            selected_ids = {r.get("_row_id") for r in selected if isinstance(r, dict) and r.get("_row_id")}

            if selected_ids:
                st.session_state["fa_grid_df"] = (
                    st.session_state["fa_grid_df"][
                        ~st.session_state["fa_grid_df"]["_row_id"].isin(selected_ids)
                    ].reset_index(drop=True)
                )
                st.rerun()
            else:
                st.warning("No rows selected.")

        if add_clicked:
            # IMPORTANT: do NOT rerun here; just extend the df and let the same run render it.
            df_current = st.session_state["fa_grid_df"]
            blank = {c: None for c in df_current.columns}
            blank["_row_id"] = str(uuid.uuid4())
            blank["_provenance"] = "manual"  # default for manually added rows

            st.session_state["fa_grid_df"] = pd.concat(
                [df_current, pd.DataFrame([blank])],
                ignore_index=True,
            )

            # Update df used for rendering in this run (prevents odd UI state)
            df = st.session_state["fa_grid_df"].copy()

        df_grid = df.copy().astype(object).where(pd.notna(df), None)

        def _json_safe(v):
            if v is None or isinstance(v, (int, float, str, bool)):
                return v
            try:
                return str(v)
            except Exception:
                return None

        for c in df_grid.columns:
            df_grid[c] = df_grid[c].map(_json_safe)

        is_empty_col = df_grid.apply(
            lambda col: not pd.Series(col)
            .astype(str)
            .str.strip()
            .replace({"None": "", "nan": ""})
            .ne("")
            .any(),
            axis=0,
        )
        empty_cols = [c for c, e in is_empty_col.items() if e]

        with st.expander("Columns with no values", expanded=False):
            show_empty_cols = st.checkbox("Show empty columns", value=False, key="fa_show_empty_cols")
            st.write("Empty columns:", empty_cols)

        gb = GridOptionsBuilder.from_dataframe(df_grid)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True)

        if "_row_id" in df_grid.columns:
            gb.configure_column("_row_id", header_name="Row ID", filter=False, editable=False, hide=True)

        gb.configure_column("_provenance", header_name="Prov", filter=True, editable=False)

        for col in df_grid.columns:
            if col in ["_row_id", "_provenance"]:
                continue
            gb.configure_column(
                col,
                header_name=col.replace("_", " ").title(),
                filter=True,
                editable=True,
                hide=(col in empty_cols and not show_empty_cols),
            )

        gb.configure_selection(
            selection_mode="multiple",
            use_checkbox=True,
            header_checkbox=True,
            header_checkbox_filtered_only=True,
        )

        grid_options = gb.build()
        grid_options["rowClassRules"] = {
            "kb-row": "function(params) { return params && params.data && params.data._provenance === 'kb'; }",
            "llm-row": "function(params) { return params && params.data && params.data._provenance === 'llm'; }",
        }
        grid_options["domLayout"] = "normal"

        grid_response = AgGrid(
            df_grid,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,  # smoother for editing than MODEL_CHANGED
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            height=420,
            theme="ag-theme-alpine",
            custom_css=AGGRID_CUSTOM_CSS,
        )

        st.session_state["fa_selected_rows"] = list(grid_response.get("selected_rows", []) or [])

        st.markdown(
            """
            <style>
            .ag-theme-streamlit .kb-row .ag-cell { background-color: #e6f2ff !important; }
            .ag-theme-streamlit .llm-row .ag-cell { background-color: #fff9e6 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        edited_df = pd.DataFrame(grid_response["data"])

        def _sint(v):
            try:
                return int(str(v).strip())
            except Exception:
                return 0

        if all(c in edited_df.columns for c in ["s1", "o1", "d1"]):
            edited_df["rpn1"] = edited_df.apply(
                lambda r: _sint(r.get("s1")) * _sint(r.get("o1")) * _sint(r.get("d1")),
                axis=1,
            )
        if all(c in edited_df.columns for c in ["s2", "o2", "d2"]):
            edited_df["rpn2"] = edited_df.apply(
                lambda r: _sint(r.get("s2")) * _sint(r.get("o2")) * _sint(r.get("d2")),
                axis=1,
            )

        # Keep your existing sync approach (so Save/Export always see the latest)
        st.session_state["fa_grid_df"] = edited_df.copy()
        st.session_state["edited_df"] = edited_df

    # 4b) PPR editor + Generate PPR
    if "proposed_rows" in st.session_state:
        st.markdown("---")
        st.subheader("PPR")

        timelines = []
        kb_ms = st.session_state.get("fa_fmea_kb_ms")
        llm_ms = st.session_state.get("fa_fmea_llm_ms")
        tot_ms = st.session_state.get("fa_fmea_ms")
        if kb_ms is not None:
            timelines.append(f"KB: {kb_ms} ms")
        if llm_ms is not None:
            timelines.append(f"LLM: {llm_ms} ms")
        if tot_ms is not None:
            timelines.append(f"FMEA total: {tot_ms} ms")
        if st.session_state.get("fa_ppr_ms") is not None:
            timelines.append(f"PPR: {st.session_state['fa_ppr_ms']} ms")
        if timelines:
            st.caption(" | ".join(timelines))

        st.session_state["assistant_ppr"] = _normalize_ppr_safe(
            st.session_state.get(
                "assistant_ppr",
                {"input_products": [], "products": [], "processes": [], "resources": []},
            )
        )

        if st.button("Generate PPR", key="fa_generate_ppr", use_container_width=True):
            import time

            t0 = time.time()
            try:
                if "edited_df" in st.session_state and st.session_state["edited_df"] is not None:
                    rows_for_ppr = pd.DataFrame(st.session_state["edited_df"]).to_dict(orient="records")
                else:
                    rows_for_ppr = st.session_state.get("proposed_rows", [])
            except Exception:
                rows_for_ppr = st.session_state.get("proposed_rows", [])

            if not rows_for_ppr and not (user_text and user_text.strip()):
                st.warning(
                    "No FMEA rows available and description is empty. "
                    "Please generate FMEA or enter a description."
                )
            else:
                with st.spinner("Requesting PPR from LLM (KB-style)..."):
                    ppr_new = _llm_ppr_same_as_kb(user_text, rows_for_ppr)

                if any(ppr_new.values()):
                    st.session_state["assistant_ppr"] = _normalize_ppr_safe(ppr_new)
                    st.session_state["fa_ppr_ms"] = int((time.time() - t0) * 1000)
                    st.rerun()
                else:
                    st.error(
                        "LLM returned empty PPR. Check the debug expander above for payload and raw output."
                    )

        ppr_cur = _normalize_ppr_safe(
            st.session_state.get(
                "assistant_ppr",
                {"input_products": [], "products": [], "processes": [], "resources": []},
            )
        )
        edited_ppr = ppr_editor_block("fa_ppr", ppr_cur)
        st.session_state["assistant_ppr"] = _normalize_ppr_safe(edited_ppr)

        # 5) Save as test case
        st.markdown("### Save as test case")

        c1, c2 = st.columns([2, 3])
        with c1:
            case_title = st.text_input(
                "Case title",
                placeholder="Manual Aluminium airframe TIG welding.",
                key="fa_case_title",
            )
        with c2:
            default_case_desc = (
                st.session_state.get("fa_case_desc")
                or st.session_state.get("fa_user_text", "")
            )
            case_desc = st.text_area(
                "Case description",
                height=140,
                value=default_case_desc,
                placeholder=(
                    "Manual Aluminium Airframe TIG Welding involves joining aluminum airframe "
                    "components using TIG welding with suitable filler material and shielding gas. "
                    "The process includes preparation, setup, welding execution, post-weld treatments, "
                    "and inspection. Skilled operators utilize TIG welding and NDT equipment to ensure "
                    "high-quality, defect-free welds."
                ),
                key="fa_case_desc",
            )

        def _sanitize_rows_for_db_from_df(df_in: pd.DataFrame) -> list[dict]:
            df = df_in.copy()
            rename_map = {
                "systemelement": "system_element",
                "potentialfailure": "potential_failure",
                "potentialeffect": "potential_effect",
                "potentialcause": "potential_cause",
                "currentpreventiveaction": "current_preventive_action",
                "currentdetectionaction": "current_detection_action",
                "recommendedaction": "recommended_action",
                "actiontaken": "action_taken",
            }
            df.rename(
                columns={k: v for k, v in rename_map.items() if k in df.columns},
                inplace=True,
            )
            int_cols = ["s1", "o1", "d1", "rpn1", "s2", "o2", "d2", "rpn2"]
            for col in int_cols:
                if col in df.columns:

                    def to_int_or_none(x):
                        if x is None:
                            return None
                        sx = str(x).strip()
                        if sx == "" or sx.lower() == "nan":
                            return None
                        try:
                            return int(float(sx))
                        except Exception:
                            return None

                    df[col] = df[col].map(to_int_or_none)

            for col in df.columns:
                if col not in int_cols:

                    def to_str_or_none(x):
                        if x is None or (isinstance(x, float) and pd.isna(x)):
                            return None
                        sx = str(x).strip()
                        return None if sx.lower() == "nan" or sx == "" else sx

                    df[col] = df[col].map(to_str_or_none)

            allowed = [
                "system_element",
                "function",
                "potential_failure",
                "c1",
                "potential_effect",
                "s1",
                "c2",
                "c3",
                "potential_cause",
                "o1",
                "current_preventive_action",
                "current_detection_action",
                "d1",
                "rpn1",
                "recommended_action",
                "rd",
                "action_taken",
                "s2",
                "o2",
                "d2",
                "rpn2",
                "notes",
            ]
            for col in allowed:
                if col not in df.columns:
                    df[col] = None
            return df[allowed].to_dict(orient="records")

        def _validate_rows_before_save(df_in: pd.DataFrame) -> list[str]:
            if df_in is None or not isinstance(df_in, pd.DataFrame) or df_in.empty:
                return ["No FMEA rows available."]
            optional = {"_row_id", "_provenance"}  # case_id is not in grid; it is set on insert
            required_cols = [c for c in df_in.columns if c not in optional]

            errors = []
            for i, row in df_in.iterrows():
                missing = []
                for c in required_cols:
                    v = row.get(c)
                    if v is None:
                        missing.append(c)
                    else:
                        s = str(v).strip()
                        if s == "" or s.lower() == "nan":
                            missing.append(c)
                if missing:
                    rid = row.get("_row_id", i)
                    errors.append(f"Row {i+1} (id={rid}): missing {', '.join(missing)}")
            return errors

        if st.button("Save test case", key="fa_save_test_case"):
            if not case_title or not case_title.strip():
                st.error("Please enter a case title before saving.")
            elif not case_desc or not case_desc.strip():
                st.error("Please enter a case description before saving.")
            else:
                if not _get_secret("SUPABASE_URL") or not _get_secret("SUPABASE_ANON_KEY"):
                    st.error("SUPABASE_URL or SUPABASE_ANON_KEY not set.")
                    st.stop()

                # NEW: mandatory fields validation (prov optional; others required)
                if "edited_df" in st.session_state and isinstance(st.session_state["edited_df"], pd.DataFrame):
                    _errs = _validate_rows_before_save(st.session_state["edited_df"])
                    if _errs:
                        st.error("Please complete all mandatory fields in the FMEA table before saving.")
                        st.write(_errs[:30])
                        st.stop()

                try:
                    sb = _build_supabase()

                    # 1) Create case
                    title = case_title.strip()
                    desc = case_desc.strip()
                    case_resp = sb.table("cases").insert({"title": title, "description": desc}).execute()
                    case_id = case_resp.data[0]["id"]
                    st.session_state["last_saved_case_id"] = case_id

                    # 2) Insert FMEA rows
                    if "edited_df" not in st.session_state or st.session_state["edited_df"] is None:
                        raise ValueError("No edited FMEA rows available to save.")
                    fmea_rows_clean = _sanitize_rows_for_db_from_df(st.session_state["edited_df"])
                    for r in fmea_rows_clean:
                        r["case_id"] = case_id
                    if fmea_rows_clean:
                        sb.table("fmea_rows").insert(fmea_rows_clean).execute()
                    else:
                        st.warning("No FMEA rows to insert after sanitization.")

                    # 3) PPR tables and links
                    ppr = _normalize_ppr_safe(
                        st.session_state.get(
                            "assistant_ppr",
                            {"input_products": [], "products": [], "processes": [], "resources": []},
                        )
                    )
                    inputs_list = [x for x in ppr["input_products"] if x and x.strip()]
                    prods_list = [x for x in ppr["products"] if x and x.strip()]
                    procs_list = [x for x in ppr["processes"] if x and x.strip()]
                    ress_list = [x for x in ppr["resources"] if x and x.strip()]

                    inputs_list = sorted({x.strip() for x in inputs_list})
                    prods_list = sorted({x.strip() for x in prods_list})
                    procs_list = sorted({x.strip() for x in procs_list})
                    ress_list = sorted({x.strip() for x in ress_list})

                    def _get_or_create_ppr_local(sb, table, name):
                        name = (name or "").strip()
                        if not name:
                            return None
                        existing = (
                            sb.table(table).select("id").eq("name", name).limit(1).execute().data
                        )
                        if existing:
                            return existing[0]["id"]
                        rec = sb.table(table).insert({"name": name}).execute().data
                        return rec[0]["id"] if rec and isinstance(rec, list) else None

                    input_ids = ([_get_or_create_ppr_local(sb, "inputs", n) for n in inputs_list] if inputs_list else [])
                    prod_ids = [_get_or_create_ppr_local(sb, "products", n) for n in prods_list]
                    proc_ids = [_get_or_create_ppr_local(sb, "processes", n) for n in procs_list]
                    res_ids = [_get_or_create_ppr_local(sb, "resources", n) for n in ress_list]

                    def _link_case_ppr_local(sb, case_id, table, id_field, ids):
                        rows = [{"case_id": case_id, id_field: pid} for pid in ids if pid]
                        if rows:
                            sb.table(table).upsert(rows, on_conflict=f"case_id,{id_field}").execute()

                    if input_ids:
                        _link_case_ppr_local(sb, case_id, "case_inputs", "input_id", input_ids)
                    _link_case_ppr_local(sb, case_id, "case_products", "product_id", prod_ids)
                    _link_case_ppr_local(sb, case_id, "case_processes", "process_id", proc_ids)
                    _link_case_ppr_local(sb, case_id, "case_resources", "resource_id", res_ids)

                    def _upsert_case_scoped_ppr(sb, table: str, case_id: int, names: list[str], name_col: str = "name"):
                        for nm in names:
                            if not nm:
                                continue
                            exists = (
                                sb.table(table)
                                .select("id")
                                .eq("case_id", case_id)
                                .eq(name_col, nm)
                                .limit(1)
                                .execute()
                                .data
                                or []
                            )
                            if exists:
                                continue
                            try:
                                sb.table(table).insert({name_col: nm, "case_id": case_id}).execute()
                            except Exception:
                                try:
                                    sb.table(table).update({"case_id": case_id}).eq(name_col, nm).is_("case_id", "null").execute()
                                except Exception:
                                    pass

                    _upsert_case_scoped_ppr(sb, "inputs", case_id, inputs_list)
                    _upsert_case_scoped_ppr(sb, "products", case_id, prods_list)
                    _upsert_case_scoped_ppr(sb, "processes", case_id, procs_list)
                    _upsert_case_scoped_ppr(sb, "resources", case_id, ress_list)

                    # 4) RAG index (kb_index)
                    inputs_txt = ", ".join(inputs_list)
                    prod_txt = ", ".join(prods_list)
                    proc_txt = ", ".join(procs_list)
                    res_txt = ", ".join(ress_list)

                    inp_vec = _to_plain_list(embedder.embed(inputs_txt)) if inputs_txt else None
                    prod_vec = _to_plain_list(embedder.embed(prod_txt)) if prod_txt else None
                    proc_vec = _to_plain_list(embedder.embed(proc_txt)) if proc_txt else None
                    res_vec = _to_plain_list(embedder.embed(res_txt)) if res_txt else None

                    rec_full = {
                        "case_id": case_id,
                        "inputs_text": inputs_txt or None,
                        "products_text": prod_txt or None,
                        "processes_text": proc_txt or None,
                        "resources_text": res_txt or None,
                        "inp_vec": inp_vec,
                        "prod_vec": prod_vec,
                        "proc_vec": proc_vec,
                        "res_vec": res_vec,
                    }

                    import numpy as np

                    # HARD DEBUG: check for non-finite values before calling Supabase
                    for k in ["inp_vec", "prod_vec", "proc_vec", "res_vec"]:
                        v = rec_full[k]
                        if isinstance(v, list):
                            arr = np.array(v, dtype=float)
                            if not np.isfinite(arr).all():
                                print(">>> NON-FINITE in", k, "for case", case_id)
                                print("raw vector:", v[:10], "...")
                                arr[~np.isfinite(arr)] = 0.0
                                rec_full[k] = [float(x) for x in arr]

                    # Also verify JSON encoding right here
                    json.dumps(rec_full)

                    try:
                        sb.table("kb_index").upsert(rec_full, on_conflict="case_id").execute()
                    except Exception as e:
                        print(">>> kb_index upsert failed for case", case_id)
                        print("rec_full snippet:", {k: rec_full[k] for k in rec_full if k.endswith("_text")})
                        raise

                    st.success(f"Created test case #{case_id} with FMEA rows, PPR links, and kb_index.")

                except Exception as e:
                    st.error(f"Save failed: {e}")
                else:
                    st.session_state["fa_save_success_msg"] = f"Saved test case #{case_id}."
                    st.rerun()

        msg = st.session_state.pop("fa_save_success_msg", None)
        if msg:
            st.success(msg)

        # 6) Export (after Save)
        st.markdown("---")
        st.subheader("Export")

        case_id_for_export = st.session_state.get("last_saved_case_id")
        case_title_for_export = st.session_state.get("fa_case_title", "")
        case_desc_for_export = st.session_state.get("fa_case_desc", "")
        model_label = (
            LLM_REGISTRY[st.session_state["active_model_id"]]["label"]
            if "active_model_id" in st.session_state
            else ""
        )
        timing_fmea_ms = st.session_state.get("fa_fmea_ms")
        timing_ppr_ms = st.session_state.get("fa_ppr_ms")
        timing_fmea_kb_ms = st.session_state.get("fa_fmea_kb_ms")
        timing_fmea_llm_ms = st.session_state.get("fa_fmea_llm_ms")

        ppr_for_export = _normalize_ppr_safe(st.session_state.get("assistant_ppr") or {})
        edited = st.session_state.get("edited_df")
        if isinstance(edited, pd.DataFrame):
            fmea_df = edited.copy()
        elif isinstance(edited, list):
            fmea_df = pd.DataFrame(edited)
        else:
            fmea_df = pd.DataFrame()

        excel_bytes = to_pretty_excel_bytes(
            case_id=case_id_for_export,
            case_title=case_title_for_export,
            case_desc=case_desc_for_export,
            model_label=model_label,
            timing_fmea_ms=timing_fmea_ms,
            timing_fpr_ms_ppr=timing_ppr_ms,
            timing_fmea_kb_ms=timing_fmea_kb_ms,
            timing_fmea_llm_ms=timing_fmea_llm_ms,
            ppr=ppr_for_export,
            fmea_rows_df=fmea_df,
        )

        xml_bytes = to_structured_xml_bytes(
            case_id=case_id_for_export,
            case_title=case_title_for_export,
            case_desc=case_desc_for_export,
            model_label=model_label,
            timing_fmea_ms=timing_fmea_ms,
            timing_ppr_ms=timing_ppr_ms,
            timing_fmea_kb_ms=timing_fmea_kb_ms,
            timing_fmea_llm_ms=timing_fmea_llm_ms,
            ppr=ppr_for_export,
            fmea_rows=fmea_df.to_dict(orient="records"),
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name=f"fmea_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="fa_download_excel",
            )
        with c2:
            st.download_button(
                "Download XML",
                data=xml_bytes,
                file_name=f"fmea_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                mime="application/xml",
                key="fa_download_xml",
            )
