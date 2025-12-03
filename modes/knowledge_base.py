import time
import json
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from styles import AGGRID_CUSTOM_CSS
from backend.llm import LLM, LLM_REGISTRY
from backend.file_parser import parse_xml_fmea
from backend.backend_fmea_pipeline import process_excel_for_preview


def render_knowledge_base(embedder, helpers):
    """
    Knowledge Base (manual PPR; buttons; proper linking).

    embedder: Embeddings instance from app.py
    helpers: dict of helper functions from app.py
    """
    _guess_mime = helpers["_guess_mime"]
    _normalize_ppr_safe = helpers["_normalize_ppr_safe"]
    ppr_editor_block = helpers["ppr_editor_block"]
    _build_supabase = helpers["_build_supabase"]
    _supabase_bucket_name = helpers["_supabase_bucket_name"]
    _to_plain_list = helpers["_to_plain_list"]

    st.title("Knowledge Base Uploader")
    st.markdown(
        "Upload APIS based Excel, review FMEA, type in or optionally generate PPR, "
        "then save as a new case."
    )

    if "SUPABASE_URL" not in st.secrets or "SUPABASE_ANON_KEY" not in st.secrets:
        st.error("SUPABASE_URL or SUPABASE_ANON_KEY not set.")
        st.stop()

    # Session defaults
    for k, v in [
        ("parsed_fmea", None),
        ("parsed_ppr", None),
        ("uploaded_file", None),
        ("uploaded_bytes", None),
        ("uploaded_mime", None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    files = st.file_uploader(
        "Select a file", type=["xml", "xlsx", "xls"], accept_multiple_files=False
    )

    # Parse only; do not upload yet
    if files:
        upl = files
        try:
            data = upl.read()
            mime = _guess_mime(upl.name)
            if upl.name.lower().endswith((".xls", ".xlsx")):
                try:
                    fmea_rows = process_excel_for_preview(data)
                except Exception as e:
                    st.error(f"Excel parse failed: {e}")
                    fmea_rows = []
            elif upl.name.lower().endswith(".xml"):
                fmea_rows = parse_xml_fmea(data)
            else:
                fmea_rows = []

            if isinstance(fmea_rows, pd.DataFrame):
                fmea_rows = fmea_rows.to_dict(orient="records")
            elif not isinstance(fmea_rows, list):
                fmea_rows = []

            st.write(f"Parsed rows: {len(fmea_rows)} from {upl.name}")
            if not fmea_rows:
                st.error(f"No FMEA data extracted from {upl.name}.")
            else:
                st.session_state["parsed_fmea"] = fmea_rows
                st.session_state["uploaded_file"] = upl.name
                st.session_state["uploaded_bytes"] = data
                st.session_state["uploaded_mime"] = mime
                st.session_state["parsed_ppr"] = _normalize_ppr_safe(
                    st.session_state.get("parsed_ppr") or {}
                )
        except Exception as e:
            st.error(f"Error processing {upl.name}: {e}")

    # FMEA grid
    if st.session_state.get("parsed_fmea"):
        st.subheader(
            f"Review parsed FMEA - {st.session_state.get('uploaded_file', '')}"
        )
        df_preview = pd.DataFrame(st.session_state["parsed_fmea"])
        df_grid = df_preview.copy()

        is_empty_col = df_grid.apply(
            lambda col: not col.astype(str)
            .str.strip()
            .replace({"None": "", "nan": ""})
            .ne("")
            .any(),
            axis=0,
        )
        empty_cols = [c for c, e in is_empty_col.items() if e]
        with st.expander("Columns with no values", expanded=False):
            show_empty_cols = st.checkbox(
                "Show empty columns", value=False, key="kb_show_empty_cols"
            )
            st.write("Empty columns:", empty_cols)

        gb = GridOptionsBuilder.from_dataframe(df_grid)
        gb.configure_default_column(
            filterable=True, sortable=True, resizable=True, editable=True
        )
        for col_name in df_grid.columns:
            gb.configure_column(
                col_name,
                header_name=col_name.replace("_", " ").title(),
                filter=True,
                editable=True,
                hide=(col_name in empty_cols and not show_empty_cols),
            )
        grid_options = gb.build()

        grid_response = AgGrid(
            df_grid,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            height=400,
            theme="ag-theme-alpine",
            custom_css=AGGRID_CUSTOM_CSS,
        )
        edited_fmea_df = grid_response["data"]
        st.session_state["parsed_fmea"] = edited_fmea_df.to_dict(orient="records")

        # --- PPR generation from user description ---
        with st.expander("**Optionally auto-generate PPR**", expanded=False):
            desc = st.text_area(
                "Short description (what the file contains, processes, products, resources)",
                placeholder=(
                    "Example: Manual Aluminium Airframe TIG Welding involves joining "
                    "aluminum airframe components using TIG welding with suitable filler "
                    "material and shielding gas. The process includes preparation, setup, "
                    "welding execution, post-weld treatments, and inspection."
                ),
                height=80,
                key="kb_ppr_desc",
            )

            # --- LLM selector (KB) ---
            _kb_model_items = [(mid, cfg["label"]) for mid, cfg in LLM_REGISTRY.items()]

            if "active_model_id" not in st.session_state:
                _default_mid = (
                    "perplexity/sonar-pro"
                    if "perplexity/sonar-pro" in LLM_REGISTRY
                    else _kb_model_items[0][0]
                )
                st.session_state["active_model_id"] = _default_mid

            _kb_current_label = LLM_REGISTRY[st.session_state["active_model_id"]][
                "label"
            ]

            kb_c1, kb_c2 = st.columns([3, 2])
            with kb_c1:
                kb_selected_label = st.selectbox(
                    "Select LLM (KB)",
                    options=[label for _, label in _kb_model_items],
                    index=[label for _, label in _kb_model_items].index(
                        _kb_current_label
                    ),
                    key="kb_llm_select",
                )
            with kb_c2:
                st.caption("This model will be used for Knowledge Base PPR generation.")

            for mid, label in _kb_model_items:
                if label == kb_selected_label:
                    st.session_state["active_model_id"] = mid
                    break

            from os import getenv as _getenv

            _cfg_kb = LLM_REGISTRY[st.session_state["active_model_id"]]
            _api_ok_kb = bool(_getenv(_cfg_kb["env"], ""))
            st.markdown(
                f"<small>Using: <code>{_cfg_kb['label']}</code> · API key: "
                f"<span style='color:{'lime' if _api_ok_kb else 'tomato'}'>"
                f"{'OK' if _api_ok_kb else 'Missing'}</span></small>",
                unsafe_allow_html=True,
            )

            llm_kb = LLM(model_name=st.session_state["active_model_id"])
            llm_kb.set_model(st.session_state["active_model_id"])

            if st.button("Generate PPR", key="kb_generate_ppr_from_desc"):
                t0 = time.time()
                try:
                    sample_rows = (st.session_state.get("parsed_fmea") or [])[:10]
                    prompt = {
                        "instruction": (
                            "You are a manufacturing PPR extraction assistant. "
                            "From the user description and the sample rows, produce four lists only: "
                            "input_products, products (outputs), processes, resources. "
                            "Treat Input Products as consumables and base materials fed into the process. "
                            "Return concise, deduplicated strings; JSON only."
                        ),
                        "user_description": (desc or "").strip(),
                        "file_name": st.session_state.get("uploaded_file", ""),
                        "sample_rows": sample_rows,
                    }
                    payload = json.dumps(prompt, ensure_ascii=False)

                    ppr = llm_kb.generate_ppr_from_text(
                        context_text=payload,
                        ppr_hint=None,
                    )

                    st.session_state["parsed_ppr"] = _normalize_ppr_safe(
                        ppr if isinstance(ppr, dict) else {}
                    )
                    elapsed_ms = int((time.time() - t0) * 1000)
                    st.success(
                        f"PPR generated from description in {elapsed_ms} ms. "
                        "Review/edit below."
                    )
                except Exception as e:
                    st.error(f"PPR generation failed: {e}")

        # --- PPR editor (4-pillar) ---
        st.subheader("Review/Edit PPR (mandatory)")
        st.info(
            "Enter comma-separated values. Example — Input Products: aluminium tube, "
            "shielding gas; Output Products: welded bottle; Processes: ultrasonic "
            "welding; Resources: welding gun"
        )

        current_ppr = _normalize_ppr_safe(st.session_state.get("parsed_ppr"))
        st.session_state["parsed_ppr"] = ppr_editor_block("kb_ppr", current_ppr)

        # Guard: require at least one list populated
        pp = _normalize_ppr_safe(st.session_state.get("parsed_ppr"))
        inputs_list = [x for x in pp["input_products"] if x and x.strip()]
        prods_list = [x for x in pp["products"] if x and x.strip()]
        procs_list = [x for x in pp["processes"] if x and x.strip()]
        ress_list = [x for x in pp["resources"] if x and x.strip()]

        if not any([inputs_list, prods_list, procs_list, ress_list]):
            st.warning(
                "Please add at least one Input/Product/Process/Resource before saving."
            )

        # --- New case details ---
        st.markdown("### New case details")
        case_title = st.text_input(
            "Case title",
            value="",
            placeholder="e.g., Ultrasonic welding of battery tray v1",
        )
        case_desc = st.text_area(
            "Case description",
            height=80,
            value="",
            placeholder="e.g., Imported from supplier FMEA; QA reviewed",
        )

        # Save new case, then FMEA, PPR, kb_index
        if st.button("Save as New Case", key="kb_save_new_case"):
            try:
                data = st.session_state.get("uploaded_bytes", None)
                name = st.session_state.get("uploaded_file", None)
                mime = st.session_state.get(
                    "uploaded_mime", "application/octet-stream"
                )
                if not data or not name:
                    st.error(
                        "Source file not available in session. Please re-upload."
                    )
                    st.stop()

                if not any([inputs_list, prods_list, procs_list, ress_list]):
                    st.error(
                        "Enter at least one Input/Product/Process/Resource before saving."
                    )
                    st.stop()

                sb = _build_supabase()
                bucket = _supabase_bucket_name()

                # 1) Upload RAW file
                path = f"{int(time.time())}_{name}"
                with st.spinner(f"Saving RAW file to Supabase: {path}"):
                    sb.storage.from_(bucket).upload(
                        path, data, {"content-type": mime}
                    )
                    try:
                        sb.schema("public").table("kb_files").insert(
                            {
                                "name": name,
                                "mime": mime,
                                "size_bytes": len(data),
                                "path": path,
                            }
                        ).execute()
                    except Exception as e:
                        st.warning(
                            f"KB metadata insert skipped/failed: {e}"
                        )

                # 2) Create case
                if not case_title.strip():
                    st.error("Please enter a case title.")
                    st.stop()
                elif not case_desc.strip():
                    st.error("Please enter a case description.")
                    st.stop()

                case_resp = (
                    sb.table("cases")
                    .insert(
                        {
                            "title": case_title.strip(),
                            "description": case_desc.strip(),
                        }
                    )
                    .execute()
                )
                case_id = case_resp.data[0]["id"]

                # 3) Insert FMEA rows
                def _sanitize_rows_for_db(rows):
                    df = pd.DataFrame(rows)
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
                    df = df.rename(
                        columns={
                            k: v for k, v in rename_map.items() if k in df.columns
                        }
                    )
                    int_cols = [
                        "s1",
                        "o1",
                        "d1",
                        "rpn1",
                        "s2",
                        "o2",
                        "d2",
                        "rpn2",
                    ]
                    for col in int_cols:
                        if col in df.columns:

