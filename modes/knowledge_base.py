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

    # ---- helpers coming from app.py ----
    _guess_mime = helpers["_guess_mime"]
    _normalize_ppr_safe = helpers["_normalize_ppr_safe"]
    ppr_editor_block = helpers["ppr_editor_block"]
    _build_supabase = helpers["_build_supabase"]
    _supabase_bucket_name = helpers["_supabase_bucket_name"]
    _to_plain_list = helpers["_to_plain_list"]

    st.title("Knowledge Base Uploader")
    st.markdown("Upload APIS based Excel, review FMEA, type in or optionally generate PPR, then save as a new case.")

    if "SUPABASE_URL" not in st.secrets or "SUPABASE_ANON_KEY" not in st.secrets:
        st.error("SUPABASE_URL or SUPABASE_ANON_KEY not set.")
        st.stop()

    # Session defaults
    for k, v in [
        ("parsed_fmea", None), ("parsed_ppr", None),
        ("uploaded_file", None), ("uploaded_bytes", None), ("uploaded_mime", None)
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    files = st.file_uploader("Select a file", type=["xml", "xlsx", "xls"], accept_multiple_files=False)

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
                st.session_state["parsed_ppr"] = _normalize_ppr_safe(st.session_state.get("parsed_ppr") or {})
        except Exception as e:
            st.error(f"Error processing {upl.name}: {e}")

    # FMEA grid
    if st.session_state.get("parsed_fmea"):
        st.subheader(f"Review parsed FMEA - {st.session_state.get('uploaded_file', '')}")
        df_preview = pd.DataFrame(st.session_state["parsed_fmea"])
        df_grid = df_preview.copy()

        is_empty_col = df_grid.apply(
            lambda col: not col.astype(str).str.strip().replace({"None": "", "nan": ""}).ne("").any(), axis=0
        )
        empty_cols = [c for c, e in is_empty_col.items() if e]
        with st.expander("Columns with no values", expanded=False):
            show_empty_cols = st.checkbox("Show empty columns", value=False, key="kb_show_empty_cols")
            st.write("Empty columns:", empty_cols)

        gb = GridOptionsBuilder.from_dataframe(df_grid)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True, editable=True)
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
            theme="ag-theme-alpine",  # <--- add this for consistent styling
            custom_css=AGGRID_CUSTOM_CSS,
        )
        edited_fmea_df = grid_response["data"]
        st.session_state["parsed_fmea"] = edited_fmea_df.to_dict(orient="records")


        # --- PPR editor (4-pillar) ---
        # --- PPR generation from user description ---
        with st.expander("**Optionally auto-generate PPR**", expanded=False):
            desc = st.text_area(
                "Short description (what the file contains, processes, products, resources)",
                placeholder=(
                    "Example: Manual Aluminium Airframe TIG Welding involves joining aluminum airframe components "
                    "using TIG welding with suitable filler material and shielding gas. The process includes "
                    "preparation, setup, welding execution, post-weld treatments, and inspection. Skilled operators "
                    "utilize TIG welding and NDT equipment to ensure high-quality, defect-free welds."
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

            _kb_current_label = LLM_REGISTRY[st.session_state["active_model_id"]]["label"]

            kb_c1, kb_c2 = st.columns([3, 2])
            with kb_c1:
                kb_selected_label = st.selectbox(
                    "Select LLM (KB)",
                    options=[label for _, label in _kb_model_items],
                    index=[label for _, label in _kb_model_items].index(_kb_current_label),
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

            # Create client for this run and set selected model
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
                            "Treat Input Products as consumables and base materials fed into the process "
                            "(e.g., aluminium extrusions/profiles, sheets/plates, filler wire/rod ER4043, "
                            "shielding gas argon/CO2, adhesives/primers, fasteners). "
                            "Do not leave input_products empty if such items are present. "
                            "Return concise, deduplicated strings. No explanations, just JSON keys with arrays. "
                            "Do not invent; be concise, deduplicate, and use manufacturing terms."
                        ),
                        "user_description": (desc or "").strip(),
                        "file_name": st.session_state.get("uploaded_file", ""),
                        "sample_rows": sample_rows,
                    }
                    payload = json.dumps(prompt, ensure_ascii=False)

                    # PPR-only LLM call
                    ppr = llm_kb.generate_ppr_from_text(
                        context_text=payload,
                        ppr_hint=None,
                    )

                    st.session_state["parsed_ppr"] = _normalize_ppr_safe(
                        ppr if isinstance(ppr, dict) else {}
                    )
                    elapsed_ms = int((time.time() - t0) * 1000)
                    st.success(
                        f"PPR generated from description in {elapsed_ms} ms. Review/edit below."
                    )
                except Exception as e:
                    st.error(f"PPR generation failed: {e}")

        # --- PPR editor (4-pillar) ---
        st.subheader("Review/Edit PPR (mandatory)")
        st.info(
            "Enter comma-separated values. Example — Input Products: aluminium tube, shielding gas; "
            "Output Products: welded bottle; Processes: ultrasonic welding; Resources: welding gun"
        )

        current_ppr = _normalize_ppr_safe(st.session_state.get("parsed_ppr"))
        st.session_state["parsed_ppr"] = ppr_editor_block("kb_ppr", current_ppr)




        # Guard: require at least one list populated
        # Re-read the latest editor values just-in-time
        pp = _normalize_ppr_safe(st.session_state.get("parsed_ppr"))
        inputs_list = [x for x in pp["input_products"] if x and x.strip()]
        prods_list  = [x for x in pp["products"]       if x and x.strip()]
        procs_list  = [x for x in pp["processes"]      if x and x.strip()]
        ress_list   = [x for x in pp["resources"]      if x and x.strip()]


        if not any([inputs_list, prods_list, procs_list, ress_list]):
            st.warning("Please add at least one Input/Product/Process/Resource before saving.")

        # --- New case details ---
        st.markdown("### New case details")
        case_title = st.text_input("Case title", value="", placeholder="e.g., Ultrasonic welding of battery tray v1")
        case_desc  = st.text_area("Case description", height=80, value="", placeholder="e.g., Imported from supplier FMEA; QA reviewed")

        # Save new case, then FMEA, PPR, kb_index
        if st.button("Save as New Case", key="kb_save_new_case"):
            try:
                data = st.session_state.get("uploaded_bytes", None)
                name = st.session_state.get("uploaded_file", None)
                mime = st.session_state.get("uploaded_mime", "application/octet-stream")
                if not data or not name:
                    st.error("Source file not available in session. Please re-upload.")
                    st.stop()

                if not any([inputs_list, prods_list, procs_list, ress_list]):
                    st.error("Enter at least one Input/Product/Process/Resource before saving.")
                    st.stop()

                sb = _build_supabase()
                bucket = _supabase_bucket_name()

                # 1) Upload RAW file
                path = f"{int(time.time())}_{name}"
                with st.spinner(f"Saving RAW file to Supabase: {path}"):
                    sb.storage.from_(bucket).upload(path, data, {"content-type": mime})
                    try:
                        sb.schema("public").table("kb_files").insert({
                            "name": name, "mime": mime, "size_bytes": len(data), "path": path
                        }).execute()
                    except Exception as e:
                        st.warning(f"KB metadata insert skipped/failed: {e}")

                # 2) Create case
                if not case_title.strip():
                    st.error("Please enter a case title.")
                    st.stop()
                elif not case_desc.strip():
                    st.error("Please enter a case description.")
                    st.stop()

                case_resp = sb.table("cases").insert({
                    "title": case_title.strip(),
                    "description": case_desc.strip()
                }).execute()
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
                    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                    int_cols = ["s1","o1","d1","rpn1","s2","o2","d2","rpn2"]
                    for col in int_cols:
                        if col in df.columns:
                            def to_int_or_none(x):
                                if x is None: return None
                                sx=str(x).strip()
                                if sx=="" or sx.lower()=="nan": return None
                                try: return int(float(sx))
                                except: return None
                            df[col]=df[col].map(to_int_or_none)
                    for col in df.columns:
                        if col not in int_cols:
                            def to_str_or_none(x):
                                if x is None or (isinstance(x,float) and pd.isna(x)): return None
                                sx=str(x).strip()
                                return None if sx.lower()=="nan" or sx=="" else sx
                            df[col]=df[col].map(to_str_or_none)
                    allowed = [
                        "system_element","function","potential_failure","c1",
                        "potential_effect","s1","c2","c3",
                        "potential_cause","o1","current_preventive_action",
                        "current_detection_action","d1","rpn1",
                        "recommended_action","rd","action_taken",
                        "s2","o2","d2","rpn2","notes"
                    ]
                    for col in allowed:
                        if col not in df.columns: df[col]=None
                    df = df[allowed]
                    return df.to_dict(orient="records")

                fmea_rows_clean = _sanitize_rows_for_db(st.session_state["parsed_fmea"])
                for r in fmea_rows_clean: r["case_id"] = case_id
                if fmea_rows_clean:
                    sb.table("fmea_rows").insert(fmea_rows_clean).execute()
                else:
                    st.warning("No FMEA rows to insert after sanitization.")

                # 4) PPR catalogs + case links (includes inputs)
                def _get_or_create_ppr_local(sb, table, name):
                    name = (name or "").strip()
                    if not name: return None
                    existing = sb.table(table).select("id").eq("name", name).limit(1).execute().data
                    if existing: return existing[0]["id"]
                    rec = sb.table(table).insert({"name": name}).execute().data
                    return rec[0]["id"] if rec and isinstance(rec, list) else None

                # De-dup lists
                inputs_list = sorted({x.strip() for x in inputs_list if x and x.strip()})
                prods_list  = sorted({x.strip() for x in prods_list  if x and x.strip()})
                procs_list  = sorted({x.strip() for x in procs_list  if x and x.strip()})
                ress_list   = sorted({x.strip() for x in ress_list   if x and x.strip()})

                # Catalog ids
                input_ids = [_get_or_create_ppr_local(sb, "inputs", n)    for n in inputs_list] if inputs_list else []
                prod_ids  = [_get_or_create_ppr_local(sb, "products", n)  for n in prods_list]
                proc_ids  = [_get_or_create_ppr_local(sb, "processes", n) for n in procs_list]
                res_ids   = [_get_or_create_ppr_local(sb, "resources", n) for n in ress_list]

                def _link_case_ppr_local(sb, case_id, table, id_field, ids):
                    rows = [{"case_id": case_id, id_field: pid} for pid in ids if pid]
                    if rows:
                        sb.table(table).upsert(rows, on_conflict=f"case_id,{id_field}").execute()

                # Join links
                if input_ids:
                    _link_case_ppr_local(sb, case_id, "case_inputs", "input_id", input_ids)
                _link_case_ppr_local(sb, case_id, "case_products",  "product_id",  prod_ids)
                _link_case_ppr_local(sb, case_id, "case_processes", "process_id",  proc_ids)
                _link_case_ppr_local(sb, case_id, "case_resources", "resource_id", res_ids)

                # 5) Optional: per-case ownership in base tables (idempotent)
                def _upsert_case_scoped_ppr(sb, table: str, case_id: int, names: list[str], name_col="name"):
                    for nm in names:
                        if not nm:
                            continue
                        exists = sb.table(table).select("id").eq("case_id", case_id).eq(name_col, nm).limit(1).execute().data or []
                        if exists:
                            continue
                        try:
                            sb.table(table).insert({name_col: nm, "case_id": case_id}).execute()
                        except Exception:
                            try:
                                sb.table(table).update({"case_id": case_id}).eq(name_col, nm).is_("case_id", "null").execute()
                            except Exception:
                                pass

                _upsert_case_scoped_ppr(sb, "inputs",    case_id, inputs_list)
                _upsert_case_scoped_ppr(sb, "products",  case_id, prods_list)
                _upsert_case_scoped_ppr(sb, "processes", case_id, procs_list)
                _upsert_case_scoped_ppr(sb, "resources", case_id, ress_list)

                # 5) RAG (kb_index) with inputs_text + inp_vec
                inputs_txt = ", ".join(inputs_list)
                prod_txt   = ", ".join(prods_list)
                proc_txt   = ", ".join(procs_list)
                res_txt    = ", ".join(ress_list)


                inp_vec  = _to_plain_list(embedder.embed(inputs_txt)) if inputs_txt else None
                prod_vec = _to_plain_list(embedder.embed(prod_txt))  if prod_txt   else None
                proc_vec = _to_plain_list(embedder.embed(proc_txt))  if proc_txt   else None
                res_vec  = _to_plain_list(embedder.embed(res_txt))   if res_txt    else None

                rec_full = {
                    "case_id": case_id,
                    "inputs_text": inputs_txt or None,
                    "products_text": prod_txt or None,
                    "processes_text": proc_txt or None,
                    "resources_text": res_txt or None,
                    "inp_vec": inp_vec,
                    "prod_vec": prod_vec,
                    "proc_vec": proc_vec,
                    "res_vec":  res_vec,
                }

                try:
                    sb.table("kb_index").upsert(rec_full, on_conflict="case_id").execute()
                except Exception as e:
                    st.warning(f"kb_index upsert (full) failed: {e}")
                    # Minimal legacy fallback if new columns not present
                    rec_legacy = {
                        "case_id": case_id,
                        "products_text": prod_txt or None,
                        "processes_text": proc_txt or None,
                        "resources_text": res_txt or None,
                        "prod_vec": prod_vec,
                        "proc_vec": proc_vec,
                        "res_vec":  res_vec,
                    }
                    sb.table("kb_index").upsert(rec_legacy, on_conflict="case_id").execute()
                st.success(f"Created case #{case_id} with RAW file, FMEA rows, PPR links (including inputs), and kb_index.")
            except Exception as e:
                st.error(f"Save failed: {e}")
