import os
import time
import io
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# Backend imports
from backend.db import init_and_seed, get_conn
from backend.llm import Embeddings, LLM
from backend.repository import get_all_ontology, search_ontology_by_type
from backend.retrieval import map_text_to_ppr, retrieve_candidates
from backend.reuse import propose_fmea_rows
from backend.retain import persist_case
from backend.export import to_csv, to_excel_bytes
from backend.generate import generate_fmea_from_llm
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Google Drive uploader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload


st.set_page_config(page_title="CBR FMEA Assistant", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["FMEA Assistant", "Knowledge Base"], index=0)

@st.cache_resource
def bootstrap():
    init_and_seed()
    embedder = Embeddings()
    llm = LLM(model_name="sonar-pro")
    return embedder, llm

embedder, llm = bootstrap()

# Common helpers
def find_id_by_name(items, name):
    for it in items:
        if it["name"] == name:
            return it["id"]
    return None

APIS_COLUMNS = [
    "system_element", "function", "potential_failure", "c1",
    "potential_effect", "s1", "c2", "c3",
    "potential_cause", "o1", "current_preventive_action",
    "current_detection_action", "d1", "rpn1",
    "recommended_action", "rd", "action_taken",
    "s2", "o2", "d2", "rpn2", "notes"
]

def map_legacy_row_to_apis(r, ctrl_name=None):
    return {
        "system_element": r.get("system_element",""),
        "function": r.get("function",""),
        "potential_failure": r.get("potential_failure") or r.get("failure_mode",""),
        "c1": r.get("c1",""),
        "potential_effect": r.get("potential_effect") or r.get("effect",""),
        "s1": r.get("s1") or r.get("severity",""),
        "c2": r.get("c2",""),
        "c3": r.get("c3",""),
        "potential_cause": r.get("potential_cause") or r.get("cause",""),
        "o1": r.get("o1") or r.get("occurrence",""),
        "current_preventive_action": r.get("current_preventive_action") or (ctrl_name or r.get("control","")),
        "current_detection_action": r.get("current_detection_action",""),
        "d1": r.get("d1") or r.get("detection",""),
        "rpn1": r.get("rpn1",""),
        "recommended_action": r.get("recommended_action",""),
        "rd": r.get("rd",""),
        "action_taken": r.get("action_taken",""),
        "s2": r.get("s2",""),
        "o2": r.get("o2",""),
        "d2": r.get("d2",""),
        "rpn2": r.get("rpn2",""),
        "notes": r.get("notes",""),
    }

def normalize_to_apis(rows):
    out = []
    for row in rows or []:
        mapped = map_legacy_row_to_apis(row)
        mapped = {k: mapped.get(k, "") for k in APIS_COLUMNS}
        out.append(mapped)
    return out

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

# Google Drive helpers
def _build_drive_service():
    creds_path = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
    if not creds_path:
        raise RuntimeError("GDRIVE_SERVICE_ACCOUNT_JSON env var not set")
    scopes = ["https://www.googleapis.com/auth/drive.file"]
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
    return build("drive", "v3", credentials=creds)

def _guess_mime(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".xml"):
        return "application/xml"
    if fn.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if fn.endswith(".xls"):
        return "application/vnd.ms-excel"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".csv"):
        return "text/csv"
    if fn.endswith(".json"):
        return "application/json"
    return "application/octet-stream"

def upload_to_drive(file_bytes: bytes, filename: str, mime_type: str, folder_id: str) -> str:
    service = _build_drive_service()
    file_metadata = {"name": filename, "parents": [folder_id]}
    media = MediaInMemoryUpload(file_bytes, mimetype=mime_type, resumable=False)
    created = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return created.get("id")

# Page routing
if mode == "FMEA Assistant":
    st.title("Case-based FMEA Reuse (PPR + CBR)")

    with get_conn() as conn:
        ontology = get_all_ontology(conn)
        products = search_ontology_by_type(conn, "Product")
        processes = search_ontology_by_type(conn, "Process")
        resources = search_ontology_by_type(conn, "Resource")

    st.subheader("1) Describe your production case")
    user_text = st.text_area(
        "Describe the product, process, and resources",
        height=120,
        placeholder="Example: Aluminium profile welding with laser welding using shielding gas for chassis components."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_product = st.selectbox("Select Product class (optional)", ["Auto-map"]+[p["name"] for p in products])
    with col2:
        sel_process = st.selectbox("Select Process class (optional)", ["Auto-map"]+[p["name"] for p in processes])
    with col3:
        sel_resource = st.selectbox("Select Resource class (optional)", ["Auto-map"]+[p["name"] for p in resources])

    st.subheader("Optional: Auto-generate 10–15 FMEA rows with LLM")
    if st.button("Generate via LLM (auto-map + insert + open for review)"):
        if not user_text or len(user_text.strip()) < 10:
            st.warning("Please enter a meaningful description first.")
        else:
            with st.spinner("Calling LLM to generate FMEA rows, validating, and saving as a new case..."):
                with get_conn() as conn:
                    try:
                        case_id, saved_rows = generate_fmea_from_llm(
                            conn=conn,
                            embedder=embedder,
                            llm=llm,
                            user_text=user_text.strip(),
                            topk=3,
                            case_title="LLM-generated case"
                        )
                        st.success(f"LLM-generated case saved: #{case_id}. Loaded into editor below.")
                        expanded = []
                        for r in saved_rows:
                            if r.get("control_ids") and r.get("controls"):
                                for ctrl_name in r.get("controls", []):
                                    expanded.append(map_legacy_row_to_apis(r, ctrl_name=ctrl_name))
                            else:
                                expanded.append(map_legacy_row_to_apis(r, ctrl_name=None))
                        st.session_state["proposed_rows"] = normalize_to_apis(expanded)
                        st.write("DEBUG normalized LLM rows sample:", st.session_state["proposed_rows"][:3])
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")

    if st.button("Retrieve and Propose FMEA", type="primary"):
        with st.spinner("Mapping to PPR classes and retrieving candidates..."):
            with get_conn() as conn:
                mapped = map_text_to_ppr(user_text or "", ontology, embedder, topk=3)
                if sel_product != "Auto-map":
                    mapped["Product"] = [find_id_by_name(products, sel_product)]
                if sel_process != "Auto-map":
                    mapped["Process"] = [find_id_by_name(processes, sel_process)]
                if sel_resource != "Auto-map":
                    mapped["Resource"] = [find_id_by_name(resources, sel_resource)]

                cands = retrieve_candidates(conn, mapped)
                rows = propose_fmea_rows(conn, cands, ontology, mapped)

                st.session_state["mapped"] = mapped
                st.session_state["proposed_rows"] = normalize_to_apis(rows)
                st.write("DEBUG normalized retrieved rows sample:", st.session_state["proposed_rows"][:3])

        st.info(f"Mapped Product IDs: {mapped.get('Product')}, Process IDs: {mapped.get('Process')}, Resource IDs: {mapped.get('Resource')}")

    if "proposed_rows" in st.session_state:
        st.write("DEBUG: proposed_rows content", st.session_state["proposed_rows"])
        st.subheader("2) Review proposed FMEA with filters")

        df = pd.DataFrame(st.session_state["proposed_rows"])
        st.write("DEBUG dataframe head:", df.head())

        column_defs = [
            {"headerName": "System element", "field": "system_element"},
            {"headerName": "Function", "field": "function"},
            {"headerName": "Potential failure", "field": "potential_failure"},
            {"headerName": "C", "field": "c1"},
            {"headerName": "Potential effect(s) of failure", "field": "potential_effect"},
            {"headerName": "S", "field": "s1"},
            {"headerName": "C", "field": "c2"},
            {"headerName": "C", "field": "c3"},
            {"headerName": "Potential cause(s) of failure", "field": "potential_cause"},
            {"headerName": "O", "field": "o1"},
            {"headerName": "Current preventive action", "field": "current_preventive_action"},
            {"headerName": "Current detection action", "field": "current_detection_action"},
            {"headerName": "D", "field": "d1"},
            {"headerName": "RPN", "field": "rpn1"},
            {"headerName": "Recommended action", "field": "recommended_action"},
            {"headerName": "R/D", "field": "rd"},
            {"headerName": "Action taken", "field": "action_taken"},
            {"headerName": "S", "field": "s2"},
            {"headerName": "O", "field": "o2"},
            {"headerName": "D", "field": "d2"},
            {"headerName": "RPN", "field": "rpn2"},
        ]

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True)
        for col in column_defs:
            gb.configure_column(col["field"], header_name=col["headerName"], filter=True, editable=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            height=400,
        )

        edited_df = grid_response["data"]

        edited_df["rpn1"] = edited_df.apply(
            lambda r: _safe_int(r.get("s1")) * _safe_int(r.get("o1")) * _safe_int(r.get("d1")), axis=1
        )
        edited_df["rpn2"] = edited_df.apply(
            lambda r: _safe_int(r.get("s2")) * _safe_int(r.get("o2")) * _safe_int(r.get("d2")), axis=1
        )

        apis_columns = [
            "system_element", "function", "potential_failure", "c1",
            "potential_effect", "s1", "c2", "c3",
            "potential_cause", "o1",
            "current_preventive_action", "current_detection_action", "d1", "rpn1",
            "recommended_action", "rd", "action_taken", "s2", "o2", "d2", "rpn2"
        ]

        apis_columns_available = [col for col in apis_columns if col in edited_df.columns]
        edited_df_apis = edited_df[apis_columns_available]

        st.write("Preview with APIS columns:")
        st.dataframe(edited_df_apis, use_container_width=True)

        st.session_state["edited_df"] = edited_df_apis

        st.subheader("3) Export or Retain")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                "Download CSV",
                data=to_csv(st.session_state["edited_df"]),
                file_name="fmea.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                "Download Excel",
                data=to_excel_bytes(st.session_state["edited_df"]),
                file_name="fmea.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with c3:
            title = st.text_input("Case title", value="New Production Case")
            description = st.text_area("Case description", height=80, value=user_text or "")
            if st.button("Save to Case Base"):
                def pick_one_id(val):
                    if val is None:
                        return None
                    if isinstance(val, list):
                        return val[0] if val else None
                    if isinstance(val, (int,)):
                        return val
                    try:
                        return int(val)
                    except Exception:
                        return None

                with get_conn() as conn:
                    mapped = st.session_state.get("mapped", {})
                    product_id = pick_one_id(mapped.get("Product"))
                    process_id = pick_one_id(mapped.get("Process"))
                    resource_id = pick_one_id(mapped.get("Resource"))

                    if sel_product != "Auto-map":
                        product_id = pick_one_id(find_id_by_name(products, sel_product))
                    if sel_process != "Auto-map":
                        process_id = pick_one_id(find_id_by_name(processes, sel_process))
                    if sel_resource != "Auto-map":
                        resource_id = pick_one_id(find_id_by_name(resources, sel_resource))

                    for name, v in [("Product", product_id), ("Process", process_id), ("Resource", resource_id)]:
                        if v is not None and not isinstance(v, int):
                            st.error(f"{name} ID must be an integer or None; got {type(v)} = {v}.")
                            st.stop()

                    rows_to_save = []
                    for _, r in st.session_state["edited_df"].iterrows():
                        rows_to_save.append({
                            "failure_mode_id": r.get("failure_mode_id"),
                            "cause_id": r.get("cause_id"),
                            "effect_id": r.get("effect_id"),
                            "control_id": r.get("control_id"),
                            # Map APIS to DB schema S/O/D
                            "severity": r.get("s1"),
                            "occurrence": r.get("o1"),
                            "detection": r.get("d1"),
                            "notes": r.get("notes", "")
                        })
                    case_id = persist_case(conn, title, description, product_id, process_id, resource_id, rows_to_save)
                    st.success(f"Saved case #{case_id} to case base")

elif mode == "Knowledge Base":
    st.title("Knowledge Base Uploader")
    st.markdown("Upload XML, Excel, PDF, CSV, or JSON files to your Google Drive knowledge base folder.")

    gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if not gdrive_folder_id:
        st.error("GDRIVE_FOLDER_ID env var not set. Please set it to your Drive folder ID.")
        st.stop()

    files = st.file_uploader(
        "Select files",
        type=["xml", "xlsx", "xls", "pdf", "csv", "json"],
        accept_multiple_files=True
    )

    if files:
        for upl in files:
            try:
                data = upl.read()
                mime = _guess_mime(upl.name)
                with st.spinner(f"Uploading {upl.name} to Drive..."):
                    file_id = upload_to_drive(data, upl.name, mime, gdrive_folder_id)
                st.success(f"Uploaded {upl.name} to Drive (fileId={file_id}).")
                rec = {
                    "name": upl.name,
                    "size_bytes": len(data),
                    "mime": mime,
                    "drive_id": file_id,
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.setdefault("kb_manifest", []).append(rec)
            except Exception as e:
                st.error(f"Failed to upload {upl.name}: {e}")

    if st.session_state.get("kb_manifest"):
        st.subheader("Uploaded this session")
        st.dataframe(pd.DataFrame(st.session_state["kb_manifest"]))
    else:
        st.info("No files uploaded yet.")
