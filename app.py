# app.py
# Streamlit app integrating:
# - Knowledge Base (KB) page: Upload XML/Excel → parse FMEA → auto-generate PPR → edit → save by manual case_id
#   - Save persists FMEA rows, global PPR catalogs (deduped), case-to-PPR links, and kb_index vectors
# - FMEA Assistant: KB-first retrieval + LLM gap-fill with provenance coloring
import os
import time
import json
import pandas as pd
import streamlit as st
import datetime as dt
st.set_page_config(page_title="CBR FMEA Assistant", layout="wide")
from backend.export import to_pretty_excel_bytes, to_structured_xml_bytes
from supabase import create_client, Client
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Backend utilities (project-specific; keep names as in your repo)
#from backend.db import init_and_seed, get_conn
from backend.llm import Embeddings, LLM, LLM_REGISTRY
#from backend.repository import get_all_ontology, search_ontology_by_type
from backend.file_parser import parse_xml_fmea
from backend.backend_fmea_pipeline import (
    process_excel_for_preview,
    apply_enhancement,       # returns enhanced rows + PPR (3 keys)
    generate_ppr_only,       # returns PPR with "input_products" + classic keys
)


# ---------- Global styling ----------
st.markdown(
    """
    <style>
    /* Global page background and content width */
    .stApp {
        background: radial-gradient(circle at top left, #e0f2fe 0, #f4f3ed 55%, #e5e7eb 100%);
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Typography */
    body, h1, h2, h3, h4, h5, h6, p {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #111827;
    }
    h1 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    h2 {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

        /* Primary buttons (Generate FMEA, etc.) */
    .stButton > button {
        background: linear-gradient(135deg, #0f766e, #22c55e);
        color: #ffffff;
        border: none;
        border-radius: 999px !important; /* ensure pill */
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        font-size: 0.92rem;
        box-shadow: 0 6px 18px rgba(15, 118, 110, 0.35);
        cursor: pointer;
        transition: background-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
        }
    .stButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.45);
    }

    /* Make sure primary and secondary kinds both use our colors, not plain white */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0f766e, #22c55e);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        filter: brightness(1.06);
    }

    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background: #e5f2ff;              /* light tint to avoid plain white */
        color: #0f172a;
        border-radius: 999px;
        border: 1px solid #cbd5f5;
        box-shadow: none;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #dbeafe;
        border-color: #0f766e;
    }

    /* Card-like containers (unchanged) */
    .block-container > div {
        border-radius: 16px;
    }

    /* Tabs: spacing from top + bold labels so they read clearly as tabs */
    [data-testid="stTabs"] {
        margin-top: 1.5rem;      /* space below Streamlit header bar */
        margin-bottom: 1.5rem;
    }
    [data-testid="stTabs"] button[role="tab"] {
        font-weight: 700;        /* bold tab labels */
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }

        /* Make tab labels bold (including emojis) */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-weight: 700 !important;
    }

    /* Optional: slightly darker color for the active tab */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"]
        [data-testid="stMarkdownContainer"] p {
        color: #0f172a;
    }


    /* AG-Grid refinements (as before) */
    .ag-theme-alpine .ag-header {
        background-color: #0f172a !important;
    }
    .ag-theme-alpine .ag-header-cell,
    .ag-theme-alpine .ag-header-cell-label {
        color: #f9fafb !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .ag-theme-alpine .ag-cell {
        font-size: 14px !important;
        white-space: nowrap;       /* single line only */
        overflow: hidden;          /* hide overflow if any */
        text-overflow: ellipsis;   /* show ... if column is still too narrow */
        font-size: 14px !important;
    }
    .ag-theme-alpine .ag-row-hover {
        background-color: #ecfeff !important;
    }

        /* Streamlit dataframes (st.dataframe) */
    [data-testid="stDataFrame"] > div {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;   /* light grey table card */
    }
    [data-testid="stDataFrame"] table {
        font-size: 0.9rem;
    }
    [data-testid="stDataFrame"] thead tr {
        background-color: #0f172a;   /* dark header */
        color: #f9fafb;
        font-weight: 600;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f3f4f6;   /* zebra striping */
    }
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #e0f2fe;   /* hover row */
    }

    /* AgGrid overall card and rows */
    .ag-theme-alpine {
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        background-color: #f9fafb;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .ag-theme-alpine .ag-row:nth-child(even) {
        background-color: #f3f4f6;
    }
    .ag-theme-alpine .ag-row:hover {
        background-color: #e0f2fe !important;
    }

        /* Streamlit dataframes (st.dataframe) */
    div[data-testid="stDataFrame"] > div {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.9rem;
    }
    div[data-testid="stDataFrame"] thead tr {
        background-color: #0f172a;
        color: #f9fafb;
        font-weight: 600;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f3f4f6;
    }
    div[data-testid="stDataFrame"] tbody tr:hover {
        background-color: #e0f2fe;
    }


    /* Description textarea */
    .stTextArea textarea {
        border: 1px solid #d4d4d8 !important;
        border-radius: 14px !important;
        background-color: #f9fafb !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus-visible {
        outline: none !important;
        border: 1px solid #0f766e !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.4);
        background-color: #ffffff !important;
    }

        /* Text inputs (Case title, etc.) */
    .stTextInput input {
        border: 1px solid #d4d4d8 !important;
        border-radius: 999px !important;
        background-color: #f9fafb !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    .stTextInput input:focus-visible {
        outline: none !important;
        border: 1px solid #0f766e !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.4);
        background-color: #ffffff !important;
    }


    /* "Select LLM" selectboxes – softer fill, no harsh white */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 999px !important;
        border: 1px solid #d4d4d8 !important;
        background: linear-gradient(135deg, #eef2ff, #f9fafb) !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        padding: 2px 8px;
    }

    /* Remove any inner white blocks so the whole pill looks consistent */
    .stSelectbox > div[data-baseweb="select"] > div {
        background-color: transparent !important;
    }

    /* Hover / focus accent in your teal theme */
    .stSelectbox > div[data-baseweb="select"]:hover,
    .stSelectbox > div[data-baseweb="select"]:focus-within {
        border-color: #0f766e !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.3);
    }

    .stSelectbox > div[data-baseweb="select"] > div {
    border-radius: 999px !important; /* force inner div to pill */
    }
    
    .stSelectbox [data-baseweb="select"] div[role="combobox"] {
    border-radius: 999px !important;
    background-color: transparent !important;
    }
    
    .stSelectbox [data-baseweb="tag"] {
    border-radius: 999px !important;
    }

    /* ---------- TIGHT PILL / CARD FOR INPUTS & SELECTS ---------- */

    /* Text inputs (Case title, etc.) – remove inner white band */
    .stTextInput > div > div {
        border-radius: 999px !important;
        background-color: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    .stTextInput input {
        border: 1px solid #d4d4d8 !important;
        border-radius: 999px !important;
        background-color: #f9fafb !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    
    /* Textareas (description, PPR fields) – remove outer white frame */
    .stTextArea > div {
        border-radius: 14px !important;
        background-color: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    .stTextArea textarea {
        border: 1px solid #d4d4d8 !important;
        border-radius: 14px !important;
        background-color: #f9fafb !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    
    /* Select LLM – remove inner white band & tighten control */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 999px !important;
        border: 1px solid #d4d4d8 !important;
        background: linear-gradient(135deg, #eef2ff, #f9fafb) !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        padding: 2px 8px !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] {
        margin: 0 !important;
        padding: 4px 10px !important;
        border-radius: 999px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] > div {
        background-color: transparent !important;
        border-radius: 999px !important;
    }
    
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        background-color: transparent !important;
    }

    /* ---- FIX: Case title input pill ---- */
    /* Remove extra inner white band */
    .stTextInput:has(input[placeholder^="e.g., Ultrasonic welding of battery tray v1"]) > div > div {
        background-color: transparent !important;
        padding: 0 !important;
        border-radius: 999px !important;
        box-shadow: none !important;
    }
    
    .stTextInput:has(input[placeholder^="e.g., Ultrasonic welding of battery tray v1"]) input {
        border: 1px solid #d4d4d8 !important;
        border-radius: 999px !important;
        background-color: #f9fafb !important;
        padding: 0.6rem 1rem !important;
    }
    
    /* ---- FIX: Select LLM pill ---- */
    /* Outer pill container */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 999px !important;
        border: 1px solid #0f766e !important;
        background: linear-gradient(135deg, #eef2ff, #f9fafb) !important;
        padding: 0 !important;
    }
    
    /* Inner combobox (removes white band) */
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] {
        margin: 2px !important;
        padding: 6px 12px !important;
        border-radius: 999px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] > div {
        background-color: transparent !important;
        border-radius: 999px !important;
    }



    </style>
    """,
    unsafe_allow_html=True,
)
# ---------- end styling ----------

AGGRID_CUSTOM_CSS = {
    # Overall grid background + border
    ".ag-root-wrapper": {
        "border-radius": "14px",
        "border": "1px solid #e5e7eb",
        "box-shadow": "0 4px 14px rgba(15, 23, 42, 0.06)",
        "overflow": "hidden",
        "background-color": "#f9fafb",
    },
    # Header row
    ".ag-header": {
        "background-color": "#0f172a",
        "color": "#f9fafb",
        "font-weight": "600",
        "font-size": "14px",
    },
    ".ag-header-cell-label": {
        "color": "#f9fafb",
        "font-weight": "600",
        "font-size": "14px",
    },
    # Body rows: zebra and hover
    ".ag-row:nth-child(even)": {
        "background-color": "#f3f4f6",
    },
    ".ag-row-hover": {
        "background-color": "#e0f2fe !important",
    },
}

# -----------------------
# Env and helpers
# -----------------------

@st.cache_resource
def get_embedder():
    return Embeddings()

embedder = get_embedder()


def _get_secret(name: str, default: str | None = None) -> str | None:
    # Try st.secrets first (Streamlit), then fall back to environment variables for local use
    return st.secrets.get(name) or os.getenv(name, default)

def _build_supabase() -> Client:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")

    if not url or not key:
        st.error("SUPABASE_URL or SUPABASE_ANON_KEY not set.")
        st.stop()

    return create_client(url, key)


def _supabase_bucket_name() -> str:
    return st.secrets.get("SUPABASE_BUCKET", "kb-files")


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

@st.cache_resource
def bootstrap():
    embedder = Embeddings()
    llm = LLM(model_name="sonar-pro")
    return embedder, llm

embedder, llm = bootstrap()

test_vec = embedder.embed("test")
#st.write("Embedding length:", len(test_vec))

# -----------------------
# PPR helpers (4-pillar)
# -----------------------
def _coerce_ppr_keys(ppr: dict) -> dict:
    if not isinstance(ppr, dict):
        return {}
    p = dict(ppr)
    if "inputs" in p and "input_products" not in p:
        p["input_products"] = p.get("inputs", [])
    if "input" in p and "input_products" not in p:
        p["input_products"] = p.get("input", [])
    if "output_products" in p and "products" not in p:
        p["products"] = p.get("output_products", [])
    return p

def _normalize_ppr(ppr: dict) -> dict:
    return {
        "input_products": list((ppr or {}).get("input_products", [])) or [],
        "products": list((ppr or {}).get("products", [])) or [],
        "processes": list((ppr or {}).get("processes", [])) or [],
        "resources": list((ppr or {}).get("resources", [])) or [],
    }

def _normalize_ppr_safe(ppr: dict) -> dict:
    return _normalize_ppr(_coerce_ppr_keys(ppr or {}))

def _csv_to_list(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def ppr_editor_block(key_prefix: str, ppr: dict) -> dict:
    """4-column editor: Input Products, Output Products, Processes, Resources."""
    ppr = _normalize_ppr_safe(ppr)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_inputs = st.text_area(
            "Input Products",
            ", ".join(ppr["input_products"]),
            key=f"{key_prefix}_input_products",
            placeholder="e.g., aluminium tube, shielding gas, filler rod",
            height=120,
        )
    with c2:
        t_products = st.text_area(
            "Output Products",
            ", ".join(ppr["products"]),
            key=f"{key_prefix}_products",
            placeholder="e.g., welded aluminium airframe, finished bottle",
            height=120,
        )
    with c3:
        t_processes = st.text_area(
            "Processes",
            ", ".join(ppr["processes"]),
            key=f"{key_prefix}_processes",
            placeholder="e.g., TIG welding, setup, inspection, post‑weld treatment",
            height=120,
        )
    with c4:
        t_resources = st.text_area(
            "Resources",
            ", ".join(ppr["resources"]),
            key=f"{key_prefix}_resources",
            placeholder="e.g., TIG welding machine, fixtures, NDT equipment",
            height=120,
        )
    return {
        "input_products": _csv_to_list(t_inputs),
        "products": _csv_to_list(t_products),
        "processes": _csv_to_list(t_processes),
        "resources": _csv_to_list(t_resources),
    }

def ppr_to_cytoscape(ppr: dict):
    """Convert 4-pillar PPR into Cytoscape elements with simple layered edges."""
    ppr = _normalize_ppr_safe(ppr)
    elements = []
    for i, name in enumerate(ppr["input_products"]):
        elements.append({"data": {"id": f"inp_{i}", "label": name, "kind": "Input"}})
    for i, name in enumerate(ppr["processes"]):
        elements.append({"data": {"id": f"pro_{i}", "label": name, "kind": "Process"}})
    for i, name in enumerate(ppr["resources"]):
        elements.append({"data": {"id": f"res_{i}", "label": name, "kind": "Resource"}})
    for i, name in enumerate(ppr["products"]):
        elements.append({"data": {"id": f"out_{i}", "label": name, "kind": "Output"}})
    # Layered edges
    for i in range(len(ppr["input_products"])):
        for j in range(len(ppr["processes"])):
            elements.append({"data": {"source": f"inp_{i}", "target": f"pro_{j}"}})
    for j in range(len(ppr["processes"])):
        for r in range(len(ppr["resources"])):
            elements.append({"data": {"source": f"pro_{j}", "target": f"res_{r}"}})
    for r in range(len(ppr["resources"])):
        for o in range(len(ppr["products"])):
            elements.append({"data": {"source": f"res_{r}", "target": f"out_{o}"}})
    stylesheet = [
        {"selector": 'node[kind = "Input"]', "style": {"background-color": "#7faaff", "label": "data(label)"}},
        {"selector": 'node[kind = "Process"]', "style": {"background-color": "#b55fa1", "label": "data(label)"}},
        {"selector": 'node[kind = "Resource"]', "style": {"background-color": "#888888", "label": "data(label)"}},
        {"selector": 'node[kind = "Output"]', "style": {"background-color": "#2793ff", "label": "data(label)"}},
        {"selector": "edge", "style": {"line-color": "#cfcfcf", "curve-style": "bezier",
                                        "target-arrow-shape": "triangle", "target-arrow-color": "#cfcfcf"}},
    ]
    return elements, stylesheet


# -----------------------
# Case-based KB retrieval utilities (unchanged functionality)
# -----------------------
def _concat_list(v):
    if not v: return ""
    return ", ".join([str(x).strip() for x in v]) if isinstance(v, list) else str(v).strip()

def _to_plain_list(v):
    """
    Convert an embedding (NumPy array or list) to a plain Python list of finite floats.
    Any NaN / Inf / -Inf values are removed so JSON encoding is always valid.
    """
    if v is None:
        return None
    try:
        import numpy as np
        arr = np.array(v, dtype=float).ravel()
        # Replace non-finite with 0.0, then filter if you prefer
        arr[~np.isfinite(arr)] = 0.0
        clean = [float(x) for x in arr]
        return clean or None
    except Exception:
        return None

                    
def _fetch_candidate_cases(sb: Client, embedder: Embeddings, query_ppr: dict, top_k: int = 10) -> list:
    idx = sb.table("kb_index").select("*").execute().data or []

    import json, numpy as np
    def _to_vec(v):
        if v is None:
            return None
        if isinstance(v, str):
            try: return json.loads(v)
            except Exception: return None
        return v

    q_inp = _concat_list(query_ppr.get("input_products"))
    q_prod = _concat_list(query_ppr.get("products"))
    q_proc = _concat_list(query_ppr.get("processes"))
    q_res  = _concat_list(query_ppr.get("resources"))

    qv_inp  = embedder.embed(q_inp)  if q_inp  else None
    qv_prod = embedder.embed(q_prod) if q_prod else None
    qv_proc = embedder.embed(q_proc) if q_proc else None
    qv_res  = embedder.embed(q_res)  if q_res  else None

    def cos(a, b):
        if a is None or b is None:
            return 0.0
        try:
            va = np.asarray(a, dtype=float); vb = np.asarray(b, dtype=float)
            if va.ndim != 1 or vb.ndim != 1: return 0.0
            d = min(va.size, vb.size); va = va[:d]; vb = vb[:d]
            na = np.linalg.norm(va); nb = np.linalg.norm(vb)
            if na == 0.0 or nb == 0.0: return 0.0
            return float(np.dot(va, vb) / (na * nb))
        except Exception:
            return 0.0

    ranked = []
    for r in idx:
        iv = _to_vec(r.get("inp_vec"))
        pv = _to_vec(r.get("prod_vec"))
        pr = _to_vec(r.get("proc_vec"))
        rs = _to_vec(r.get("res_vec"))
        s_inp  = cos(qv_inp,  iv) if qv_inp  and iv else 0.0
        s_prod = cos(qv_prod, pv) if qv_prod and pv else 0.0
        s_proc = cos(qv_proc, pr) if qv_proc and pr else 0.0
        s_res  = cos(qv_res,  rs) if qv_res  and rs else 0.0
        # Weight inputs more when present; tune as you like
        score = 0.35*s_inp + 0.35*s_prod + 0.2*s_proc + 0.1*s_res
        ranked.append({"case_id": r["case_id"], "score": score})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def _fetch_rows_for_cases(sb: Client, case_ids: list[int]) -> list[dict]:
    out=[]
    for cid in case_ids:
        rows = sb.table("fmea_rows").select("*").eq("case_id", cid).execute().data or []
        for r in rows: out.append(r)
    return out

def _score_row_against_query(row: dict, query_ppr: dict) -> float:
    import re
    def toks(s): return set(re.findall(r"[A-Za-z0-9]+", (s or "").lower()))
    q_tokens = toks(_concat_list(query_ppr.get("products"))+" "+_concat_list(query_ppr.get("processes"))+" "+_concat_list(query_ppr.get("resources")))
    row_txt = " ".join([
        str(row.get("system_element","")), str(row.get("function","")), str(row.get("potential_failure","")),
        str(row.get("potential_effect","")), str(row.get("potential_cause","")),
        str(row.get("current_preventive_action","")), str(row.get("current_detection_action","")),
        str(row.get("recommended_action","")), str(row.get("notes","")),
    ])
    r_tokens = toks(row_txt)
    inter = len(q_tokens & r_tokens)
    return inter/(1+len(q_tokens))

def _select_kb_rows(sb: Client, embedder: Embeddings, query_ppr: dict, top_cases=8, top_rows=30) -> list[dict]:
    cand_cases = _fetch_candidate_cases(sb, embedder, query_ppr, top_k=top_cases)
    case_ids = [c["case_id"] for c in cand_cases]
    rows = _fetch_rows_for_cases(sb, case_ids)
    scored = [{"row": r, "score": _score_row_against_query(r, query_ppr)} for r in rows]
    scored.sort(key=lambda x: x["score"], reverse=True)
    picked = [s["row"] for s in scored[:top_rows]]
    for r in picked:
        r["_provenance"]="kb"
        for k in ["id","created_at"]:
            r.pop(k, None)
    return picked

def _complete_missing_with_llm(
    kb_rows: list[dict],
    query_ppr: dict,
    llm: LLM,
) -> list[dict]:
    """
    Ask the LLM only for extra FMEA rows (no PPR),
    based on KB rows + derived PPR from the description.
    Returned rows are tagged with _provenance='llm'.
    """
    import json

    prompt = {
        "instruction": (
            "Given the user PPR description and the retrieved KB FMEA rows, "
            "propose ADDITIONAL FMEA rows that fill coverage gaps. "
            "Avoid duplicates of the KB rows. Use this exact schema for each row: "
            "system_element,function,potential_failure,c1,"
            "potential_effect,s1,c2,c3,"
            "potential_cause,o1,current_preventive_action,"
            "current_detection_action,d1,rpn1,"
            "recommended_action,rd,action_taken,"
            "s2,o2,d2,rpn2,notes."
        ),
        "query_ppr": query_ppr,
        "kb_rows": kb_rows,
    }

    try:
        ctx = json.dumps(prompt, ensure_ascii=False)
        # FMEA‑only call (uses your existing generate_fmea_rows_json)
        gen_rows = llm.generate_fmea_rows_json(context_text=ctx, ppr_hint=query_ppr)
    except Exception:
        return []

    out: list[dict] = []
    if isinstance(gen_rows, list):
        for r in gen_rows:
            row = dict(r)
            row["_provenance"] = "llm"
            out.append(row)
    return out


def _normalize_numeric_and_rpn(rows: list[dict]) -> list[dict]:
    out=[]
    for r in rows:
        rr=dict(r)
        for k in ["s1","o1","d1","s2","o2","d2","rpn1","rpn2"]:
            v=rr.get(k)
            try: rr[k]=int(str(v).strip())
            except: rr[k]=None
        s1,o1,d1=rr.get("s1"),rr.get("o1"),rr.get("d1")
        rr["rpn1"]=s1*o1*d1 if all(isinstance(x,int) for x in [s1,o1,d1]) else None
        s2,o2,d2=rr.get("s2"),rr.get("o2"),rr.get("d2")
        rr["rpn2"]=s2*o2*d2 if all(isinstance(x,int) for x in [s2,o2,d2]) else None
        out.append(rr)
    return out

# PPR catalog + linking helpers
def _get_or_create_ppr(sb: Client, table: str, name: str):
    name = (name or "").strip()
    if not name:
        return None
    existing = sb.table(table).select("id").eq("name", name).limit(1).execute().data
    if existing:
        return existing[0]["id"]
    rec = sb.table(table).insert({"name": name}).execute().data
    return rec[0]["id"] if rec and isinstance(rec, list) else None

def _link_case_ppr(sb: Client, case_id: int, table: str, id_field: str, ids: list[int]):
    rows = [{"case_id": case_id, id_field: pid} for pid in ids if pid]
    if not rows:
        return
    sb.table(table).upsert(rows, on_conflict=f"case_id,{id_field}").execute()

# -----------------------
# Top navigation via tabs (works on all recent Streamlit versions)
# -----------------------
tab_fmea, tab_kb, tab_cases = st.tabs(
    ["🤖 FMEA Assistant", "📚 Knowledge Base", "🗂️ Cases Explorer"]
)

# -----------------------
# Knowledge Base (manual PPR; buttons; proper linking)
# -----------------------
with tab_kb:
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
                import time
                t0 = time.time()
                try:
                    sample_rows = (st.session_state.get("parsed_fmea") or [])[:10]
                    prompt = {
                        "instruction": (
                            "You are a manufacturing PPR extraction assistant. "
                            "From the user description and the sample rows, produce four lists only: "
                            "input_products, products (outputs), processes, resources. "
                            "Extract four lists only: input_products, products (outputs), processes, resources. "
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

                    _rows, ppr = llm_kb.generate_fmea_and_ppr_json(
                        context_text=payload, ppr_hint=None
                    )

                    # Normalize and keep
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



# -----------------------
# FMEA Assistant (FMEA first; PPR editor visible after; KB-style PPR gen with input hints)
# -----------------------
with tab_fmea:
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
                st.session_state["fa_fmea_kb_ms"] = int(
                    (time.time() - t_kb0) * 1000
                )

            with st.spinner("Filling gaps with LLM..."):
                t_llm0 = time.time()
                llm_rows = _complete_missing_with_llm(kb_rows, query_ppr, llm)
                st.session_state["fa_fmea_llm_ms"] = int(
                    (time.time() - t_llm0) * 1000
                )

            merged = _normalize_numeric_and_rpn(kb_rows + llm_rows)

            st.session_state["proposed_rows"] = merged
            st.session_state["_provenance_vec"] = [
                r.get("_provenance", "kb") for r in merged
            ]

            # Reset PPR
            st.session_state["assistant_ppr"] = _normalize_ppr_safe(
                {
                    "input_products": [],
                    "products": [],
                    "processes": [],
                    "resources": [],
                }
            )

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
            st.write(
                f"Sample rows used: {len(sample_rows)} (of {len(rows_sample) if rows_sample else 0})"
            )
            st.write("Payload to LLM (truncated 1,500 chars):")
            st.code(
                payload[:1500] + ("..." if len(payload) > 1500 else ""),
                language="json",
            )

        try:
            _rows, ppr = llm.generate_fmea_and_ppr_json(
                context_text=payload, ppr_hint=None
            )
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return {}

        with st.expander("Raw LLM return (repr)", expanded=False):
            try:
                st.code(repr((_rows, ppr))[:4000], language="text")
            except Exception:
                st.write(ppr)

        ppr = ppr if isinstance(ppr, dict) else {}
        normalized = _normalize_ppr_safe(ppr)

        if not normalized.get("input_products") and input_hints:
            normalized["input_products"] = sorted(
                {h for h in input_hints if h and h.strip()}
            )[:5]

        with st.expander("Normalized PPR (LLM KB-style)", expanded=False):
            st.json(normalized)

        return normalized

    # 4) Review grid (FMEA rows)
    if "proposed_rows" in st.session_state:
        st.subheader("Review and export")
        df = pd.DataFrame(st.session_state["proposed_rows"])
        if "_provenance_vec" in st.session_state and len(
            st.session_state["_provenance_vec"]
        ) == len(df):
            df["_provenance"] = st.session_state["_provenance_vec"]
        else:
            df["_provenance"] = "llm"

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
            show_empty_cols = st.checkbox(
                "Show empty columns", value=False, key="fa_show_empty_cols"
            )
            st.write("Empty columns:", empty_cols)

        gb = GridOptionsBuilder.from_dataframe(df_grid)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True)
        gb.configure_column("_provenance", header_name="Prov", filter=True, editable=False)
        for col in df_grid.columns:
            gb.configure_column(
                col,
                header_name=col.replace("_", " ").title(),
                filter=True,
                editable=True,
                hide=(col in empty_cols and not show_empty_cols),
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
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            height=420,
            theme="ag-theme-alpine",
            custom_css=AGGRID_CUSTOM_CSS,
        )

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
                lambda r: _sint(r.get("s1"))
                * _sint(r.get("o1"))
                * _sint(r.get("d1")),
                axis=1,
            )
        if all(c in edited_df.columns for c in ["s2", "o2", "d2"]):
            edited_df["rpn2"] = edited_df.apply(
                lambda r: _sint(r.get("s2"))
                * _sint(r.get("o2"))
                * _sint(r.get("d2")),
                axis=1,
            )

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
                {
                    "input_products": [],
                    "products": [],
                    "processes": [],
                    "resources": [],
                },
            )
        )

        if st.button("Generate PPR", key="fa_generate_ppr", use_container_width=True):
            import time

            t0 = time.time()
            try:
                if (
                    "edited_df" in st.session_state
                    and st.session_state["edited_df"] is not None
                ):
                    rows_for_ppr = pd.DataFrame(
                        st.session_state["edited_df"]
                    ).to_dict(orient="records")
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
                    st.session_state["fa_ppr_ms"] = int(
                        (time.time() - t0) * 1000
                    )
                    st.rerun()
                else:
                    st.error(
                        "LLM returned empty PPR. Check the debug expander above for payload and raw output."
                    )

        ppr_cur = _normalize_ppr_safe(
            st.session_state.get(
                "assistant_ppr",
                {
                    "input_products": [],
                    "products": [],
                    "processes": [],
                    "resources": [],
                },
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

        if st.button("Save test case", key="fa_save_test_case"):
            if not case_title or not case_title.strip():
                st.error("Please enter a case title before saving.")
            elif not case_desc or not case_desc.strip():
                st.error("Please enter a case description before saving.")
            else:
                if not _get_secret("SUPABASE_URL") or not _get_secret(
                    "SUPABASE_ANON_KEY"
                ):
                    st.error("SUPABASE_URL or SUPABASE_ANON_KEY not set.")
                    st.stop()
                try:
                    sb = _build_supabase()

                    # 1) Create case
                    title = case_title.strip()
                    desc = case_desc.strip()
                    case_resp = (
                        sb.table("cases")
                        .insert({"title": title, "description": desc})
                        .execute()
                    )
                    case_id = case_resp.data[0]["id"]
                    st.session_state["last_saved_case_id"] = case_id

                    # 2) Insert FMEA rows
                    if (
                        "edited_df" not in st.session_state
                        or st.session_state["edited_df"] is None
                    ):
                        raise ValueError("No edited FMEA rows available to save.")
                    fmea_rows_clean = _sanitize_rows_for_db_from_df(
                        st.session_state["edited_df"]
                    )
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
                            {
                                "input_products": [],
                                "products": [],
                                "processes": [],
                                "resources": [],
                            },
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
                            sb.table(table)
                            .select("id")
                            .eq("name", name)
                            .limit(1)
                            .execute()
                            .data
                        )
                        if existing:
                            return existing[0]["id"]
                        rec = sb.table(table).insert({"name": name}).execute().data
                        return rec[0]["id"] if rec and isinstance(rec, list) else None

                    input_ids = (
                        [
                            _get_or_create_ppr_local(sb, "inputs", n)
                            for n in inputs_list
                        ]
                        if inputs_list
                        else []
                    )
                    prod_ids = [
                        _get_or_create_ppr_local(sb, "products", n) for n in prods_list
                    ]
                    proc_ids = [
                        _get_or_create_ppr_local(sb, "processes", n) for n in procs_list
                    ]
                    res_ids = [
                        _get_or_create_ppr_local(sb, "resources", n) for n in ress_list
                    ]

                    def _link_case_ppr_local(sb, case_id, table, id_field, ids):
                        rows = [
                            {"case_id": case_id, id_field: pid}
                            for pid in ids
                            if pid
                        ]
                        if rows:
                            sb.table(table).upsert(
                                rows, on_conflict=f"case_id,{id_field}"
                            ).execute()

                    if input_ids:
                        _link_case_ppr_local(
                            sb, case_id, "case_inputs", "input_id", input_ids
                        )
                    _link_case_ppr_local(
                        sb, case_id, "case_products", "product_id", prod_ids
                    )
                    _link_case_ppr_local(
                        sb, case_id, "case_processes", "process_id", proc_ids
                    )
                    _link_case_ppr_local(
                        sb, case_id, "case_resources", "resource_id", res_ids
                    )

                    def _upsert_case_scoped_ppr(
                        sb, table: str, case_id: int, names: list[str], name_col: str = "name"
                    ):
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
                                sb.table(table).insert(
                                    {name_col: nm, "case_id": case_id}
                                ).execute()
                            except Exception:
                                try:
                                    sb.table(table).update({"case_id": case_id}).eq(
                                        name_col, nm
                                    ).is_("case_id", "null").execute()
                                except Exception:
                                    pass

                    _upsert_case_scoped_ppr(sb, "inputs", case_id, inputs_list)
                    _upsert_case_scoped_ppr(sb, "products", case_id, prods_list)
                    _upsert_case_scoped_ppr(sb, "processes", case_id, procs_list)
                    _upsert_case_scoped_ppr(sb, "resources", case_id, ress_list)

                    # 4) RAG index (kb_index)
                    inputs_txt = ", ".join(inputs_list)
                    prod_txt   = ", ".join(prods_list)
                    proc_txt   = ", ".join(procs_list)
                    res_txt    = ", ".join(ress_list)

                    inp_vec  = _to_plain_list(embedder.embed(inputs_txt)) if inputs_txt else None
                    prod_vec = _to_plain_list(embedder.embed(prod_txt))   if prod_txt   else None
                    proc_vec = _to_plain_list(embedder.embed(proc_txt))   if proc_txt   else None
                    res_vec  = _to_plain_list(embedder.embed(res_txt))    if res_txt    else None

                    rec_full = {
                        "case_id": case_id,
                        "inputs_text":   inputs_txt or None,
                        "products_text": prod_txt  or None,
                        "processes_text": proc_txt or None,
                        "resources_text": res_txt  or None,
                        "inp_vec":  inp_vec,
                        "prod_vec": prod_vec,
                        "proc_vec": proc_vec,
                        "res_vec":  res_vec,
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
                                # Clean aggressively before continuing
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


                    st.success(
                        f"Created test case #{case_id} with FMEA rows, PPR links, and kb_index."
                    )

                except Exception as e:
                    st.error(f"Save failed: {e}")
                else:
                    st.session_state["fa_save_success_msg"] = (
                        f"Saved test case #{case_id}."
                    )
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

        ppr_for_export = _normalize_ppr_safe(
            st.session_state.get("assistant_ppr") or {}
        )
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

# -----------------------
# Cases Explorer (browse Supabase cases, view PPR + FMEA; tables only)
# -----------------------
with tab_cases:
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
                st.write("Inputs")
                st.dataframe(_list_to_df("Input Products", ppr.get("input_products")), use_container_width=True)
                st.write("Processes")
                st.dataframe(_list_to_df("Processes", ppr.get("processes")), use_container_width=True)
            with c2:
                st.write("Products")
                st.dataframe(_list_to_df("Products", ppr.get("products")), use_container_width=True)
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
