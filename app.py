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
from backend.export import to_pretty_excel_bytes, to_structured_xml_bytes
from supabase import create_client, Client
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from styles import apply_global_styles, AGGRID_CUSTOM_CSS

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
from modes.knowledge_base import render_knowledge_base
from modes.cases_explorer import render_cases_explorer
from modes.fmea_assistant import render_fmea_assistant

st.set_page_config(page_title="CBR FMEA Assistant", layout="wide")
apply_global_styles()




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

def _complete_missing_with_llm(kb_rows: list[dict], query_ppr: dict, llm: LLM) -> list[dict]:
    prompt = {
        "instruction": (
            "Given the user PPR and relevant KB FMEA rows, propose additional FMEA rows ONLY for gaps. "
            "Avoid duplicates. Use EXACT schema keys."
        ),
        "ppr": query_ppr,
        "kb_rows": kb_rows,
        "schema": [
            "system_element","function","potential_failure","c1",
            "potential_effect","s1","c2","c3",
            "potential_cause","o1","current_preventive_action",
            "current_detection_action","d1","rpn1",
            "recommended_action","rd","action_taken",
            "s2","o2","d2","rpn2","notes",
        ],
    }

    try:
        rows_json = json.dumps(prompt, ensure_ascii=False)
        #st.write(f"DEBUG: _complete_missing_with_llm called; kb_rows = {len(kb_rows)}")
        gen_rows, _ = llm.generate_fmea_and_ppr_json(context_text=rows_json, ppr_hint=None)
        #st.write(f"DEBUG: LLM FMEA rows len = {len(gen_rows) if isinstance(gen_rows, list) else 'n/a'}")
        out = []
        if isinstance(gen_rows, list):
            for r in gen_rows:
                r["_provenance"] = "llm"
                out.append(r)
        return out
    except Exception as e:
        #st.error(f"DEBUG _complete_missing_with_llm error: {e}")
        return []



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


helpers = {
    "_guess_mime": _guess_mime,                # for KB
    "_build_supabase": _build_supabase,        # KB + FMEA + Cases
    "_supabase_bucket_name": _supabase_bucket_name,  # KB
    "_normalize_ppr_safe": _normalize_ppr_safe,      # KB + FMEA
    "ppr_editor_block": ppr_editor_block,      # KB + FMEA
    "ppr_to_cytoscape": ppr_to_cytoscape,      # if used anywhere
    "_concat_list": _concat_list,
    "_to_plain_list": _to_plain_list,          # KB + FMEA
    "_fetch_candidate_cases": _fetch_candidate_cases,
    "_fetch_rows_for_cases": _fetch_rows_for_cases,
    "_score_row_against_query": _score_row_against_query,
    "_select_kb_rows": _select_kb_rows,        # FMEA Assistant
    "_complete_missing_with_llm": _complete_missing_with_llm,  # FMEA
    "_normalize_numeric_and_rpn": _normalize_numeric_and_rpn,  # FMEA
    "_get_or_create_ppr": _get_or_create_ppr,  # if you use global version
    "_link_case_ppr": _link_case_ppr,          # if you use global version
    "_get_secret": _get_secret,                # FMEA save test case
}

# -----------------------
# Top navigation via tabs (works on all recent Streamlit versions)
# -----------------------
tab_fmea, tab_kb, tab_cases = st.tabs(
    ["🤖 FMEA Assistant", "📚 Knowledge Base", "🗂️ Cases Explorer"]
)

with tab_fmea:
    render_fmea_assistant(embedder, helpers)

with tab_kb:
    render_knowledge_base(embedder, helpers)

with tab_cases:
    render_cases_explorer(helpers)
