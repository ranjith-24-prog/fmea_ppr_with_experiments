# backend/rag_index.py
import io
import time
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

from backend.llm import Embeddings  # reuse your embedding class

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def extract_text_from_xml(xml_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []
    TAGS = [
        "SystemElement","Subsystem","ProcessStep","Function","FailureMode","PotentialFailure",
        "Effect","PotentialEffect","Severity","Cause","PotentialCause","Occurrence",
        "PreventionControl","CurrentPreventiveAction","DetectionControl","CurrentDetectionAction",
        "RecommendedAction","ActionTaken","Notes"
    ]
    for el in root.iter():
        fields = {}
        for ch in list(el):
            name = _strip_ns(ch.tag)
            if name in TAGS:
                val = (ch.text or "").strip()
                if val:
                    fields[name] = val
        if len(fields) >= 3:
            text = " | ".join([f"{k}={v}" for k, v in fields.items()])
            chunks.append({
                "text": text,
                "meta": {
                    "source_type": "xml",
                    "file_name": file_name,
                    "ppr_products": [],
                    "ppr_processes": [],
                    "ppr_resources": [],
                }
            })
    return chunks

def extract_text_from_excel(excel_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    except Exception:
        return []
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
        except Exception:
            continue
        df = df.fillna("")
        for _, row in df.iterrows():
            parts = []
            for col, val in row.items():
                sval = str(val).strip()
                if sval:
                    parts.append(f"{col}={sval}")
            if len(parts) >= 3:
                text = " | ".join(parts)
                chunks.append({
                    "text": text,
                    "meta": {
                        "source_type": "excel",
                        "file_name": f"{file_name}:{sheet}",
                        "ppr_products": [],
                        "ppr_processes": [],
                        "ppr_resources": [],
                    }
                })
    return chunks

def extract_text_from_db_rows(conn) -> List[Dict[str, Any]]:
    chunks = []
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, system_element, function, potential_failure, potential_effect, potential_cause, s1, o1, d1, rpn1, notes FROM fmea_rows")
        for r in cur.fetchall():
            rid, se, fn, pf, pe, pc, s1, o1, d1, rpn1, notes = r
            parts = []
            def add(k, v):
                if v not in (None, ""):
                    parts.append(f"{k}={v}")
            add("SystemElement", se)
            add("Function", fn)
            add("PotentialFailure", pf)
            add("PotentialEffect", pe)
            add("PotentialCause", pc)
            add("S", s1)
            add("O", o1)
            add("D", d1)
            add("RPN", rpn1)
            add("Notes", notes)
            if len(parts) >= 3:
                text = " | ".join(parts)
                chunks.append({
                    "text": text,
                    "meta": {
                        "source_type": "db_row",
                        "file_name": "",
                        "row_id": rid,
                        "ppr_products": [],
                        "ppr_processes": [],
                        "ppr_resources": [],
                    }
                })
    except Exception:
        pass
    return chunks

def chunk_and_embed_to_supabase(sb_client, chunks: List[Dict[str, Any]]):
    emb = Embeddings()
    for ch in chunks:
        text = ch["text"]
        meta = ch["meta"]
        try:
            vector = emb.embed(text)
        except Exception:
            continue
        payload = {
            "text": text,
            "embedding": vector,
            "source_type": meta.get("source_type", ""),
            "file_name": meta.get("file_name", ""),
            "row_id": meta.get("row_id", None),
            "ppr_products": meta.get("ppr_products", []),
            "ppr_processes": meta.get("ppr_processes", []),
            "ppr_resources": meta.get("ppr_resources", []),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        sb_client.table("kb_chunks").insert(payload).execute()

def semantic_search(sb_client, query_vector: list[float], top_k=10, filters=None):
    # Replace this with RPC when available. Temporary: naive select limited to top_k.
    # WARNING: Without RPC sorting by cosine similarity will not be applied here.
    res = sb_client.table("kb_chunks").select("*").limit(top_k).execute()
    return res.data or []
