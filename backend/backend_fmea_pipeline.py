# backend_fmea_pipeline.py
# Purpose:
# - Replace broken file_parser for Excel→FMEA by porting your working ipynb logic.
# - No FastAPI. Import and call from app.py.
# - Flow: read Excel → normalize like notebook → return df for UI → optional LLM enhance →
#         finalize → save raw Excel to Supabase + save FMEA/PPR to DB.

import io
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import os
import pandas as pd

# If your llm.py exposes a class LLM with methods used in the app, import it:
from backend.llm import LLM  # matches your existing file

# -----------------------------
# Schema constants
# -----------------------------
APIS_COLUMNS = [
    "systemelement", "function", "potentialfailure", "c1",
    "potentialeffect", "s1", "c2", "c3",
    "potentialcause", "o1", "currentpreventiveaction",
    "currentdetectionaction", "d1", "rpn1",
    "recommendedaction", "rd", "actiontaken",
    "s2", "o2", "d2", "rpn2", "notes"
]

# CSV header -> APIS internal
CSV_TO_APIS_COLMAP = {
    "System element": "systemelement",
    "Function": "function",
    "Potential failure": "potentialfailure",
    "C": "c1",
    "Potential effect(s) of failure": "potentialeffect",
    "S": "s1",
    "C.1": "c2",
    "C.2": "c3",
    "Potential cause(s) of failure": "potentialcause",
    "O": "o1",
    "Current preventive action": "currentpreventiveaction",
    "Current detection action": "currentdetectionaction",
    "D": "d1",
    "RPN": "rpn1",
    "Recommended action": "recommendedaction",
    "R/D": "rd",
    "Action taken": "actiontaken",
    "S.1": "s2",
    "O.1": "o2",
    "D.1": "d2",
    "RPN.1": "rpn2",
}

# APIS internal -> UI (snake_case used by Streamlit grid)
APIS_TO_UI = {
    "systemelement": "system_element",
    "function": "function",
    "potentialfailure": "potential_failure",
    "c1": "c1",
    "potentialeffect": "potential_effect",
    "s1": "s1",
    "c2": "c2",
    "c3": "c3",
    "potentialcause": "potential_cause",
    "o1": "o1",
    "currentpreventiveaction": "current_preventive_action",
    "currentdetectionaction": "current_detection_action",
    "d1": "d1",
    "rpn1": "rpn1",
    "recommendedaction": "recommended_action",
    "rd": "rd",
    "actiontaken": "action_taken",
    "s2": "s2",
    "o2": "o2",
    "d2": "d2",
    "rpn2": "rpn2",
    "notes": "notes",
}

# Final UI order the grid expects
UI_ORDER = [
    "system_element","function","potential_failure","c1",
    "potential_effect","s1","c2","c3",
    "potential_cause","o1","current_preventive_action",
    "current_detection_action","d1","rpn1",
    "recommended_action","rd","action_taken",
    "s2","o2","d2","rpn2","notes"
]

# -----------------------------
# Notebook-parity helpers
# -----------------------------
def _read_excel_first_sheet(excel_bytes: bytes) -> pd.DataFrame:
    with io.BytesIO(excel_bytes) as fh:
        df = pd.read_excel(fh, sheet_name=0, header=None)
    return df

def _drop_first_five_rows_and_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    df_trim = df.drop(index=list(range(0, 5)), errors="ignore").reset_index(drop=True)
    header = df_trim.iloc[0].astype(str).tolist()
    df_trim = df_trim.iloc[1:].reset_index(drop=True)
    df_trim.columns = header
    return df_trim


def _insert_system_element_first_col(df_trim: pd.DataFrame) -> pd.DataFrame:
    if "System element" in df_trim.columns:
        col = df_trim.pop("System element")
        df_trim.insert(0, "System element", col)
    else:
        df_trim.insert(0, "System element", None)
    return df_trim

def _drop_nan_named_columns(df_trim: pd.DataFrame) -> pd.DataFrame:
    col_index = pd.Index(df_trim.columns)
    cond_missing = col_index.isnull()
    col_as_str = col_index.map(lambda x: str(x).strip() if not pd.isna(x) else x)
    cond_literal_nan = col_as_str.map(lambda x: isinstance(x, str) and x.lower() == "nan")
    mask_invalid = cond_missing | cond_literal_nan
    to_drop = list(col_index[mask_invalid])
    if to_drop:
        df_trim = df_trim.drop(columns=to_drop)
    return df_trim

def _propagate_system_element_from_function(df_trim: pd.DataFrame) -> pd.DataFrame:
    if "Function" not in df_trim.columns:
        return df_trim
    func = df_trim["Function"].fillna("").astype(str)
    pattern = re.compile(r"^\s*System\s*element\s*:\s*(.*)\s*$", flags=re.IGNORECASE)
    extracted = func.map(lambda x: pattern.match(x).group(1).strip() if pattern.match(x) else None)
    if "System element" not in df_trim.columns:
        df_trim.insert(0, "System element", None)
    df_trim.loc[extracted.notna(), "System element"] = extracted[extracted.notna()]
    df_trim["System element"] = df_trim["System element"].ffill()
    marker_rows = func.map(lambda x: bool(pattern.match(x)))
    df_trim = df_trim.loc[~marker_rows].reset_index(drop=True)
    return df_trim

def _forward_fill_all_columns(df_trim: pd.DataFrame) -> pd.DataFrame:
    return df_trim.ffill().reset_index(drop=True)

# backend_fmea_pipeline.py  (ADD NEAR OTHER SMALL HELPERS)

def _merge_ppr_dicts(base: dict, add: dict) -> dict:
    """Merge two PPR dicts including the new 'input_products' key without affecting FMEA rows."""
    return {
        "input_products": sorted(set(base.get("input_products", [])) | set(add.get("input_products", []))),
        "products": sorted(set(base.get("products", [])) | set(add.get("products", []))),
        "processes": sorted(set(base.get("processes", [])) | set(add.get("processes", []))),
        "resources": sorted(set(base.get("resources", [])) | set(add.get("resources", []))),
    }

# -----------------------------
# Positional duplicate handler
# -----------------------------
def _assign_positional_ratings_and_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle sheets where headers are duplicated (e.g., 'S','S','O','O','D','D','RPN','RPN','C','C','C')
    by assigning first/second occurrence to s1/s2, o1/o2, d1/d2, rpn1/rpn2 and first three C to c1/c2/c3.
    This runs BEFORE CSV_TO_APIS_COLMAP so downstream renames don't lose data.
    """
    if df is None or df.empty:
        return df
    cols = list(map(str, df.columns))
    df2 = df.copy()

    # temp index->target key map
    target_keys = [None] * len(cols)

    # C columns left-to-right to c1,c2,c3
    c_count = 0
    for i, name in enumerate(cols):
        if name.strip().upper() == "C":
            c_count += 1
            if c_count == 1:
                target_keys[i] = "c1"
            elif c_count == 2:
                target_keys[i] = "c2"
            elif c_count == 3:
                target_keys[i] = "c3"

    # assign two occurrences helper
    def assign_two(name_upper, key1, key2):
        idxs = [i for i, n in enumerate(cols) if n.strip().upper() == name_upper]
        if idxs:
            target_keys[idxs[0]] = key1
        if len(idxs) > 1:
            target_keys[idxs[1]] = key2

    assign_two("S", "s1", "s2")
    assign_two("O", "o1", "o2")
    assign_two("D", "d1", "d2")
    assign_two("RPN", "rpn1", "rpn2")

    # Copy assigned source columns into new APIS-internal columns
    for idx, key in enumerate(target_keys):
        if key and key not in df2.columns:
            df2[key] = df2.iloc[:, idx]

    return df2

# -----------------------------
# Canonical normalizer (CSV/APIS/UI)
# -----------------------------
def normalize_to_apis_ui(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid mutating caller
    src = df.copy()

    # 0) Keep original headers to append any unmapped columns for debugging
    orig_cols = list(map(str, src.columns))

    # 0.a) Handle positional duplicates for C/S/O/D/RPN
    src = _assign_positional_ratings_and_controls(src)

    # 1) Bring any CSV-style names to APIS internal keys (with trimmed comparison)
    trimmed_map = {}
    for k, v in CSV_TO_APIS_COLMAP.items():
        if k in src.columns:
            trimmed_map[k] = v
        else:
            k_trim = k.strip()
            col_hits = [c for c in src.columns if str(c).strip() == k_trim]
            if col_hits:
                trimmed_map[col_hits[0]] = v
    df1 = src.rename(columns=trimmed_map)

    # 2) Ensure APIS internal keys exist and order them
    for col in APIS_COLUMNS:
        if col not in df1.columns:
            df1[col] = None
    df1 = df1[APIS_COLUMNS]

    # 3) Convert APIS internal keys to final UI snake_case keys
    df2 = df1.rename(columns=APIS_TO_UI)

    # 4) Ensure UI columns exist and order them
    for col in UI_ORDER:
        if col not in df2.columns:
            df2[col] = None
    df2 = df2[UI_ORDER]

    # 5) Append any UNMAPPED original columns at the end for debugging
    mapped_src_names = set(CSV_TO_APIS_COLMAP.keys())
    ui_names_set = set(UI_ORDER)
    passthrough = []
    for oc in orig_cols:
        oc_str = str(oc)
        if oc_str in mapped_src_names:  # already mapped via CSV map
            continue
        if oc_str in ui_names_set:      # already present as UI column
            continue
        if oc_str in APIS_TO_UI.values():  # already an APIS->UI name
            continue
        passthrough.append(oc_str)
    for col in passthrough:
        if col not in df2.columns:
            if col in src.columns:
                df2[col] = src[col]
            else:
                df2[col] = None

    # 6) Keep rows that have at least a core field
    core_fields = ["system_element", "function", "potential_failure", "potential_effect", "potential_cause"]
    def _has_core(r):
        for k in core_fields:
            v = r.get(k) if isinstance(r, dict) else r[k]
            if v is not None and str(v).strip():
                return True
        return False
    if isinstance(df2, pd.DataFrame):
        mask_keep = df2.apply(lambda r: _has_core(r), axis=1)
        df2 = df2.loc[mask_keep].reset_index(drop=True)

    # 7) Stringify non-empty values for stable CSV export/UI
    df2 = df2.apply(lambda col: col.where(col.notna(), None))
    df2 = df2.applymap(lambda v: v if v is None or isinstance(v, str) else str(v))

    return df2

# -----------------------------
# End-to-end parse
# -----------------------------
def _auto_promote_header(df: pd.DataFrame, search_rows: int = 15) -> pd.DataFrame:
    """
    Scan the first `search_rows` rows to find a header line that contains required FMEA labels.
    If found, promote that row to header and drop rows above; else fall back to 'drop 5 rows' heuristic.
    """
    # Candidate labels that should appear in a proper header row (any subset suffices)
    required_any = [
        "System element", "Function", "Potential failure",
        "Potential effect(s) of failure", "Potential cause(s) of failure",
        "Current preventive action", "Current detection action",
        "S", "O", "D", "RPN", "C"
    ]
    n = min(search_rows, len(df))
    for i in range(n):
        row = df.iloc[i].astype(str).str.strip().tolist()
        # count how many known header tokens appear anywhere in this row
        hits = sum(1 for token in required_any if token in row)
        if hits >= 3:
            # Use this row as header
            header = row
            df_trim = df.iloc[i+1:].reset_index(drop=True)
            df_trim.columns = header
            return df_trim
    # Fallback: old behavior (drop first 5 rows)
    df_trim = df.drop(index=list(range(0, 5)), errors="ignore").reset_index(drop=True)
    header = df_trim.iloc[0].astype(str).tolist()
    df_trim = df_trim.iloc[1:].reset_index(drop=True)
    df_trim.columns = header
    return df_trim


def parse_excel_using_notebook_logic(excel_bytes: bytes) -> pd.DataFrame:
    raw = _read_excel_first_sheet(excel_bytes)
    step0 = _auto_promote_header(raw)  # robust header finder
    step1 = step0
    step2 = _insert_system_element_first_col(step1)
    step3 = _drop_nan_named_columns(step2)
    step4 = _propagate_system_element_from_function(step3)
    step5 = _forward_fill_all_columns(step4)
    final_df = normalize_to_apis_ui(step5)
    return final_df

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.to_dict(orient="records")

# -----------------------------
# LLM enhancement
# -----------------------------
def enhance_with_llm(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    llm = LLM(model_name="sonar-pro")
    rows_json = json.dumps(rows, ensure_ascii=False)
    ppr = {"products": [], "processes": [], "resources": []}

    out_rows = rows
    try:
        fmea_generated, ppr_out = llm.generate_fmea_and_ppr_json(context_text=rows_json, ppr_hint=None)
        if isinstance(fmea_generated, list) and fmea_generated:
            out_rows = fmea_generated
        if isinstance(ppr_out, dict):
            ppr = {**ppr, **ppr_out}
    except Exception:
        try:
            ppr_out = llm.generate_ppr_from_fmea(rows_json)
            if isinstance(ppr_out, dict):
                ppr = {**ppr, **ppr_out}
        except Exception:
            pass

    df = pd.DataFrame(out_rows)
    for col in UI_ORDER:
        if col not in df.columns:
            df[col] = None
    df = df[UI_ORDER].fillna(method="ffill")
    enhanced = df.to_dict(orient="records")

    for k in ("products", "processes", "resources"):
        if k not in ppr or not isinstance(ppr[k], list):
            ppr[k] = []
    return enhanced, ppr

# -----------------------------
# Integration Hooks for app.py
# -----------------------------
def process_excel_for_preview(excel_bytes: bytes) -> List[Dict[str, Any]]:
    df = parse_excel_using_notebook_logic(excel_bytes)
    return df.to_dict(orient="records")

# Generate PPR lists from parsed FMEA rows (expects list[dict])
def generate_ppr_only(rows: list[dict]) -> dict:
    """
    Extend the naive PPR extractor to a 4-pillar model:
      - input_products: consumables, base/work materials, fillers, gases, adhesives, wires, etc.
      - processes: verbs/operations (weld, bond, cut, drill, assemble, braze, heat-treat, coat...)
      - resources: equipment, fixtures, tools, cameras, guns, robots...
      - products: everything else that looks like an artifact/output when not matched above.

    This is deliberately heuristic and safe to swap with your robust rules later.
    It does NOT modify FMEA row schemas or APIS columns.
    """
    input_products, products, processes, resources = set(), set(), set(), set()

    # Keyword buckets (expand freely)
    process_kw = (
        "weld", "bond", "cut", "drill", "assemble", "braze", "solder", "rivet",
        "deburr", "machine", "mill", "turn", "grind", "coat", "paint",
        "anodize", "heat treat", "form", "bend", "stamp", "laser", "polish"
    )
    resource_kw = (
        "fixture", "camera", "gun", "tool", "jig", "robot", "cobot",
        "welder", "nozzle", "torch", "spindle", "press", "furnace",
        "ovens", "conveyor", "sensor", "tester", "gauge", "gaging", "vision"
    )
    input_kw = (
        # materials and consumables that feed processes
        "gas", "argon", "co2", "shielding", "adhesive", "epoxy", "glue",
        "filler", "wire", "flux", "powder", "rod", "solder",
        "base material", "workpiece", "sheet", "plate", "bar", "stock",
        "fastener", "bolt", "screw", "nut", "insert", "sealant", "primer"
    )

    # Columns to scan (same as your current function)
    cols = ["system_element", "function", "potential_failure", "potential_cause", "recommended_action", "notes"]

    def classify(val: str):
        txt = val.lower()
        # Processes: look for verbs/operations
        if any(k in txt for k in process_kw):
            processes.add(val)
            return
        # Resources: equipment/fixtures
        if any(k in txt for k in resource_kw):
            resources.add(val)
            return
        # Input products: consumables/base materials
        if any(k in txt for k in input_kw):
            input_products.add(val)
            return
        # Otherwise, treat as product-like artifact
        products.add(val)

    for r in rows or []:
        for key in cols:
            val = (r.get(key) or "").strip()
            if not val:
                continue
            classify(val)

    return {
        "input_products": sorted(x for x in input_products if len(x) > 2),
        "products": sorted(x for x in products if len(x) > 2),
        "processes": sorted(x for x in processes if len(x) > 2),
        "resources": sorted(x for x in resources if len(x) > 2),
    }



def apply_enhancement(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Enhance FMEA rows with LLM:
    - Fill missing values where possible using domain knowledge.
    - Improve clarity/consistency of textual fields (not change meaning).
    - Validate S/O/D to be integers in 1..10; recompute RPN1/RPN2 if needed.
    - Return fully-populated rows in the exact UI schema, plus PPR recommendations.
    """
    # 0) Enforce UI schema before prompt
    df_in = pd.DataFrame(rows)
    for col in UI_ORDER:
        if col not in df_in.columns:
            df_in[col] = None
    df_in = df_in[UI_ORDER]
    # Keep an original copy to merge if LLM leaves cells blank
    orig_rows = df_in.to_dict(orient="records")

    # 1) Build explicit instruction for the LLM
    schema_desc = {
        "system_element": "string",
        "function": "string",
        "potential_failure": "string",
        "c1": "string",
        "potential_effect": "string",
        "s1": "int(1-10)",
        "c2": "string",
        "c3": "string",
        "potential_cause": "string",
        "o1": "int(1-10)",
        "current_preventive_action": "string",
        "current_detection_action": "string",
        "d1": "int(1-10)",
        "rpn1": "int = s1*o1*d1",
        "recommended_action": "string",
        "rd": "string",
        "action_taken": "string",
        "s2": "int(1-10) optional",
        "o2": "int(1-10) optional",
        "d2": "int(1-10) optional",
        "rpn2": "int = s2*o2*d2 if s2/o2/d2 present else null",
        "notes": "string optional"
    }
    system_instruction = (
        "You are a manufacturing FMEA assistant.\n"
        "Task:\n"
        "1) Review the provided FMEA rows.\n"
        "2) Fill missing fields conservatively using domain knowledge and internal consistency.\n"
        "3) Improve clarity and normalize style for textual fields without changing intent.\n"
        "4) Validate severity/occurrence/detection (s1/o1/d1 and s2/o2/d2) as integers in 1..10.\n"
        "5) Compute RPNs as rpn1=s1*o1*d1 and rpn2=s2*o2*d2 when second ratings exist.\n"
        "6) Produce PPR recommendations (products/processes/resources) from the completed FMEA.\n"
        "7) Return JSON with keys: 'fmea' (list of rows in EXACT schema) and 'ppr' "
        "   with keys 'products','processes','resources' (arrays of strings).\n"
        "Exact schema (snake_case):\n"
        f"{json.dumps(schema_desc, ensure_ascii=False)}\n"
        "Only include keys in this schema for each row. Do not add new columns.\n"
        "Be concise and consistent; do not fabricate unrealistic data.\n"
    )

    payload_rows = df_in.to_dict(orient="records")
    user_input = {
        "fmea": payload_rows
    }

    # 2) Call the LLM (single combined endpoint preferred)
    llm = LLM(model_name="sonar-pro")
    try:
        fmea_generated, ppr_out = llm.generate_fmea_and_ppr_json(
            context_text=f"{system_instruction}\n\nCurrent FMEA JSON:\n{json.dumps(user_input, ensure_ascii=False)}",
            ppr_hint=None
        )
    except Exception:
        # Fallback to existing behavior (PPR only)
        fmea_generated, ppr_out = None, None

    # 3) Merge and sanitize outputs
    # If LLM returned a new FMEA list, use it; otherwise fall back to original rows
    if isinstance(fmea_generated, list) and fmea_generated:
        df_out = pd.DataFrame(fmea_generated)
    else:
        df_out = df_in.copy()

    # Enforce schema/order; if LLM missed fields, keep original values
    for col in UI_ORDER:
        if col not in df_out.columns:
            df_out[col] = None
    df_out = df_out[UI_ORDER]

    # For any empty cell from LLM, backfill with original
    df_orig = pd.DataFrame(orig_rows)
    def _choose(llm_val, orig_val):
        return llm_val if (llm_val is not None and str(llm_val).strip() != "") else orig_val
    for col in UI_ORDER:
        df_out[col] = [ _choose(df_out.at[i,col], df_orig.at[i,col]) for i in range(len(df_out)) ]

    # Normalize numeric ratings and recompute RPNs
    def _to_int_1_10(x):
        try:
            v = int(str(x).strip())
            if 1 <= v <= 10:
                return v
        except Exception:
            pass
        return None

    for col in ["s1","o1","d1","s2","o2","d2"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(_to_int_1_10)

    def _rpn(a,b,c):
        return a*b*c if all(isinstance(v,int) for v in [a,b,c]) else None
    df_out["rpn1"] = [ _rpn(a,b,c) for a,b,c in zip(df_out["s1"], df_out["o1"], df_out["d1"]) ]
    if "s2" in df_out.columns and "o2" in df_out.columns and "d2" in df_out.columns:
        df_out["rpn2"] = [ _rpn(a,b,c) for a,b,c in zip(df_out["s2"], df_out["o2"], df_out["d2"]) ]

    # Final tidy: forward fill only for 'system_element' (grouping), not for technical ratings
    if "system_element" in df_out.columns:
        df_out["system_element"] = df_out["system_element"].ffill()

    # Build PPR
    ppr = {"products": [], "processes": [], "resources": []}
    if isinstance(ppr_out, dict):
        ppr = {
            "products": [x for x in (ppr_out.get("products") or []) if isinstance(x,str) and x.strip()],
            "processes": [x for x in (ppr_out.get("processes") or []) if isinstance(x,str) and x.strip()],
            "resources": [x for x in (ppr_out.get("resources") or []) if isinstance(x,str) and x.strip()],
        }
    else:
        # Best-effort fallback from current table if the model didn’t provide PPR
        try:
            text_blob = " ".join([
                " ".join(filter(None, map(str, row.values()))) for row in payload_rows
            ])[:4000]
            ppr_guess = llm.generate_ppr_from_fmea(text_blob)
            if isinstance(ppr_guess, dict):
                ppr = {
                    "products": [x for x in (ppr_guess.get("products") or []) if isinstance(x,str) and x.strip()],
                    "processes": [x for x in (ppr_guess.get("processes") or []) if isinstance(x,str) and x.strip()],
                    "resources": [x for x in (ppr_guess.get("resources") or []) if isinstance(x,str) and x.strip()],
                }
        except Exception:
            pass

    final_rows = df_out.to_dict(orient="records")
    return final_rows, ppr


# -----------------------------
# Finalize and save to Supabase (PPR persist extended with inputs – optional)
# -----------------------------
def finalize_and_save(
    sb_client,
    excel_bytes: bytes,
    excel_filename: str,
    final_rows: List[Dict[str, Any]],
    ppr: Dict[str, List[str]],
    title: str,
    description: str
) -> Dict[str, Any]:
    bucket = os.getenv("SUPABASE_BUCKET", "kb-files")
    path = f"uploads/{int(time.time())}_{excel_filename}"
    sb_client.storage.from_(bucket).upload(path, excel_bytes, {"content-type": _guess_mime(excel_filename)})

    case_res = sb_client.table("cases").insert({"title": title, "description": description}).execute()
    case_id = case_res.data[0]["id"]

    df = normalize_to_apis_ui(pd.DataFrame(final_rows))
    records = df_to_records(df)
    for r in records:
        r["case_id"] = case_id
    if records:
        sb_client.table("fmearows").insert(records).execute()

    # Persist legacy PPR
    for name in (ppr.get("products") or []):
        if name:
            sb_client.table("products").insert({"name": name, "case_id": case_id}).execute()
    for name in (ppr.get("processes") or []):
        if name:
            sb_client.table("processes").insert({"name": name, "case_id": case_id}).execute()
    for name in (ppr.get("resources") or []):
        if name:
            sb_client.table("resources").insert({"name": name, "case_id": case_id}).execute()

    # NEW: persist input_products if your DB has an `inputs` table (optional)
    for name in (ppr.get("input_products") or []):
        if name:
            try:
                sb_client.table("inputs").insert({"name": name, "case_id": case_id}).execute()
            except Exception:
                # Ignore if table not present; you said you can alter tables.
                pass

    return {"status": "ok", "case_id": case_id, "file_path": path}

def _guess_mime(filename: str) -> str:
    f = filename.lower()
    if f.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if f.endswith(".xls"):
        return "application/vnd.ms-excel"
    return "application/octet-stream"
