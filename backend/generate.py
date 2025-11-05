import math
from typing import List, Dict, Any, Tuple
from .db import get_conn
from .llm import LLM, Embeddings
from .repository import search_ontology_by_type
from .retrieval import map_text_to_ppr
from .retain import persist_case

def clamp_int(x, lo=1, hi=10):
    try:
        xi = int(x)
        return max(lo, min(hi, xi))
    except Exception:
        return None

def ensure_ontology_class(conn, name: str, type_: str, parent_id=None, description=None) -> int:
    cur = conn.execute("SELECT id FROM ontology_class WHERE name=? AND type=?", (name, type_))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO ontology_class(name,type,parent_id,description) VALUES(?,?,?,?)",
        (name, type_, parent_id, description)
    )
    return cur.lastrowid

def ensure_failure_mode(conn, name: str, desc: str, process_class_id: int|None) -> int:
    cur = conn.execute("SELECT id FROM risk_failure_mode WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_failure_mode(name,description,process_class_id) VALUES(?,?,?)",
        (name, desc, process_class_id)
    )
    return cur.lastrowid

def ensure_cause(conn, name: str, desc: str, resource_class_id: int|None) -> int:
    cur = conn.execute("SELECT id FROM risk_cause WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_cause(name,description,resource_class_id) VALUES(?,?,?)",
        (name, desc, resource_class_id)
    )
    return cur.lastrowid

def ensure_effect(conn, name: str, desc: str, product_class_id: int|None) -> int:
    cur = conn.execute("SELECT id FROM risk_effect WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_effect(name,description,product_class_id) VALUES(?,?,?)",
        (name, desc, product_class_id)
    )
    return cur.lastrowid

def ensure_control(conn, name: str, desc: str|None) -> int:
    cur = conn.execute("SELECT id FROM risk_control WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_control(name,description) VALUES(?,?)",
        (name, desc)
    )
    return cur.lastrowid

def link_once(conn, table: str, a: int, b: int):
    conn.execute(f"INSERT OR IGNORE INTO {table} VALUES(?,?)", (a,b))

def pick_one_id(val):
    if val is None:
        return None
    if isinstance(val, list):
        return val[0] if val else None
    try:
        return int(val)
    except Exception:
        return None

def generate_fmea_from_llm(conn, embedder: Embeddings, llm: LLM, user_text: str, topk=3, case_title="LLM-generated case") -> Tuple[int, List[Dict[str, Any]]]:
    # 1) Map to PPR
    ontology = conn.execute("SELECT * FROM ontology_class").fetchall()
    ontology_dicts = [dict(r) for r in ontology]
    mapped = map_text_to_ppr(user_text, ontology_dicts, embedder, topk=topk)

    def names_by_ids(type_name, ids):
        if not ids: return []
        q = "SELECT name FROM ontology_class WHERE id IN ({})".format(",".join("?"*len(ids)))
        rows = conn.execute(q, ids).fetchall()
        return [r["name"] for r in rows]

    ppr_hint = {
        "Product": names_by_ids("Product", mapped.get("Product") or []),
        "Process": names_by_ids("Process", mapped.get("Process") or []),
        "Resource": names_by_ids("Resource", mapped.get("Resource") or []),
    }

    # 2) Call LLM to generate APIS-style FMEA rows
    raw_rows = llm.generate_fmea_rows_json(user_text, ppr_hint)

    # 3) Validate / normalize rows for all expected keys
    clean_rows = []
    for it in raw_rows:
        def norm_str(k): return llm.normalize_label(it.get(k) or "")
        def norm_int(k): return clamp_int(it.get(k))

        # Minimal required fields
        pfailure = norm_str("potential_failure")
        pcause = norm_str("potential_cause")
        peffect = norm_str("potential_effect")
        if not pfailure or not pcause or not peffect:
            continue

        # Compose normalized row including fallback for legacy keys
        clean_rows.append({
            "system_element": norm_str("system_element"),
            "function": norm_str("function"),
            "potential_failure": pfailure,
            "c1": norm_str("c1"),
            "potential_effect": peffect,
            "s1": norm_int("s1"),
            "c2": norm_str("c2"),
            "c3": norm_str("c3"),
            "potential_cause": pcause,
            "o1": norm_int("o1"),
            "current_preventive_action": norm_str("current_preventive_action"),
            "current_detection_action": norm_str("current_detection_action"),
            "d1": norm_int("d1"),
            "rpn1": 0,  # will be computed later if needed
            "recommended_action": norm_str("recommended_action"),
            "rd": norm_str("rd"),
            "action_taken": norm_str("action_taken"),
            "s2": norm_int("s2"),
            "o2": norm_int("o2"),
            "d2": norm_int("d2"),
            "rpn2": 0,
            "notes": norm_str("notes"),
        })

    if not clean_rows:
        raise ValueError("LLM returned no valid FMEA rows after validation.")

    # 4) Use top mapped ids as before
    product_id = pick_one_id(mapped.get("Product"))
    process_id = pick_one_id(mapped.get("Process"))
    resource_id = pick_one_id(mapped.get("Resource"))

    # 5) Upsert risks, causes, effects, controls and link tables
    saved_rows = []
    for r in clean_rows:
        fm_id = ensure_failure_mode(conn, r["potential_failure"], desc=None, process_class_id=process_id)
        cause_id = ensure_cause(conn, r["potential_cause"], desc=None, resource_class_id=resource_id)
        effect_id = ensure_effect(conn, r["potential_effect"], desc=None, product_class_id=product_id)

        # Controls: Try to parse controls from recommended actions or preventive action if applicable
        ctrl_names = [r["current_preventive_action"]]
        # Clean controls list (filter empty)
        ctrl_names = [c for c in ctrl_names if c]

        control_ids = []
        for c in ctrl_names:
            cid = ensure_control(conn, c, desc=None)
            control_ids.append(cid)

        # Link in junction tables
        for cid in [cause_id]:
            link_once(conn, "failure_cause", fm_id, cid)
        for eid in [effect_id]:
            link_once(conn, "failure_effect", fm_id, eid)
        for cid in control_ids:
            link_once(conn, "failure_control", fm_id, cid)

        # Attach IDs and minimal info for later display
        saved_rows.append({
            "failure_mode_id": fm_id,
            "cause_id": cause_id,
            "effect_id": effect_id,
            "control_ids": control_ids,
            # APIS columns to pass along for UI display / export
            **r
        })

    # 6) Persist case and rows: create one `case_fmea` row per control to keep DB schema
    case_rows = []
    for row in saved_rows:
        if row["control_ids"]:
            for ctrl_id in row["control_ids"]:
                case_rows.append({
                    "failure_mode_id": row["failure_mode_id"],
                    "cause_id": row["cause_id"],
                    "effect_id": row["effect_id"],
                    "control_id": ctrl_id,
                    "severity": row.get("s1"),
                    "occurrence": row.get("o1"),
                    "detection": row.get("d1"),
                    "notes": row.get("notes"),
                })
        else:
            case_rows.append({
                "failure_mode_id": row["failure_mode_id"],
                "cause_id": row["cause_id"],
                "effect_id": row["effect_id"],
                "control_id": None,
                "severity": row.get("s1"),
                "occurrence": row.get("o1"),
                "detection": row.get("d1"),
                "notes": row.get("notes"),
            })

    case_id = persist_case(conn, case_title, user_text, product_id, process_id, resource_id, case_rows)
    return case_id, saved_rows

