import io
import pandas as pd
import xml.etree.ElementTree as ET

APIS_KEYS = {
    "system_element","function","potential_failure","c1","potential_effect","s1",
    "c2","c3","potential_cause","o1","current_preventive_action",
    "current_detection_action","d1","rpn1","recommended_action","rd",
    "action_taken","s2","o2","d2","rpn2","notes"
}

TAG_MAP = {
    "systemelement": "system_element",
    "subsystem": "system_element",
    "processstep": "system_element",
    "system": "system_element",
    "function": "function",
    "failuremode": "potential_failure",
    "potentialfailure": "potential_failure",
    "failure": "potential_failure",
    "effect": "potential_effect",
    "potentialeffect": "potential_effect",
    "severity": "s1",
    "s1": "s1",
    "cause": "potential_cause",
    "potentialcause": "potential_cause",
    "occurrence": "o1",
    "o1": "o1",
    "preventioncontrol": "current_preventive_action",
    "currentpreventiveaction": "current_preventive_action",
    "detectioncontrol": "current_detection_action",
    "currentdetectionaction": "current_detection_action",
    "d1": "d1",
    "rpn": "rpn1",
    "rpn1": "rpn1",
    "recommendedaction": "recommended_action",
    "rd": "rd",
    "actiontaken": "action_taken",
    "s2": "s2",
    "o2": "o2",
    "d2": "d2",
    "rpn2": "rpn2",
    "notes": "notes",
    "c1": "c1",
    "c2": "c2",
    "c3": "c3",
    # Add discoveries from debug_list_tags() here, e.g.:
    # "processstepname": "system_element",
    # "failure_mode": "potential_failure",
    # "effect_of_failure": "potential_effect",
    # "detection": "d1",
}

def _strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag

def _text(el):
    return (el.text or "").strip()

def debug_list_tags(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
        seen = set()
        for el in root.iter():
            seen.add(_strip_ns(el.tag).lower())
        return sorted(list(seen))[:100]
    except Exception:
        return []

def parse_xml_fmea(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []

    # Build parent links
    parent = {root: None}
    stack = [root]
    while stack:
        node = stack.pop()
        for ch in list(node):
            parent[ch] = node
            stack.append(ch)
    def getparent(x):
        return parent.get(x)

    def is_row(elem):
        mapped_hits = 0
        nonempty_children = 0
        for ch in list(elem):
            nm = _strip_ns(ch.tag).lower()
            if TAG_MAP.get(nm) in APIS_KEYS:
                mapped_hits += 1
            if _text(ch):
                nonempty_children += 1
        return mapped_hits >= 2 or nonempty_children >= 3

    row_elems = [el for el in root.iter() if is_row(el)]
    rows = []
    for el in row_elems:
        row = {k: "" for k in APIS_KEYS}

        # Direct system element on this node
        for ch in list(el):
            nm = _strip_ns(ch.tag).lower()
            if TAG_MAP.get(nm) == "system_element" and not row["system_element"]:
                row["system_element"] = _text(ch)

        # Inherit from ancestors if still empty
        cur = getparent(el)
        while cur is not None and not row["system_element"]:
            for ch in list(cur):
                nm = _strip_ns(ch.tag).lower()
                if TAG_MAP.get(nm) == "system_element":
                    val = _text(ch)
                    if val:
                        row["system_element"] = val
                        break
            cur = getparent(cur)

        # Map known fields
        for ch in list(el):
            nm = _strip_ns(ch.tag).lower()
            key = TAG_MAP.get(nm)
            if key:
                val = _text(ch)
                if val and not row.get(key):
                    row[key] = val

        rows.append(row)

    return rows


import io
import pandas as pd

APIS_COLUMNS = [
    "system_element","function","potential_failure","c1",
    "potential_effect","s1","c2","c3",
    "potential_cause","o1","current_preventive_action",
    "current_detection_action","d1","rpn1",
    "recommended_action","rd","action_taken",
    "s2","o2","d2","rpn2","notes"
]

def parse_excel_fmea(excel_bytes):
    try:
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        fmea_rows = []

        # choose sheet: prefer FMEA, else first sheet
        sheet = "FMEA" if "FMEA" in xls.sheet_names else xls.sheet_names[0]
        df = xls.parse(sheet)

        # Basic cleaning
        df = df.fillna("")
        df.columns = [str(c).strip() for c in df.columns]

        # Map common headers to APIS keys (extend as needed)
        header_map = {
            # left: header in file, right: APIS key
            "System element": "system_element",
            "System Element": "system_element",
            "Process step": "system_element",
            "Function": "function",
            "Failure mode": "potential_failure",
            "Failure Mode": "potential_failure",
            "Potential failure": "potential_failure",
            "C1": "c1",
            "C 1": "c1",
            "Effect": "potential_effect",
            "Potential effect": "potential_effect",
            "Severity": "s1",
            "S": "s1",
            "C2": "c2",
            "C 2": "c2",
            "C3": "c3",
            "C 3": "c3",
            "Cause": "potential_cause",
            "Potential cause": "potential_cause",
            "Occurrence": "o1",
            "O": "o1",
            "Current preventive action": "current_preventive_action",
            "Prevention control": "current_preventive_action",
            "Current detection action": "current_detection_action",
            "Detection control": "current_detection_action",
            "Detection": "d1",
            "D": "d1",
            "RPN": "rpn1",
            "RPN1": "rpn1",
            "Recommended action": "recommended_action",
            "R/D": "rd",
            "Action taken": "action_taken",
            "S2": "s2",
            "O2": "o2",
            "D2": "d2",
            "RPN2": "rpn2",
            "Notes": "notes",
        }

        # Pass-through if some columns already use APIS names
        for k in APIS_COLUMNS:
            if k in df.columns:
                header_map[k] = k

        # Apply rename for columns that exist
        rename_map = {k: v for k, v in header_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Build row dicts
        for _, row in df.iterrows():
            r = {
                "system_element": row.get("system_element", ""),
                "function": row.get("function", ""),
                "potential_failure": row.get("potential_failure", ""),
                "c1": row.get("c1", ""),
                "potential_effect": row.get("potential_effect", ""),
                "s1": row.get("s1", ""),
                "c2": row.get("c2", ""),
                "c3": row.get("c3", ""),
                "potential_cause": row.get("potential_cause", ""),
                "o1": row.get("o1", ""),
                "current_preventive_action": row.get("current_preventive_action", ""),
                "current_detection_action": row.get("current_detection_action", ""),
                "d1": row.get("d1", ""),
                "rpn1": row.get("rpn1", ""),
                "recommended_action": row.get("recommended_action", ""),
                "rd": row.get("rd", ""),
                "action_taken": row.get("action_taken", ""),
                "s2": row.get("s2", ""),
                "o2": row.get("o2", ""),
                "d2": row.get("d2", ""),
                "rpn2": row.get("rpn2", ""),
                "notes": row.get("notes", ""),
            }
            # stringify values for display
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = ""
                elif not isinstance(v, str):
                    r[k] = str(v)
            # keep non-empty rows
            if any(v.strip() for v in r.values()):
                fmea_rows.append(r)

        return fmea_rows
    except Exception:
        return []


