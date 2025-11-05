from collections import defaultdict
from .db import get_conn

def index_by_id(rows):
    return {r["id"]: r for r in rows}

def propose_fmea_rows(conn, candidates, ontology, mapped_ppr):
    product_ids = set(mapped_ppr.get("Product", []))
    failure_modes = candidates["failure_modes"]
    causes = candidates["causes"]
    effects = candidates["effects"]
    controls = candidates["controls"]

    with get_conn() as conn2:
        fm_effects = {}
        fm_causes = {}
        fm_controls = {}
        cur = conn2.execute("SELECT * FROM failure_effect")
        for r in cur.fetchall():
            fm_effects.setdefault(r["failure_mode_id"], []).append(r["effect_id"])
        cur = conn2.execute("SELECT * FROM failure_cause")
        for r in cur.fetchall():
            fm_causes.setdefault(r["failure_mode_id"], []).append(r["cause_id"])
        cur = conn2.execute("SELECT * FROM failure_control")
        for r in cur.fetchall():
            fm_controls.setdefault(r["failure_mode_id"], []).append(r["control_id"])

    causes_idx = index_by_id(causes)
    effects_idx = index_by_id(effects)
    controls_idx = index_by_id(controls)

    final_rows = []
    for fm in failure_modes:
        cause_ids = fm_causes.get(fm["id"], []) or [None]
        effect_ids = fm_effects.get(fm["id"], []) or [None]
        control_ids = fm_controls.get(fm["id"], []) or [None]

        # Filter effects by product specificity
        filtered_effect_ids = []
        for eid in effect_ids:
            if eid is None:
                filtered_effect_ids.append(None)
                continue
            eff = effects_idx.get(eid)
            if not eff:
                continue
            pcid = eff.get("product_class_id")
            if (pcid is None) or (pcid in product_ids):
                filtered_effect_ids.append(eid)

        if not filtered_effect_ids:
            filtered_effect_ids = [None]

        for cid in cause_ids:
            for eid in filtered_effect_ids:
                for ctrl in control_ids:
                    row = {
                        "failure_mode_id": fm["id"],
                        "failure_mode": fm["name"],
                        "cause_id": cid,
                        "cause": causes_idx[cid]["name"] if cid in causes_idx else None,
                        "effect_id": eid,
                        "effect": effects_idx[eid]["name"] if eid in effects_idx else None,
                        "control_id": ctrl,
                        "control": controls_idx[ctrl]["name"] if ctrl in controls_idx else None,
                        "severity": None,
                        "occurrence": None,
                        "detection": None,
                        "notes": ""
                    }
                    final_rows.append(row)

    # Deduplicate
    seen = set()
    unique_rows = []
    for r in final_rows:
        key = (r["failure_mode_id"], r["cause_id"], r["effect_id"], r["control_id"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(r)
    return unique_rows
