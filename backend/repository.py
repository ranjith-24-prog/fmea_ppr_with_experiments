from .db import get_conn

def get_all_ontology(conn):
    cur = conn.execute("SELECT * FROM ontology_class")
    return [dict(r) for r in cur.fetchall()]

def search_ontology_by_type(conn, type_):
    cur = conn.execute("SELECT * FROM ontology_class WHERE type=?", (type_,))
    return [dict(r) for r in cur.fetchall()]

def get_failure_modes_by_process(conn, process_class_ids):
    if not process_class_ids:
        return []
    q = "SELECT * FROM risk_failure_mode WHERE process_class_id IN ({})".format(
        ",".join("?"*len(process_class_ids))
    )
    cur = conn.execute(q, process_class_ids)
    return [dict(r) for r in cur.fetchall()]

def get_causes_for_failure(conn, failure_mode_ids):
    if not failure_mode_ids: return []
    q = """SELECT DISTINCT rc.* FROM risk_cause rc
           JOIN failure_cause fc ON rc.id=fc.cause_id
           WHERE fc.failure_mode_id IN ({})""".format(",".join("?"*len(failure_mode_ids)))
    cur = conn.execute(q, failure_mode_ids)
    return [dict(r) for r in cur.fetchall()]

def get_effects_for_failure(conn, failure_mode_ids):
    if not failure_mode_ids: return []
    q = """SELECT DISTINCT re.* FROM risk_effect re
           JOIN failure_effect fe ON re.id=fe.effect_id
           WHERE fe.failure_mode_id IN ({})""".format(",".join("?"*len(failure_mode_ids)))
    cur = conn.execute(q, failure_mode_ids)
    return [dict(r) for r in cur.fetchall()]

def get_controls_for_failure(conn, failure_mode_ids):
    if not failure_mode_ids: return []
    q = """SELECT DISTINCT rc.* FROM risk_control rc
           JOIN failure_control fc ON rc.id=fc.control_id
           WHERE fc.failure_mode_id IN ({})""".format(",".join("?"*len(failure_mode_ids)))
    cur = conn.execute(q, failure_mode_ids)
    return [dict(r) for r in cur.fetchall()]

def create_case(conn, title, description):
    cur = conn.execute("INSERT INTO case_header(title, description) VALUES(?,?)", (title, description))
    return cur.lastrowid

def add_case_ppr_binding(conn, case_id, product_id, process_id, resource_id):
    cur = conn.execute(
        "INSERT INTO case_ppr_binding(case_id,product_class_id,process_class_id,resource_class_id) VALUES(?,?,?,?)",
        (case_id, product_id, process_id, resource_id)
    )
    return cur.lastrowid

def add_case_fmea_row(conn, case_id, failure_mode_id, cause_id, effect_id, control_id, severity, occurrence, detection, notes):
    cur = conn.execute(
        """INSERT INTO case_fmea(case_id,failure_mode_id,cause_id,effect_id,control_id,severity,occurrence,detection,notes)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (case_id, failure_mode_id, cause_id, effect_id, control_id, severity, occurrence, detection, notes)
    )
    return cur.lastrowid

def list_cases(conn):
    cur = conn.execute("SELECT * FROM case_header ORDER BY created_at DESC")
    return [dict(r) for r in cur.fetchall()]

def get_case_fmea(conn, case_id):
    cur = conn.execute("SELECT * FROM case_fmea WHERE case_id=?", (case_id,))
    return [dict(r) for r in cur.fetchall()]
