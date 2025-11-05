from .db import get_conn

def ensure_class(conn, name, type_, parent_id=None, description=None):
    cur = conn.execute(
        "SELECT id FROM ontology_class WHERE name=? AND type=?",
        (name, type_)
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO ontology_class(name,type,parent_id,description) VALUES(?,?,?,?)",
        (name, type_, parent_id, description)
    )
    return cur.lastrowid

def ensure_failure_mode(conn, name, desc=None, process_class_id=None):
    cur = conn.execute("SELECT id FROM risk_failure_mode WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_failure_mode(name,description,process_class_id) VALUES(?,?,?)",
        (name, desc, process_class_id)
    )
    return cur.lastrowid

def ensure_cause(conn, name, desc=None, resource_class_id=None):
    cur = conn.execute("SELECT id FROM risk_cause WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_cause(name,description,resource_class_id) VALUES(?,?,?)",
        (name, desc, resource_class_id)
    )
    return cur.lastrowid

def ensure_effect(conn, name, desc=None, product_class_id=None):
    cur = conn.execute("SELECT id FROM risk_effect WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_effect(name,description,product_class_id) VALUES(?,?,?)",
        (name, desc, product_class_id)
    )
    return cur.lastrowid

def ensure_control(conn, name, desc=None):
    cur = conn.execute("SELECT id FROM risk_control WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row["id"]
    cur = conn.execute(
        "INSERT INTO risk_control(name,description) VALUES(?,?)",
        (name, desc)
    )
    return cur.lastrowid

def link(conn, table, a, b):
    conn.execute(f"INSERT OR IGNORE INTO {table} VALUES(?,?)", (a,b))

def seed():
    with get_conn() as conn:
        # PPR ontology
        prod = ensure_class(conn, "Welded Product", "Product", None, "Joined product from welding")
        welded_copper = ensure_class(conn, "Welded Copper", "Product", prod)
        welded_aluminium = ensure_class(conn, "Welded Aluminium", "Product", prod)
        consumables = ensure_class(conn, "Consumables", "Product", None)
        shielding_gas = ensure_class(conn, "Shielding Gas", "Product", consumables)
        aluminium_profiles = ensure_class(conn, "Aluminium Profiles", "Product", welded_aluminium)

        proc_joining = ensure_class(conn, "Joining", "Process", None)
        welding = ensure_class(conn, "Welding", "Process", proc_joining)
        laser_welding = ensure_class(conn, "Laser Welding", "Process", welding)

        res_machine = ensure_class(conn, "Machine", "Resource", None)
        laser_system = ensure_class(conn, "Laser System", "Resource", res_machine)

        # Risks (from your paper examples)
        fm_overpower = ensure_failure_mode(conn, "Weld Overpowering", "Excessive laser power", laser_welding)
        cause_miscal = ensure_cause(conn, "Miscalibrated Laser", "Calibration drift", laser_system)
        cause_mispower = ensure_cause(conn, "Misconfigured Laser Power", "Wrong power setpoint", laser_system)
        effect_underpen = ensure_effect(conn, "Underpenetration", "Insufficient weld penetration", welded_copper)
        effect_inacc_dim = ensure_effect(conn, "Inaccurate Weld Dimension", "Dimensional deviations", prod)
        cause_cont_gas = ensure_cause(conn, "Contaminated Shielding Gas", "Impurities in gas", None)

        control_power_check = ensure_control(conn, "Laser Power Verification", "Routine check of setpoints")
        control_gas_quality = ensure_control(conn, "Gas Purity Check", "Analyze gas purity")

        # Links between risk entities
        link(conn, "failure_cause", fm_overpower, cause_miscal)
        link(conn, "failure_cause", fm_overpower, cause_mispower)
        link(conn, "failure_effect", fm_overpower, effect_underpen)
        link(conn, "failure_effect", fm_overpower, effect_inacc_dim)
        link(conn, "failure_control", fm_overpower, control_power_check)

        conn.commit()
