from .repository import create_case, add_case_ppr_binding, add_case_fmea_row

def persist_case(conn, title, description, selected_product_id, selected_process_id, selected_resource_id, fmea_rows):
    case_id = create_case(conn, title, description)
    add_case_ppr_binding(conn, case_id, selected_product_id, selected_process_id, selected_resource_id)
    for r in fmea_rows:
        add_case_fmea_row(
            conn,
            case_id,
            r.get("failure_mode_id"),
            r.get("cause_id"),
            r.get("effect_id"),
            r.get("control_id"),
            r.get("severity"),
            r.get("occurrence"),
            r.get("detection"),
            r.get("notes","")
        )
    return case_id
