# export.py

import io
import datetime as dt
import pandas as pd
from typing import Dict, List, Any
import xml.etree.ElementTree as ET

def to_pretty_excel_bytes(
    case_id: int | None,
    case_title: str | None,
    case_desc: str | None,
    model_label: str | None,
    timing_fmea_ms: int | None,
    timing_fpr_ms_ppr: int | None,      # renamed param to avoid ambiguous naming collision
    timing_fmea_kb_ms: int | None,
    timing_fmea_llm_ms: int | None,
    ppr: Dict[str, List[str]],
    fmea_rows_df: pd.DataFrame
) -> bytes:
    """
    Build a polished multi-sheet XLSX:
    - Summary: case meta + split timings (KB/LLM/Overall) + PPR collapsed to text
    - PPR: each pillar one column, one value per row
    - FMEA: edited grid with styles
    """
    created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Normalize PPR lists
    ppr = ppr or {}
    inputs = ppr.get("input_products") or []
    prods  = ppr.get("products") or []
    procs  = ppr.get("processes") or []
    ress   = ppr.get("resources") or []

    # Summary sheet as single-row DataFrame
    summary_dict = {
        "Case ID": [case_id if case_id is not None else ""],
        "Case Title": [case_title or ""],
        "Case Description": [case_desc or ""],
        "Created At": [created_at],
        "Model": [model_label or ""],
        "KB Time (ms)": [timing_fmea_kb_ms if timing_fmea_kb_ms is not None else ""],
        "LLM Time (ms)": [timing_fmea_llm_ms if timing_fmea_llm_ms is not None else ""],
        "FMEA Total Time (ms)": [timing_fmea_ms if timing_fmea_ms is not None else ""],
        "PPR Time (ms)": [timing_fpr_ms_ppr if timing_fpr_ms_ppr is not None else ""],
        "Input Products": [", ".join(inputs)],
        "Output Products": [", ".join(prods)],
        "Processes": [", ".join(procs)],
        "Resources": [", ".join(ress)],
    }
    df_summary = pd.DataFrame(summary_dict)

    # PPR sheet (column-wise, one item per row)
    max_len = max(len(inputs), len(prods), len(procs), len(ress), 1)
    def pad(lst: List[str], n: int) -> List[str]:
        lst = lst or []
        return lst + [""] * (n - len(lst))
    df_ppr = pd.DataFrame({
        "Input Products": pad(inputs, max_len),
        "Output Products": pad(prods,  max_len),
        "Processes":       pad(procs,  max_len),
        "Resources":       pad(ress,   max_len),
    })

    # FMEA sheet
    df_fmea = fmea_rows_df.copy()
    # Convert NaN to empty for nicer display
    df_fmea = df_fmea.where(pd.notna(df_fmea), "")

    # Write with styles
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        # Summary
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        wb  = writer.book
        ws1 = writer.sheets["Summary"]

        # Styles
        head_fmt = wb.add_format({"bold": True, "bg_color": "#1f4e79", "font_color": "white", "border": 1})
        text_wrap = wb.add_format({"text_wrap": True})
        normal = wb.add_format({"border": 1})

        # Header style + width
        for col_idx, _ in enumerate(df_summary.columns):
            ws1.set_column(col_idx, col_idx, 28, text_wrap)
        ws1.freeze_panes(1, 0)
        ws1.set_row(0, 22, head_fmt)

        # PPR
        df_ppr.to_excel(writer, index=False, sheet_name="PPR")
        ws2 = writer.sheets["PPR"]
        for col_idx, col_name in enumerate(df_ppr.columns):
            width = max(18, min(60, max(len(str(col_name)), int(df_ppr[col_name].astype(str).str.len().max()) + 2)))
            ws2.set_column(col_idx, col_idx, width, normal)
        ws2.freeze_panes(1, 0)
        ws2.set_row(0, 22, head_fmt)

        # FMEA
        df_fmea.to_excel(writer, index=False, sheet_name="FMEA")
        ws3 = writer.sheets["FMEA"]
        for col_idx, col_name in enumerate(df_fmea.columns):
            # Use 90th percentile width for reasonable column sizing
            try:
                maxlen = int(df_fmea[col_name].astype(str).str.len().quantile(0.9))
            except Exception:
                maxlen = 20
            width = max(14, min(70, max(len(str(col_name)), maxlen + 2)))
            ws3.set_column(col_idx, col_idx, width, normal)
        ws3.freeze_panes(1, 0)
        ws3.set_row(0, 22, head_fmt)

    return bio.getvalue()


def to_structured_xml_bytes(
    case_id: int | None,
    case_title: str | None,
    case_desc: str | None,
    model_label: str | None,
    timing_fmea_ms: int | None,
    timing_ppr_ms: int | None,
    timing_fmea_kb_ms: int | None,
    timing_fmea_llm_ms: int | None,
    ppr: Dict[str, List[str]],
    fmea_rows: List[Dict[str, Any]],
) -> bytes:
    """
    Build structured XML:
    <FMEAExport created_at="...">
      <Case id="" title="" description="" model="" kb_ms="" llm_ms="" fmea_ms="" ppr_ms=""/>
      <PPR>
        <Inputs><Item>...</Item></Inputs>
        <Products>...</Products>
        <Processes>...</Processes>
        <Resources>...</Resources>
      </PPR>
      <FMEA>
        <Row> ... </Row>
        ...
      </FMEA>
    </FMEAExport>
    """
    created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    root = ET.Element("FMEAExport", attrib={"created_at": created_at})

    # Case meta
    case_el = ET.SubElement(root, "Case")
    if case_id is not None: case_el.set("id", str(case_id))
    if case_title: case_el.set("title", case_title)
    if case_desc: case_el.set("description", case_desc)
    if model_label: case_el.set("model", model_label)
    if timing_fmea_kb_ms is not None: case_el.set("kb_ms", str(timing_fmea_kb_ms))
    if timing_fmea_llm_ms is not None: case_el.set("llm_ms", str(timing_fmea_llm_ms))
    if timing_fmea_ms is not None: case_el.set("fmea_ms", str(timing_fmea_ms))
    if timing_ppr_ms is not None: case_el.set("ppr_ms", str(timing_ppr_ms))

    # PPR
    ppr = ppr or {}
    ppr_el = ET.SubElement(root, "PPR")

    def _items(parent_name: str, values: List[str] | None):
        cont = ET.SubElement(ppr_el, parent_name)
        for v in values or []:
            if v is None: 
                continue
            item = ET.SubElement(cont, "Item")
            item.text = str(v)

    _items("Inputs",   ppr.get("input_products"))
    _items("Products", ppr.get("products"))
    _items("Processes",ppr.get("processes"))
    _items("Resources",ppr.get("resources"))

    # FMEA rows
    fmea_el = ET.SubElement(root, "FMEA")
    for row in fmea_rows or []:
        r = ET.SubElement(fmea_el, "Row")
        for k, v in row.items():
            el = ET.SubElement(r, str(k))
            el.text = "" if v is None else str(v)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
