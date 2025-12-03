import streamlit as st

STYLE_CSS = """

<style>
    /* Global page background and content width */
    .stApp {
        background: radial-gradient(circle at top left, #e0f2fe 0, #f4f3ed 55%, #e5e7eb 100%);
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Typography */
    body, h1, h2, h3, h4, h5, h6, p {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #111827;
    }
    h1 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    h2 {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

        /* Primary buttons (Generate FMEA, etc.) */
    .stButton > button {
        background: linear-gradient(135deg, #0f766e, #22c55e);
        color: #ffffff;
        border: none;
        border-radius: 999px !important; /* ensure pill */
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        font-size: 0.92rem;
        box-shadow: 0 6px 18px rgba(15, 118, 110, 0.35);
        cursor: pointer;
        transition: background-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
        }
    .stButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.45);
    }

    /* Make sure primary and secondary kinds both use our colors, not plain white */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0f766e, #22c55e);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        filter: brightness(1.06);
    }

    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background: #e5f2ff;              /* light tint to avoid plain white */
        color: #0f172a;
        border-radius: 999px;
        border: 1px solid #cbd5f5;
        box-shadow: none;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #dbeafe;
        border-color: #0f766e;
    }

    /* Card-like containers (unchanged) */
    .block-container > div {
        border-radius: 16px;
    }

    /* Tabs: spacing from top + bold labels so they read clearly as tabs */
    [data-testid="stTabs"] {
        margin-top: 1.5rem;      /* space below Streamlit header bar */
        margin-bottom: 1.5rem;
    }
    [data-testid="stTabs"] button[role="tab"] {
        font-weight: 700;        /* bold tab labels */
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }

        /* Make tab labels bold (including emojis) */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-weight: 700 !important;
    }

    /* Optional: slightly darker color for the active tab */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"]
        [data-testid="stMarkdownContainer"] p {
        color: #0f172a;
    }


    /* AG-Grid refinements (as before) */
    .ag-theme-alpine .ag-header {
        background-color: #0f172a !important;
    }
    .ag-theme-alpine .ag-header-cell,
    .ag-theme-alpine .ag-header-cell-label {
        color: #f9fafb !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .ag-theme-alpine .ag-cell {
        font-size: 14px !important;
        white-space: nowrap;       /* single line only */
        overflow: hidden;          /* hide overflow if any */
        text-overflow: ellipsis;   /* show ... if column is still too narrow */
        font-size: 14px !important;
    }
    .ag-theme-alpine .ag-row-hover {
        background-color: #ecfeff !important;
    }

        /* Streamlit dataframes (st.dataframe) */
    [data-testid="stDataFrame"] > div {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;   /* light grey table card */
    }
    [data-testid="stDataFrame"] table {
        font-size: 0.9rem;
    }
    [data-testid="stDataFrame"] thead tr {
        background-color: #0f172a;   /* dark header */
        color: #f9fafb;
        font-weight: 600;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f3f4f6;   /* zebra striping */
    }
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #e0f2fe;   /* hover row */
    }

    /* AgGrid overall card and rows */
    .ag-theme-alpine {
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        background-color: #f9fafb;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .ag-theme-alpine .ag-row:nth-child(even) {
        background-color: #f3f4f6;
    }
    .ag-theme-alpine .ag-row:hover {
        background-color: #e0f2fe !important;
    }

        /* Streamlit dataframes (st.dataframe) */
    div[data-testid="stDataFrame"] > div {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e5e7eb;
        background-color: #f9fafb;
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.9rem;
    }
    div[data-testid="stDataFrame"] thead tr {
        background-color: #0f172a;
        color: #f9fafb;
        font-weight: 600;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f3f4f6;
    }
    div[data-testid="stDataFrame"] tbody tr:hover {
        background-color: #e0f2fe;
    }


    /* Description textarea */
    .stTextArea textarea {
        border: 1px solid #d4d4d8 !important;
        border-radius: 14px !important;
        background-color: #f9fafb !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus-visible {
        outline: none !important;
        border: 1px solid #0f766e !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.4);
        background-color: #ffffff !important;
    }

        /* Text inputs (Case title, etc.) */


    /* "Select LLM" selectboxes – softer fill, no harsh white */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 999px !important;
        border: 1px solid #d4d4d8 !important;
        background: linear-gradient(135deg, #eef2ff, #f9fafb) !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        padding: 2px 8px;
    }

    /* Remove any inner white blocks so the whole pill looks consistent */
    .stSelectbox > div[data-baseweb="select"] > div {
        background-color: transparent !important;
    }

    /* Hover / focus accent in your teal theme */
    .stSelectbox > div[data-baseweb="select"]:hover,
    .stSelectbox > div[data-baseweb="select"]:focus-within {
        border-color: #0f766e !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.3);
    }

    .stSelectbox > div[data-baseweb="select"] > div {
    border-radius: 999px !important; /* force inner div to pill */
    }
    
    .stSelectbox [data-baseweb="select"] div[role="combobox"] {
    border-radius: 999px !important;
    background-color: transparent !important;
    }
    
    .stSelectbox [data-baseweb="tag"] {
    border-radius: 999px !important;
    }

    /* ---------- TIGHT PILL / CARD FOR INPUTS & SELECTS ---------- */
    
    
    /* Textareas (description, PPR fields) – remove outer white frame */
    .stTextArea > div {
        border-radius: 14px !important;
        background-color: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    
    .stTextArea textarea {
        border: 1px solid #d4d4d8 !important;
        border-radius: 14px !important;
        background-color: #f9fafb !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    
    /* Select LLM – remove inner white band & tighten control */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 999px !important;
        border: 1px solid #d4d4d8 !important;
        background: linear-gradient(135deg, #eef2ff, #f9fafb) !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        padding: 2px 8px !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] {
        margin: 0 !important;
        padding: 4px 10px !important;
        border-radius: 999px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] > div {
        background-color: transparent !important;
        border-radius: 999px !important;
    }
    
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        background-color: transparent !important;
    }
    
    /* ---- FIX: Select LLM pill ---- */
    /* Outer pill container */
    .stSelectbox > div[data-baseweb="select"] {
        border-radius: 999px !important;
        border: 1px solid #0f766e !important;
        background: linear-gradient(135deg, #eef2ff, #f9fafb) !important;
        padding: 0 !important;
    }
    
    /* Inner combobox (removes white band) */
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] {
        margin: 2px !important;
        padding: 6px 12px !important;
        border-radius: 999px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] div[role="combobox"] > div {
        background-color: transparent !important;
        border-radius: 999px !important;
    }
    
    /* Final: Case title matches Case description */

    /* Universal single-line input style – matches Case description card */

    /* Reset any wrapper rounding/border */
    div[data-testid="stTextInput"] > div > div {
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    
    /* Actual input draws the border and radius */
    div[data-testid="stTextInput"] input {
        border: 1px solid #d4d4d8 !important;
        border-radius: 14px !important;             /* same as textarea */
        background-color: #f9fafb !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
    }
    
    /* Optional subtle focus */
    div[data-testid="stTextInput"] input:focus-visible {
        outline: none !important;
        border: 1px solid #0f766e33 !important;
        box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.12);
        background-color: #ffffff !important;
    }

</style>
"""

AGGRID_CUSTOM_CSS = {
    # Overall grid background + border
    ".ag-root-wrapper": {
        "border-radius": "14px",
        "border": "1px solid #e5e7eb",
        "box-shadow": "0 4px 14px rgba(15, 23, 42, 0.06)",
        "overflow": "hidden",
        "background-color": "#f9fafb",
    },
    # Header row
    ".ag-header": {
        "background-color": "#0f172a",
        "color": "#f9fafb",
        "font-weight": "600",
        "font-size": "14px",
    },
    ".ag-header-cell-label": {
        "color": "#f9fafb",
        "font-weight": "600",
        "font-size": "14px",
    },
    # Body rows: zebra and hover
    ".ag-row:nth-child(even)": {
        "background-color": "#f3f4f6",
    },
    ".ag-row-hover": {
        "background-color": "#e0f2fe !important",
    },
}


def apply_global_styles() -> None:
    """Inject the global CSS into the Streamlit app."""
    st.markdown(STYLE_CSS, unsafe_allow_html=True)
