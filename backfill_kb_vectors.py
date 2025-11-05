# backfill_kb_vectors.py
import os
from supabase import create_client
from backend.llm import Embeddings

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_ANON_KEY"]
sb = create_client(url, key)
emb = Embeddings()

def embed_or_none(txt):
    txt = (txt or "").strip()
    return emb.embed(txt) if txt else None

rows = sb.table("kb_index").select("case_id, products_text, processes_text, resources_text").execute().data or []
for r in rows:
    case_id = r["case_id"]
    prod_vec = embed_or_none(r.get("products_text"))
    proc_vec = embed_or_none(r.get("processes_text"))
    res_vec  = embed_or_none(r.get("resources_text"))
    sb.table("kb_index").update({
        "prod_vec": prod_vec,
        "proc_vec": proc_vec,
        "res_vec":  res_vec
    }).eq("case_id", case_id).execute()
print(f"Updated {len(rows)} kb_index rows.")
