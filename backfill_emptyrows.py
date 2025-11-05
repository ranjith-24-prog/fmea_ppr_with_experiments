from supabase import create_client
from backend.llm import Embeddings
import os

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
emb = Embeddings()

def list_names(table, case_id):
    return [r["name"] for r in sb.table(table).select("name").eq("case_id", case_id).execute().data or []]

rows = sb.table("kb_index").select("case_id, products_text, processes_text, resources_text").execute().data or []
fixed = 0
for r in rows:
    if (r.get("products_text") or r.get("processes_text") or r.get("resources_text")):
        continue
    cid = r["case_id"]
    prods = list_names("products", cid)
    procs = list_names("processes", cid)
    ress  = list_names("resources", cid)
    prod_txt = ", ".join(sorted(set([x for x in prods if x])))
    proc_txt = ", ".join(sorted(set([x for x in procs if x])))
    res_txt  = ", ".join(sorted(set([x for x in ress if x])))
    sb.table("kb_index").update({
        "products_text": prod_txt,
        "processes_text": proc_txt,
        "resources_text": res_txt,
        "prod_vec": emb.embed(prod_txt) if prod_txt else None,
        "proc_vec": emb.embed(proc_txt) if proc_txt else None,
        "res_vec":  emb.embed(res_txt)  if res_txt  else None,
    }).eq("case_id", cid).execute()
    fixed += 1
print("Fixed kb_index rows:", fixed)
