import numpy as np
from .repository import get_failure_modes_by_process, get_causes_for_failure, get_effects_for_failure, get_controls_for_failure

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)

def map_text_to_ppr(user_text: str, ontology: list, embedder, topk=3):
    texts = [f"{o['type']}: {o['name']} - {o.get('description','')}" for o in ontology]
    vecs = embedder.encode(texts)
    qvec = embedder.encode(user_text)[0]
    sims = [cosine_sim(qvec, v) for v in vecs]
    idxs = np.argsort(sims)[::-1]
    picks = {"Product": [], "Process": [], "Resource": []}
    for i in idxs:
        typ = ontology[i]["type"]
        if len(picks[typ]) < topk:
            picks[typ].append(ontology[i]["id"])
        if all(len(picks[t]) >= topk for t in picks):
            break
    return picks

def retrieve_candidates(conn, mapped_ppr):
    proc_ids = mapped_ppr.get("Process", []) or []
    failure_modes = get_failure_modes_by_process(conn, proc_ids) if proc_ids else []
    fm_ids = [fm["id"] for fm in failure_modes]
    causes = get_causes_for_failure(conn, fm_ids)
    effects = get_effects_for_failure(conn, fm_ids)
    controls = get_controls_for_failure(conn, fm_ids)
    return {
        "failure_modes": failure_modes,
        "causes": causes,
        "effects": effects,
        "controls": controls
    }
