"""Comprehensive live end-to-end test of the running Anodyne local instance.

Exercises every major API flow against http://localhost:8001 with a real
Keycloak token, and prints a PASS/FAIL table. Read-only where possible; creates
a few throwaway datasets. Run: uv run python e2e_live.py
"""

from __future__ import annotations

import time
import httpx

API = "http://localhost:8001"
KC = "http://localhost:8080/realms/anodyne/protocol/openid-connect/token"
results: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def token() -> str:
    r = httpx.post(
        KC,
        data={
            "grant_type": "password",
            "client_id": "anodyne",
            "client_secret": "dev-only-anodyne-client-secret",
            "username": "demo@anodyne.dev",
            "password": "demo",
            "scope": "openid",
        },
        timeout=30,
    )
    return r.json()["access_token"]


def poll(c: httpx.Client, path: str, terminal: set[str], field: str = "status", timeout=240):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = c.get(path)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        last = r.json().get(field)
        if last in terminal:
            return last, r.json()
        time.sleep(3)
    return last, "TIMEOUT"


def main() -> None:
    tok = token()
    rec("auth: keycloak token", bool(tok))
    c = httpx.Client(base_url=API, headers={"Authorization": f"Bearer {tok}"}, timeout=60)

    # --- basic reads ---
    me = c.get("/me")
    rec("GET /me", me.status_code == 200 and "tenant_id" in me.json(), f"tenant={me.json().get('tenant_id','')[:8]}")
    for p in ["/models", "/templates", "/datasets", "/image-providers", "/audio-providers", "/video-providers", "/reviews"]:
        r = c.get(p)
        rec(f"GET {p}", r.status_code == 200, f"HTTP {r.status_code}, n={len(r.json()) if isinstance(r.json(),list) else '?'}")

    models = c.get("/models").json()
    default_model = models[0]["id"] if models else None

    # --- LLM invoke (each registered model) ---
    for m in models:
        r = c.post("/llm/invoke", json={"model_config_id": m["id"], "messages": [{"role": "user", "content": "say pong"}]})
        rec(f"POST /llm/invoke [{m['provider']}]", r.status_code == 200 and "content" in r.json(), f"HTTP {r.status_code}")

    # --- dataset from description (schema proposal via LLM) ---
    r = c.post("/datasets", json={"name": "e2e desc", "description": "online shoppers with age and country", "target_rows": 10})
    ok = r.status_code == 201 and len(r.json().get("fields", [])) > 0
    rec("POST /datasets (description->schema)", ok, f"HTTP {r.status_code}, fields={len(r.json().get('fields',[])) if r.status_code==201 else 0}")
    ddesc = r.json().get("id") if r.status_code == 201 else None
    if ddesc:
        rec("GET /datasets/{id}", c.get(f"/datasets/{ddesc}").status_code == 200)
        rec("PATCH /datasets/{id}", c.patch(f"/datasets/{ddesc}", json={"target_rows": 15}).status_code in (200, 204))

    # --- template -> generate -> poll -> version ---
    r = c.post("/datasets/from-template", json={"template_key": "customers", "name": "e2e tmpl", "target_rows": 25})
    dsid = r.json().get("id") if r.status_code == 201 else None
    rec("POST /datasets/from-template", bool(dsid), f"HTTP {r.status_code}")
    version_id = None
    if dsid:
        jr = c.post(f"/datasets/{dsid}/generate", json={"seed": 7})
        jid = jr.json().get("id")
        st, _ = poll(c, f"/jobs/{jid}", {"succeeded", "failed"})
        rec("generate -> job succeeded", st == "succeeded", f"status={st}")
        vs = c.get(f"/datasets/{dsid}/versions")
        vlist = vs.json() if vs.status_code == 200 else []
        version_id = vlist[-1]["id"] if vlist else None
        rec("GET /datasets/{id}/versions", bool(version_id), f"n={len(vlist)}")

    # --- export ---
    if version_id:
        for fmt in ["csv", "json", "parquet"]:
            r = c.post(f"/datasets/{dsid}/versions/{version_id}/export", json={"format": fmt})
            url = (r.json() or {}).get("download_url") or (r.json() or {}).get("url") if r.status_code in (200, 201) else None
            ok = r.status_code in (200, 201)
            # try fetching the presigned url
            fetched = ""
            if url:
                try:
                    fetched = f", dl={httpx.get(url, timeout=30).status_code}"
                except Exception as e:
                    fetched = f", dl-err={type(e).__name__}"
            rec(f"POST export [{fmt}]", ok, f"HTTP {r.status_code}{fetched}")

    # --- perturbation ---
    if version_id:
        r = c.post(f"/datasets/{dsid}/versions/{version_id}/perturb", json={"family": "noise", "intensity": 0.2, "seed": 3})
        pjid = (r.json() or {}).get("id") or (r.json() or {}).get("job_id")
        rec("POST perturb", r.status_code in (200, 201, 202) and bool(pjid), f"HTTP {r.status_code}")
        if pjid:
            st, _ = poll(c, f"/perturbation-jobs/{pjid}", {"succeeded", "failed"}, timeout=180)
            rec("perturb -> job succeeded", st == "succeeded", f"status={st}")

    # --- evaluation (MoE LLM-as-a-Judge) ---
    if version_id:
        r = c.post(f"/datasets/{dsid}/versions/{version_id}/evaluate", json={"seed": 1, "sample_rows": 8})
        rid = (r.json() or {}).get("id") or (r.json() or {}).get("run_id")
        rec("POST evaluate", r.status_code in (200, 201, 202) and bool(rid), f"HTTP {r.status_code}")
        if rid:
            st, _ = poll(c, f"/evaluations/{rid}", {"succeeded", "failed", "completed"}, timeout=300)
            rec("evaluate -> run done", st in ("succeeded", "completed"), f"status={st}")
            rep = c.get(f"/evaluations/{rid}/report")
            rec("GET evaluation report", rep.status_code == 200, f"HTTP {rep.status_code}")

    # --- HITL: require_review parks, approve resumes ---
    if dsid:
        jr = c.post(f"/datasets/{dsid}/generate", json={"seed": 9, "require_review": True})
        jid2 = jr.json().get("id")
        time.sleep(3)
        reviews = c.get("/reviews", params={"status": "pending"}).json()
        rev = reviews[-1] if isinstance(reviews, list) and reviews else None
        rec("require_review creates pending review", bool(rev), f"pending={len(reviews) if isinstance(reviews,list) else '?'}")
        if rev:
            dr = c.post(f"/reviews/{rev['id']}/decision", json={"decision": "approve"})
            rec("POST /reviews/{id}/decision approve", dr.status_code in (200, 204), f"HTTP {dr.status_code}")
            st, _ = poll(c, f"/jobs/{jid2}", {"succeeded", "failed"}, timeout=180)
            rec("approved review -> job succeeded", st == "succeeded", f"status={st}")

    # --- annotations + feedback ---
    if version_id:
        a = c.post(f"/datasets/{dsid}/versions/{version_id}/annotations", json={"row_index": 0, "label": "outlier", "tags": ["review"], "comment": "check"})
        aid = (a.json() or {}).get("id")
        rec("POST annotation", a.status_code in (200, 201) and bool(aid), f"HTTP {a.status_code}")
        g = c.get(f"/datasets/{dsid}/versions/{version_id}/annotations")
        rec("GET annotations", g.status_code == 200 and len(g.json()) >= 1, f"n={len(g.json()) if g.status_code==200 else '?'}")
        if aid:
            rec("DELETE annotation", c.delete(f"/annotations/{aid}").status_code in (200, 204))
        fb = c.post("/feedback", json={"target_type": "dataset_version", "target_id": version_id, "rating": 5, "comment": "great"})
        rec("POST feedback", fb.status_code in (200, 201), f"HTTP {fb.status_code}")

    # --- summary ---
    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{len(results)} passed")
    fails = [n for n, ok, d in results if not ok]
    if fails:
        print("FAILURES:")
        for n, ok, d in results:
            if not ok:
                print(f"  - {n} ({d})")
    else:
        print("ALL GREEN")


if __name__ == "__main__":
    main()
