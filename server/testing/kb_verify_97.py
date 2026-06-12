import sys, csv, asyncio, re
sys.path.insert(0,".")
from app.core.database import async_session
CID="9a759559-446f-4751-8167-26a174d05de8"
CSVP="/Users/bharath/Desktop/projects/G-CHAT-/docs/superpowers/plans/2026-06-11-demo-prompts.csv"
OUT="/Users/bharath/Desktop/projects/G-CHAT-/docs/superpowers/plans/2026-06-12-demo-97-scorecard.md"
rows=list(csv.DictReader(open(CSVP,encoding="utf-8",errors="replace")))
def tabs(s): return [t.strip().upper() for t in re.split(r"[,&+]| and | join ", s or "", flags=re.I) if t.strip()]
async def go():
    from app.agent.graph.graph import run_agent_query
    fh=open(OUT,"w"); 
    def w(s): fh.write(s+"\n"); fh.flush()
    w(f"# Demo 97-prompt scorecard (container 9a759559) — {len(rows)} prompts\n")
    w("| ID | verdict | route | rc | want | hit | domain | complexity | prompt |")
    w("|--|--|--|--|--|--|--|--|--|")
    nhit=nnav=nabs=0
    for r in rows:
        q=r["NLP Prompt (what the user types)"]; prim=tabs(r["Primary Table(s)"]); dom=r["Domain"]; cx=r["Complexity"]
        try:
            async with async_session() as db:
                out=await run_agent_query(q, db, is_admin=True, container_id=CID, user_id="v97")
            route=out.get("route"); fu=out.get("files_used") or []; rc=out.get("row_count")
            hit=[t for t in prim if any(t in str(x).upper() for x in fu)]
            ok=bool(hit) and (rc or 0)>0
            v="HIT" if ok else ("ABSTAIN" if route=="navigator_clarify" or (rc or 0)==0 else "MISS")
            nhit+=ok; nnav+=(route=="navigator"); nabs+=(v=="ABSTAIN")
            w(f"| {r['ID']} | {v} | {route} | {rc} | {','.join(prim)} | {','.join(hit) or '-'} | {dom} | {cx} | {q[:60]} |")
        except Exception as e:
            w(f"| {r['ID']} | ERROR | - | - | {','.join(prim)} | - | {dom} | {cx} | {str(e)[:50]} |")
    w(f"\n**SUMMARY: {nhit}/{len(rows)} right-table hits | navigator-routed={nnav} | abstain/no-data={nabs}**")
    fh.close()
asyncio.run(go())
print("DONE ->", OUT)
