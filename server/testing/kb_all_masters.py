import sys, csv, asyncio, re
sys.path.insert(0,".")
from sqlalchemy import text
from app.core.database import async_session
CID="9a759559-446f-4751-8167-26a174d05de8"
CAT="/Users/bharath/Desktop/projects/G-CHAT-/docs/superpowers/plans/2026-06-11-demo-table-catalog.csv"
MODS={"AP","AR","GL","PO","PON","RCV","IBY","ZX","XLA","CE","FA","PA","PJI","PJM","MTL","MRP","MSC",
"BOM","WIP","CST","EAM","INV","OE","ONT","WSH","QP","ASO","AS","OKC","OKS","CS","CSI","IEX","JTF",
"PER","PAY","BEN","HR","HXC","HXT","IRC","OTA","FF","FND","HZ","RA","AME"}
SUF={"ALL","F","B","TL","VL","V","S","B1","A","G","X","INT","STG","TMP","ARCHIVE","HIST"}
# module->concept hint when residual is generic
GEN={"HEADER","LINE","DETAIL","DETAILS","HEADERS","LINES","DIST","DISTS","DISTRIBUTIONS","TRX","ALL","REC","RECS"}
MODCON={"PO":"purchase_order","OE":"sales_order","ONT":"sales_order","AP":"payable","AR":"receivable",
"GL":"journal","WSH":"shipment","RCV":"receipt","RA":"receivable"}
def sing(t):
    if len(t)>3 and t.endswith("IES"): return t[:-3]+"Y"
    if len(t)>3 and t.endswith("S") and not t.endswith("SS"): return t[:-1]
    return t
def concept(oebs):
    toks=oebs.upper().split("_")
    mod=toks[0] if toks and toks[0] in MODS else None
    body=toks[1:] if mod else toks[:]
    while body and body[-1] in SUF: body=body[:-1]
    if not body: body=toks[:]  # fallback
    if all(b in GEN for b in body) and mod in MODCON:
        return MODCON[mod]
    return "_".join(sing(b) for b in body).lower() or oebs.lower()
def baserank(oebs):  # lower = barer/base
    toks=oebs.upper().split("_"); mod=1 if toks and toks[0] in MODS else 0
    body=toks[mod:]; nsuf=sum(1 for t in body if t in SUF)
    nbody=len([t for t in body if t not in SUF])
    base_bonus=-1 if oebs.upper().endswith(("_ALL","_B")) else 0
    return (nbody, -nsuf, base_bonus, len(oebs))
async def go():
    cat=list(csv.DictReader(open(CAT,encoding="utf-8",errors="replace")))
    async with async_session() as s:
        names={n:i for n,i in (await s.execute(text("SELECT name,id FROM files WHERE container_id=:c"),{"c":CID})).all()}
        # map catalog -> file_id, build concept groups
        groups={}; pol={}
        for r in cat:
            fn=r["File_Name"]; fid=names.get(fn)
            if not fid: continue
            lab=concept(r["OEBS_Table"]); mod=(r["Oracle_Module"] or "").upper()
            groups.setdefault(lab,[]).append((r["OEBS_Table"],fid))
            p = "vendor" if mod in {"AP","PO","PON","RCV","IBY"} else ("customer" if mod in {"AR","OE","ONT","WSH","HZ","ASO","AS","QP","RA","CS"} else None)
            if p: pol[fid]=p
        # clear masters, then elect bare per group
        await s.execute(text("UPDATE semantic_entities SET is_canonical_master=false, master_for_entity=NULL WHERE container_id=:c"),{"c":CID})
        nset=0
        for lab,mem in groups.items():
            best=min(mem,key=lambda x:baserank(x[0]))
            fid=best[1]
            await s.execute(text("""UPDATE semantic_entities SET is_canonical_master=true, master_for_entity=:l, entity_name=:l
               WHERE container_id=:c AND file_id=:f"""),{"l":lab,"c":CID,"f":fid}); nset+=1
        # polarity human_override
        for fid,p in pol.items():
            await s.execute(text("""UPDATE erp_classifications SET domain_polarity=:p, source='human_override', confidence=1.0
               WHERE container_id=:c AND file_id=:f"""),{"p":p,"c":CID,"f":fid})
        await s.commit()
        print(f"concepts(groups)={len(groups)}  masters_set={nset}  polarity_set={len(pol)}")
        # spot-check key demo tables
        for n in ["AP_INVOICES_ALL.xlsx","PO_HEADERS_ALL.xlsx","AP_SUPPLIERS","AR_TRANSACTIONS_ALL.csv","OE_ORDER_HEADERS_ALL.xlsx","PER_ALL_PEOPLE_F.xlsx","MTL_SYSTEM_ITEMS_B.xls","GL_BALANCES.xlsx"]:
            row=(await s.execute(text("""SELECT se.entity_name, se.is_canonical_master FROM semantic_entities se JOIN files f ON f.id=se.file_id
               WHERE se.container_id=:c AND f.name LIKE :n LIMIT 1"""),{"c":CID,"n":n.split('.')[0]+'%'})).first()
            print(f"   {n:28s} -> {row}")
asyncio.run(go())
