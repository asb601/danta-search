import sys, csv, asyncio, uuid, itertools
sys.path.insert(0,".")
from sqlalchemy import text
from app.core.database import async_session
CID="9a759559-446f-4751-8167-26a174d05de8"
CAT="/Users/bharath/Desktop/projects/G-CHAT-/docs/superpowers/plans/2026-06-11-demo-table-catalog.csv"
# scope/audit cols that are NOT business join keys (data-driven ubiquity also guards)
DENY={"CREATED_BY","LAST_UPDATED_BY","LAST_UPDATE_DATE","CREATION_DATE","LAST_UPDATE_LOGIN",
"ORG_ID","ORGANIZATION_ID","SET_OF_BOOKS_ID","LEDGER_ID","CObjectVERSION_NUMBER","OBJECT_VERSION_NUMBER",
"REQUEST_ID","PROGRAM_ID","PERIOD_NAME","PERIOD_YEAR","PERIOD_NUM","QUARTER_NUM","CURRENCY_CODE",
"BUSINESS_GROUP_ID","CODE_COMBINATION_ID","ATTRIBUTE_CATEGORY"}
MIN_M, MAX_M = 2, 8   # a real FK links few masters; >MAX = ubiquitous scope col -> skip
async def go():
    cat=list(csv.DictReader(open(CAT,encoding="utf-8",errors="replace")))
    async with async_session() as s:
        names={n:i for n,i in (await s.execute(text("SELECT name,id FROM files WHERE container_id=:c"),{"c":CID})).all()}
        masters={fid:lab for fid,lab in (await s.execute(text(
          "SELECT file_id, entity_name FROM semantic_entities WHERE container_id=:c AND is_canonical_master"),{"c":CID})).all()}
        # key -> [(file_id,label,table)] for MASTERS only
        key2m={}
        for r in cat:
            fid=names.get(r["File_Name"])
            if not fid or fid not in masters: continue
            for k in [x.strip().upper() for x in (r["Key_Columns"] or "").split(",") if x.strip()]:
                if k in DENY: continue
                key2m.setdefault(k,[]).append((fid,masters[fid],r["OEBS_Table"]))
        # existing approved pairs (avoid dup)
        approved=0; skipped_ubiq=0; pairs_seen=set()
        for k,ms in key2m.items():
            if not (MIN_M<=len(ms)<=MAX_M): 
                if len(ms)>MAX_M: skipped_ubiq+=1
                continue
            for (fa,la,ta),(fb,lb,tb) in itertools.combinations(ms,2):
                pk=tuple(sorted([fa,fb])+[k])
                if pk in pairs_seen: continue
                pairs_seen.add(pk)
                # flip existing candidate to approved, else insert
                ex=(await s.execute(text("""SELECT id FROM semantic_relationships WHERE container_id=:c
                    AND ((file_a_id=:a AND file_b_id=:b) OR (file_a_id=:b AND file_b_id=:a)) LIMIT 1"""),
                    {"c":CID,"a":fa,"b":fb})).first()
                if ex:
                    await s.execute(text("""UPDATE semantic_relationships SET approval_status='approved', status='active',
                        from_column=:k, to_column=:k WHERE id=:id"""),{"k":k,"id":ex[0]})
                else:
                    await s.execute(text("""INSERT INTO semantic_relationships
                       (id,container_id,file_a_id,file_b_id,from_entity,to_entity,from_column,to_column,
                        relationship_type,approval_status,status,confidence_score,computed_at)
                       VALUES (:id,:c,:a,:b,:ea,:eb,:k,:k,'many_to_one','approved','active',1.0,now())"""),
                       {"id":str(uuid.uuid4()),"c":CID,"a":fa,"b":fb,"ea":la,"eb":lb,"k":k})
                approved+=1
        await s.commit()
        tot=(await s.execute(text("SELECT count(*) FROM semantic_relationships WHERE container_id=:c AND approval_status='approved'"),{"c":CID})).scalar()
        print(f"keys_considered={len(key2m)} skipped_ubiquitous={skipped_ubiq} joins_approved_this_run={approved} total_approved_now={tot}")
        # sample
        rows=(await s.execute(text("""SELECT fa.name,sr.from_column,fb.name FROM semantic_relationships sr
           JOIN files fa ON fa.id=sr.file_a_id JOIN files fb ON fb.id=sr.file_b_id
           WHERE sr.container_id=:c AND sr.approval_status='approved' LIMIT 12"""),{"c":CID})).all()
        for a,k,b in rows: print(f"   {a} . {k} = {b}")
asyncio.run(go())
