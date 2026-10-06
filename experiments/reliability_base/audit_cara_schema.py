"""M34 schema/metadata audit; never evaluates models or loads activity values."""
import argparse,csv,hashlib,io,json,time,zipfile
from collections import Counter
from pathlib import Path

def category(name):
    s=name.lower().replace('_',' ')
    if any(x in s for x in ('date','year','timestamp')): return 'date'
    if 'assay' in s or s=='task id': return 'assay'
    if s=='value type' or any(x in s for x in ('relation','standard type','standard units','endpoint','measurement type')): return 'endpoint'
    if 'target' in s and 'sequence' not in s: return 'target'
    return None

def audit(archive):
    start=time.monotonic(); raw=archive.read_bytes()
    out={'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive_md5':hashlib.md5(raw).hexdigest(),'archive_bytes':len(raw),'tables':[]}
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if '/Task/' not in name or not name.lower().endswith(('.csv','.tsv')): continue
            with z.open(name) as f:
                reader=csv.reader(io.TextIOWrapper(f,encoding='utf-8-sig'),delimiter='\t' if name.endswith('.tsv') else ',')
                header=next(reader); selected={i:category(h) for i,h in enumerate(header) if category(h)}
                counts={i:Counter() for i in selected}; missing={i:0 for i in selected}; rows=0
                for row in reader:
                    if len(row)!=len(header): raise ValueError(f'row width mismatch: {name}:{rows+2}')
                    rows+=1
                    for i in selected:
                        value=row[i].strip()
                        if not value or value.lower() in ('nan','none','null'): missing[i]+=1
                        else: counts[i][value]+=1
                fields={header[i]:{'category':selected[i],'nonmissing':rows-missing[i],'missing':missing[i],'unique':len(counts[i]),'top':counts[i].most_common(12),'requested_target_rows':{t:counts[i][t] for t in ('CHEMBL234','CHEMBL244','CHEMBL4792')} if selected[i]=='target' else None} for i in selected}
                out['tables'].append({'member':name,'rows':rows,'headers':header,'metadata':fields})
    out['seconds']=time.monotonic()-start
    return out

def selfcheck():
    assert category('pChEMBL Value') is None and category('Smiles') is None
    assert category('Document Year')=='date' and category('Assay ChEMBL ID')=='assay'
    assert category('Standard Type')=='endpoint' and category('Target Sequence') is None
    assert category('Target ChEMBL ID')=='target' and category('Value Type')=='endpoint'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path);p.add_argument('--selfcheck',action='store_true');a=p.parse_args()
    selfcheck()
    if not a.selfcheck:
        assert a.archive and a.output
        a.output.write_text(json.dumps(audit(a.archive),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('selfcheck passed' if a.selfcheck else str(a.output))
