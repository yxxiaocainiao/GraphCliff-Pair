"""Structure/field-only preflight; neither joint-MMP split nor model experiment."""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import subprocess
import time

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from run_phase_a import ROOT, save, sha


def summarize(groups):
    crossing=[v for v in groups.values() if len(set(v))>1]
    pairs={}
    for values in crossing:
        for a,b in itertools.combinations(sorted(set(values)),2):
            key=a+'|'+b;pairs[key]=pairs.get(key,0)+1
    return dict(group_count=len(groups),cross_role_groups=len(crossing),cross_role_rows=sum(map(len,crossing)),
                largest_group_rows=max(map(len,groups.values()),default=0),shared_groups_by_role_pair=pairs)


def self_check():
    assert MurckoScaffold.MurckoScaffoldSmiles(smiles='Cc1ccccc1')==MurckoScaffold.MurckoScaffoldSmiles(smiles='Oc1ccccc1')
    assert MurckoScaffold.MurckoScaffoldSmiles(smiles='CCC')==''
    assert summarize({'a':['fit','evaluation'],'b':['fit']})['cross_role_groups']==1
    assert summarize({})['largest_group_rows']==0
    print('PASS: exact scaffold, empty-scaffold distinction, role overlap counts')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_check:self_check();return
    assert args.output is not None and not args.output.exists(),'Refuse overwrite'
    pp=ROOT/'docs/research/aidd_plan_recheck_20261006/protocol.json'
    plan=json.loads(pp.read_text(encoding='utf-8'))
    protected={(Path(p) if Path(p).is_absolute() else ROOT/p):h for p,h in plan['inputs'].items()}
    protected[pp]=sha(pp);protected[Path(__file__)]=sha(Path(__file__))
    assert all(sha(p)==h for p,h in protected.items())
    args.output.mkdir(parents=True)
    tick=time.monotonic();state=dict(status='running',attempts=1,fits=0,checkpoint_predictions=0,official_test_labels_loaded=False,
             code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(args.output/'execution.json',state)
    try:
        results=[]
        for task in plan['tasks']:
            raw=Path(plan['data_root'])/(task+'.csv')
            with raw.open(encoding='utf-8-sig',newline='') as stream:header=next(csv.reader(stream))
            roles=json.loads((ROOT/plan['partitions']/(task+'.json')).read_text(encoding='utf-8'))['roles']
            role_by_id={i:role for role,ids in roles.items() for i in ids}
            assert len(role_by_id)==sum(map(len,roles.values()))
            ids=set(role_by_id)
            f=pd.read_csv(raw,usecols=['smiles','split'],skiprows=lambda i:i>0 and i-1 not in ids)
            assert len(f)==len(ids) and f['split'].eq('train').all()
            canonical=defaultdict(list);scaffolds=defaultdict(list);empty=0
            for source,row in zip(sorted(ids),f.itertuples(index=False)):
                mol=Chem.MolFromSmiles(row.smiles);assert mol is not None
                canonical[Chem.MolToSmiles(mol,canonical=True)].append(role_by_id[source])
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
                if scaffold:scaffolds[scaffold].append(role_by_id[source])
                else:empty+=1
            results.append(dict(dataset=task,header=header,selected_rows=len(f),role_rows={k:len(v) for k,v in roles.items()},
                canonical=summarize(canonical),nonempty_murcko=summarize(scaffolds),empty_scaffold_rows=empty,
                missing_required_metadata=[x for x in ['assay_chembl_id','standard_relation','document_date','experiment_date','chembl_release'] if x not in header],
                labels_loaded=False,official_test_records_loaded=False))
        assert all(sha(p)==h for p,h in protected.items())
        save(args.output/'results.json',dict(results=results,scope=plan['scope'],rdkit_version=rdBase.rdkitVersion,
            code_commit=state['code_commit'],input_sha256={str(p):h for p,h in protected.items()},
            fits=0,checkpoint_predictions=0,joint_mmp_audit_completed=False,time_validation_ready=False))
        state.update(status='complete',seconds=time.monotonic()-tick)
        print(json.dumps([dict(task=r['dataset'],shared_scaffolds=r['nonempty_murcko']['cross_role_groups']) for r in results]))
    except Exception as exc:
        state.update(status='failed',error=repr(exc),seconds=time.monotonic()-tick);raise
    finally:save(args.output/'execution.json',state)


if __name__=='__main__':main()
