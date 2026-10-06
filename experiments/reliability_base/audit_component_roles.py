"""M35: one label-blind component allocation and within-role finite-MMP audit."""
import argparse,hashlib,itertools,json,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from run_phase_a import ROOT,save,sha
from audit_joint_dependencies import load_fragments

def groups(nodes,components,task):
    buckets=defaultdict(list)
    for i,c in enumerate(components):buckets[c].append(i)
    return sorted(buckets.values(),key=lambda ids:(-len(ids),hashlib.sha256((task+'|'+json.dumps(sorted(nodes[i]['canonical'] for i in ids))).encode()).hexdigest()))

def allocate(buckets,targets):
    counts=[0]*len(targets);assigned={}
    for ids in buckets:
        j=max(range(len(targets)),key=lambda k:(targets[k]-counts[k],-k))
        for i in ids:assigned[i]=j
        counts[j]+=len(ids)
    return assigned

def cliff_pairs(cores,roles,activity):
    pairs=set()
    for variables in cores.values():
        for a,b in itertools.combinations(variables.values(),2):
            for i in a:
                for j in b:
                    if i!=j and roles[i]==roles[j] and abs(activity[i]-activity[j])>1:pairs.add(tuple(sorted((i,j))))
    return pairs

def selfcheck():
    assert allocate([[0,1],[2],[3]],[2,1,1])=={0:0,1:0,2:1,3:2}
    cores={'core':{'a':[0,1],'b':[2,3]},'duplicate':{'a':[0],'b':[2]}}
    assert cliff_pairs(cores,[0,0,0,1],[0.,1.,2.,5.])=={(0,2)}
    assert allocate([[0],[1],[2]],[1,1,1])=={0:0,1:1,2:2}
    print('PASS: allocation/ties, strict cliff threshold, roles, pair deduplication')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selfcheck',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.selfcheck:selfcheck();return
    ppath=ROOT/'docs/research/component_role_feasibility_20261006/protocol.json';p=json.loads(ppath.read_text(encoding='utf-8'))
    assert a.output and not a.output.exists(),'refuse overwrite'
    inputs={Path(k):v for k,v in p['inputs'].items()};inputs[Path(p['reuse_source'])]=p['reuse_sha256']
    assert all(sha(k)==v for k,v in inputs.items());fragment=load_fragments(Path(p['reuse_source']));a.output.mkdir(parents=True)
    result={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ppath),'input_sha256':{str(k):v for k,v in inputs.items()},'tasks':[],'fits':0,'inference':0,'official_test_rows':0}
    for task in p['tasks']:
        start=time.monotonic();base=ROOT/'artifacts/joint_dependency_preflight_20261006'/task
        nodes=json.loads((base/'nodes.json').read_text());components=json.loads((base/'components.json').read_text());assert len(nodes)==len(components)
        buckets=groups(nodes,components,task);roles=list(p['roles']);assignment=allocate(buckets,[len(nodes)*p['roles'][r] for r in roles]);role=[roles[assignment[i]] for i in range(len(nodes))]
        fitgroups=[ids for ids in buckets if role[ids[0]]=='fit'];fold=allocate(fitgroups,[0,0,0])
        chosen={x['source_row'] for x in nodes};f=pd.read_csv(Path(p['data_root'])/(task+'.csv'),usecols=['smiles','split','y [pEC50/pKi]'],skiprows=lambda i:i>0 and i-1 not in chosen)
        assert len(f)==len(nodes) and f['split'].eq('train').all();activity=f['y [pEC50/pKi]'].to_numpy();assert np.isfinite(activity).all()
        cores=defaultdict(lambda:defaultdict(list))
        for i,smi in enumerate(f.smiles):
            assert time.monotonic()-start<p['limits']['task_wall_seconds']
            mol=Chem.MolFromSmiles(smi);assert mol is not None and Chem.MolToSmiles(mol)==nodes[i]['canonical']
            for core,variables in fragment(mol).items():
                for v in set(variables):cores[core][v].append(i)
        for variables in cores.values():
            if len(variables)>1:assert len({components[i] for ids in variables.values() for i in ids})==1
        pairs=cliff_pairs(cores,role,activity);summaries={};fail=[]
        for r in roles:
            ids={i for i,v in enumerate(role) if v==r};cs={components[i] for i in ids};rp=[(i,j) for i,j in pairs if role[i]==r];participants={i for pair in rp for i in pair};cc={components[i] for i in participants}
            summary={'rows':len(ids),'components':len(cs),'fraction':len(ids)/len(nodes),'mmp_cliff_pairs':len(rp),'cliff_participating_rows':len(participants),'cliff_participating_components':len(cc)};summaries[r]=summary
            if len(ids)<p['minimum_rows'][r]:fail.append(r+':minimum_rows')
            if len(cs)<p['minimum_components'][r]:fail.append(r+':minimum_components')
            if abs(summary['fraction']-p['roles'][r])>p['maximum_absolute_ratio_deviation']:fail.append(r+':ratio')
            if r in ('calibration','evaluation'):
                vals=[len(participants),len(rp),len(cc)]
                for key,value in zip(p['cliff_minimum_calibration_evaluation'],vals):
                    if value<p['cliff_minimum_calibration_evaluation'][key]:fail.append(r+':cliff_'+key)
        oof=[]
        for j in range(3):
            held={i for i,v in fold.items() if v==j};reference=set(fold)-held
            summary={'fold':j,'held_rows':len(held),'held_components':len({components[i] for i in held}),'reference_rows':len(reference),'reference_components':len({components[i] for i in reference})};oof.append(summary)
            for key,minimum in p['oof_minimum'].items():
                if summary[key]<minimum:fail.append('oof'+str(j)+':'+key)
        assert all(len({role[i] for i,c in enumerate(components) if c==component})==1 for component in set(components))
        save(a.output/(task+'_tentative_roles.json'),[{'source_row':n['source_row'],'component':components[i],'role':role[i],'oof_fold':fold.get(i)} for i,n in enumerate(nodes)])
        result['tasks'].append({'dataset':task,'roles':summaries,'oof':oof,'failed_conditions':fail,'screening_pass':not fail,'seconds':time.monotonic()-start})
        assert time.monotonic()-start<p['limits']['task_wall_seconds'];save(a.output/'results.json',result);print(task,fail or 'screening_pass',flush=True)
    assert all(sha(k)==v for k,v in inputs.items())
    save(a.output/'results.json',result)
if __name__=='__main__':main()
