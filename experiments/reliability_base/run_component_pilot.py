"""M38: reference-only features then a fixed nine-fit risk pilot."""
import argparse,ast,json,subprocess,time
from collections import defaultdict,Counter
from pathlib import Path
import joblib,numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from run_phase_a import ROOT,save,sha,cq_function,metrics
from audit_joint_dependencies import load_fragments,labels,star,mmp_spanning
from component_roughness import summarize,selfcheck
from check_simple_risk_control import combination
DOC=ROOT/'docs/research/component_roughness_candidate_20261006'
NEW=['cb_disp','cb_sali','component_count','component_max_fraction']

def read(path):return pd.read_csv(path,float_precision='round_trip')
def feature_function(path):
    tree=ast.parse(path.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='featurize')
    env={'Chem':Chem,'np':np,'DataStructs':DataStructs,'_gen':rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env);return env['featurize']

def reference_graph(smiles,fragment):
    canon=defaultdict(list);scaffold=defaultdict(list);cores=defaultdict(lambda:defaultdict(list))
    for i,smi in enumerate(smiles):
        mol=Chem.MolFromSmiles(smi);assert mol is not None
        canon[Chem.MolToSmiles(mol)].append(i);scaf=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
        if scaf:scaffold[scaf].append(i)
        for core,variables in fragment(mol).items():
            for v in set(variables):cores[core][v].append(i)
    edges=set()
    for buckets in [canon,scaffold]:
        for ids in buckets.values():edges.update(star(ids))
    for variables in cores.values():edges.update(mmp_spanning(variables))
    result=labels(len(smiles),edges)
    assert all(len({int(result[i]) for ids in v.values() for i in ids})==1 for v in cores.values() if len(v)>1)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['selfcheck','prepare','run']);ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.mode=='selfcheck':selfcheck();return
    p=json.loads((DOC/'pilot_protocol.json').read_text());m=json.loads((DOC/'protocol.json').read_text());base=ROOT/p['point_prediction_cache'];dest=a.output.resolve();assert str(dest).startswith(str(ROOT/'artifacts'))
    inputs={ROOT/k:v for k,v in m['inputs'].items()};g=p['graph_reuse_contract'];inputs.update({ROOT/g['protocol_file']:g['protocol_sha256'],ROOT/g['helper_file']:g['helper_sha256'],Path(g['fragment_source']):g['fragment_source_sha256']})
    for f in [DOC/'pilot_protocol.json',DOC/'protocol.json',Path(__file__),ROOT/'experiments/reliability_base/component_roughness.py',ROOT/'experiments/reliability_base/run_phase_a.py']:inputs[f]=sha(f)
    assert all(sha(f)==h for f,h in inputs.items());start=time.monotonic()
    if a.mode=='prepare':
        assert not dest.exists();dest.mkdir(parents=True);state={'status':'preparing','jobs':[],'risk_fits_started':0,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};save(dest/'preparation.json',state)
        try:
            featurize=feature_function(ROOT/'artifacts/roughness_sources_20261005/qsar-landscape-roughness/src/build_features.py');fragment=load_fragments(Path(g['fragment_source']))
            for task in p['tasks']:
                frames=[];out=dest/task;out.mkdir()
                for name in ['fold0','fold1','fold2','full']:
                    tick=time.monotonic();folder=base/task/name;fixture=read(folder/'features/data/fixture.csv');reference=fixture[fixture.split.eq('train')].reset_index(drop=True);query=fixture[fixture.split.eq('test')].reset_index(drop=True);joined=read(folder/'joined.csv');assert query.smiles.tolist()==joined.smiles.tolist()
                    components=reference_graph(reference.smiles.tolist(),fragment);fps,_,_,ok=featurize(reference.smiles.tolist());qfps,_,_,qok=featurize(query.smiles.tolist());assert ok.all() and qok.all();y=reference.y.to_numpy();records=[];replay=[]
                    for i,qfp in enumerate(qfps):
                        assert time.monotonic()-tick<p['budgets']['max_graph_feature_wall_seconds_per_job']
                        sims=np.array(DataStructs.BulkTanimotoSimilarity(qfp,fps));nn=np.argsort(-sims)[:10];selected=[fps[j] for j in nn];pair=np.array([DataStructs.BulkTanimotoSimilarity(fp,selected) for fp in selected]);v=summarize(y[nn],pair,components[nn]);records.append({k:v[k] for k in NEW});replay.append([v['raw_disp'],v['raw_sali']])
                    delta=np.max(np.abs(np.asarray(replay)-joined[['nbr_disp','sali_mean']].to_numpy()));assert delta<=p['replay_tolerance'],(task,name,delta)
                    extra=pd.DataFrame(records);joined[NEW]=extra[NEW];joined.to_csv(out/(name+'.csv'),index=False);frames.append(joined)
                    if name=='full':save(out/'full_reference_components.json',[{'canonical':Chem.MolToSmiles(Chem.MolFromSmiles(s)),'component':int(c)} for s,c in zip(reference.smiles,components)])
                    state['jobs'].append({'task':task,'job':name,'reference_rows':len(reference),'reference_components':len(set(components)),'query_rows':len(query),'raw_replay_max_abs':float(delta),'component_count_histogram':dict(Counter(map(int,extra.component_count))),'changed_rows':int((np.max(np.abs(extra[['cb_disp','cb_sali']].to_numpy()-joined[['nbr_disp','sali_mean']].to_numpy()),axis=1)>1e-10).sum()),'seconds':time.monotonic()-tick});save(dest/'preparation.json',state);print('PREPARED',task,name,'max_delta',delta,flush=True)
                    assert time.monotonic()-start<p['budgets']['total_preparation_wall_seconds']
                new_oof=pd.concat(frames[:3]).set_index('source_row');full=frames[3].set_index('source_row')
                for role in ['oof','calibration','evaluation']:
                    old=read(base/task/(role+'.csv'));new=new_oof if role=='oof' else full;assert old.source_row.is_unique;old[NEW]=new.loc[old.source_row,NEW].to_numpy();old.to_csv(out/(role+'.csv'),index=False)
            assert all(sha(f)==h for f,h in inputs.items());state.update(status='prepared',seconds=time.monotonic()-start,input_sha256={str(f):h for f,h in inputs.items()},prepared_files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file() and f.name!='preparation.json'});save(dest/'preparation.json',state)
        except Exception as e:state.update(status='failed',error=repr(e));save(dest/'preparation.json',state);raise
        return
    prep=json.loads((dest/'preparation.json').read_text());assert prep['status']=='prepared';assert all(sha(dest/f)==h for f,h in prep['prepared_files'].items());assert all(sha(Path(f))==h for f,h in prep['input_sha256'].items());assert not (dest/'execution.json').exists()
    state={'status':'running','risk_fits_started':0,'arms_completed':0,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};save(dest/'execution.json',state);fixed=json.loads((ROOT/'experiments/reliability_base/fixed_protocol.json').read_text());cq=cq_function(ROOT/'artifacts/roughness_sources_20261005/qsar-landscape-roughness/src/conformal.py');results=[]
    try:
        for task in p['tasks']:
            out=dest/task;frames={r:read(out/(r+'.csv')) for r in ['oof','calibration','evaluation']};oof,cal,ev=(frames[r] for r in ['oof','calibration','evaluation']);target=np.abs(oof.prediction-oof.y);ref=json.loads((out/'full_reference_components.json').read_text());group={x['canonical']:x['component'] for x in ref};sizes=Counter(group.values());w=np.array([1/sizes[group[c]] for c in oof.canonical]);w/=w.mean();scores={};arms={}
            for name in ['generic','augmented','simple_combination']:
                sc,se=(combination(oof,f) if name=='simple_combination' else f[name+'_risk'].to_numpy() for f in [cal,ev]);scores[name]=(sc,se);arms[name]=metrics(ev,se,cal,sc,cq,fixed)
            for name,spec in p['arms'].items():
                assert state['risk_fits_started']<p['budgets']['max_risk_fits'] and time.monotonic()-start<p['budgets']['total_risk_fit_wall_seconds'];columns=p['generic']+spec['roughness']+p['shared_support'];assert len(columns)==9 and np.isfinite(oof[columns]).all().all();state['risk_fits_started']+=1;save(dest/'execution.json',state);tick=time.monotonic()
                rf=RandomForestRegressor(**p['rf']).fit(oof[columns],target,sample_weight=w if spec['sample_weight'] else None);sc,se=rf.predict(cal[columns]),rf.predict(ev[columns]);assert np.isfinite(sc).all() and np.isfinite(se).all();joblib.dump(rf,out/(name+'.pkl'));loaded=joblib.load(out/(name+'.pkl'));assert np.array_equal(loaded.predict(ev[columns]),se);scores[name]=(sc,se);arms[name]=metrics(ev,se,cal,sc,cq,fixed);arms[name]['fit_seconds']=time.monotonic()-tick;state['arms_completed']+=1;save(dest/'execution.json',state);print('FIT',task,name,state['risk_fits_started'],'/9',flush=True)
            for name,(sc,se) in scores.items():cal[name+'_risk']=sc;ev[name+'_risk']=se
            cal.to_csv(out/'scored_calibration.csv',index=False);ev.to_csv(out/'scored_evaluation.csv',index=False)
            nodes=json.loads((ROOT/'artifacts/joint_dependency_preflight_20261006'/task/'nodes.json').read_text());components=json.loads((ROOT/'artifacts/joint_dependency_preflight_20261006'/task/'components.json').read_text());post={n['source_row']:c for n,c in zip(nodes,components)};ec=np.array([post[int(i)] for i in ev.source_row]);largest=sorted(Counter(ec).items(),key=lambda x:(-x[1],x[0]))[:3];influence=[]
            for component,n in largest:
                keep=ec!=component;influence.append({'removed_component':int(component),'removed_rows':n,'arms':{name:metrics(ev.loc[keep].reset_index(drop=True),se[keep],cal,sc,cq,fixed)['curve_mean_rmse'] for name,(sc,se) in scores.items()}})
            results.append({'dataset':task,'seed':42,'arms':arms,'component_influence':influence});save(dest/'results.json',{'results':results,'risk_fits_started':state['risk_fits_started'],'new_Chemprop_fits':0,'official_test_rows':0,'protocol_sha256':sha(DOC/'pilot_protocol.json')})
        assert all(sha(Path(f))==h for f,h in prep['input_sha256'].items());state.update(status='complete',seconds=time.monotonic()-start);save(dest/'execution.json',state)
    except Exception as e:state.update(status='failed',error=repr(e));save(dest/'execution.json',state);raise
if __name__=='__main__':main()
