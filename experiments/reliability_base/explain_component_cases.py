"""M39: frozen nine-case evidence masks; zero fits or point-model inference."""
import argparse,hashlib,json,subprocess,time
from collections import Counter
import joblib,numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import Draw
from run_component_pilot import ROOT,read,reference_graph,feature_function,load_fragments,summarize,save,sha
DOC=ROOT/'docs/research/component_roughness_cases_20261006'
BASE=ROOT/'artifacts/component_roughness_pilot_20261006'
ARMS=['candidate','remove_local_balancing']

def selection(ev,cal):
    assert ev.source_row.is_unique and len(ev)>1
    ranks={}
    for arm in ARMS:
        order=ev.sort_values([arm+'_risk','canonical','source_row'],kind='stable').index
        ranks[arm]=pd.Series(np.arange(len(ev))/(len(ev)-1),index=order).reindex(ev.index)
    u=(ranks[ARMS[0]]-ranks[ARMS[1]])*(abs(ev.prediction-ev.y)-np.median(abs(cal.prediction-cal.y)))
    table=pd.DataFrame({'U':u,'source_row':ev.source_row});chosen={}
    for category,ascending in [('success',False),('failure',True)]:
        pool=table[table.U.gt(0) if category=='success' else table.U.lt(0)]
        chosen[category]=None if pool.empty else int(pool.sort_values(['U','source_row'],ascending=[ascending,True]).index[0])
    remaining=table.drop(index=[i for i in chosen.values() if i is not None]);remaining=remaining.assign(absolute_U=abs(remaining.U))
    chosen['ordinary']=None if remaining.empty else int(remaining.sort_values(['absolute_U','source_row']).index[0])
    return chosen,ranks,u

def evidence(row,y,pair,groups,sims,keep):
    v=summarize(y[keep],pair[np.ix_(keep,keep)],groups[keep]);result=row.copy()
    result.update({k:v[k] for k in ['cb_disp','cb_sali','component_count','component_max_fraction']})
    result.update(nbr_disp=v['raw_disp'],sali_mean=v['raw_sali'],nn_sim=float(max(sims[keep])),local_dens=float(np.mean(sims[keep])))
    return result

def masks(task,source_row,groups):
    counts=Counter(map(int,groups));largest=min(counts,key=lambda c:(-counts[c],c));drop=np.flatnonzero(groups==largest)[:3]
    result=[drop]
    for rep in range(20):
        seed=int(hashlib.sha256(f'{task}|{source_row}|{rep}'.encode()).hexdigest()[:16],16)
        result.append(np.sort(np.random.default_rng(seed).choice(len(groups),len(drop),replace=False)))
    return result

def selfcheck():
    ev=pd.DataFrame({'source_row':[3,2,1,0],'canonical':['a','b','c','d'],'candidate_risk':[0,2,1,3],'remove_local_balancing_risk':[2,0,1,3],'prediction':[3,3,3,3],'y':[0,0,0,0]})
    cal=pd.DataFrame({'prediction':[1,1],'y':[0,0]});chosen,ranks,u=selection(ev,cal)
    assert chosen=={'success':1,'failure':0,'ordinary':3} and np.isclose(u[1],4/3)
    ev['candidate_risk']=ev.remove_local_balancing_risk;chosen,_,_=selection(ev,cal);assert chosen=={'success':None,'failure':None,'ordinary':3}
    g=np.array([2,2,2,1,1,3,4,5,6,7]);a=masks('task',42,g);assert len(a)==21 and np.array_equal(a[0],[0,1,2])
    assert all(len(x)==3 and len(set(x))==3 for x in a) and all(np.array_equal(x,z) for x,z in zip(a,masks('task',42,g)))
    row={'prediction':2,'rf_var':3,'mol_size':4};v=evidence(row,np.arange(10.),np.eye(10),g,np.arange(10.)/10,np.arange(3,10));assert all(v[k]==row[k] for k in row) and v['local_dens']==.6
    print('PASS: case signs/ties/missing slots, deterministic equal-size masks, fixed point inputs')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['selfcheck','run']);a=ap.parse_args()
    if a.mode=='selfcheck':selfcheck();return
    protocol=json.loads((DOC/'protocol.json').read_text());inputs={Path(k):v for k,v in protocol['inputs'].items()}
    assert all(sha(f)==h for f,h in inputs.items());dest=ROOT/'artifacts/component_roughness_cases_20261006';assert not dest.exists();dest.mkdir()
    frozen=json.loads((ROOT/protocol['frozen_rules']).read_text());pilot=json.loads((ROOT/'docs/research/component_roughness_candidate_20261006/pilot_protocol.json').read_text());start=time.monotonic()
    state={'status':'running','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'fits':0,'new_point_predictions':0,'cases':0,'risk_predict_calls':0,'risk_prediction_rows':0,'tasks':[]};save(dest/'execution.json',state)
    try:
        featurize=feature_function(ROOT/'artifacts/roughness_sources_20261005/qsar-landscape-roughness/src/build_features.py');fragment=load_fragments(Path(pilot['graph_reuse_contract']['fragment_source']))
        cases=[];aggregates=[]
        for task in frozen['tasks']:
            tick=time.monotonic();folder=BASE/task;ev=read(folder/'scored_evaluation.csv');cal=read(folder/'scored_calibration.csv');chosen,ranks,u=selection(ev,cal)
            fixture=read(ROOT/pilot['point_prediction_cache']/task/'full/features/data/fixture.csv');ref=fixture[fixture.split.eq('train')].reset_index(drop=True)
            groups=reference_graph(ref.smiles.tolist(),fragment);assert time.monotonic()-tick<frozen['limits']['reference_graph_wall_seconds_per_task']
            expected=json.loads((folder/'full_reference_components.json').read_text());assert expected==[{'canonical':Chem.MolToSmiles(Chem.MolFromSmiles(s)),'component':int(g)} for s,g in zip(ref.smiles,groups)]
            fps,_,_,ok=featurize(ref.smiles.tolist());assert ok.all();models={arm:joblib.load(folder/(arm+'.pkl')).set_params(n_jobs=1) for arm in ARMS};summary={'task':task,'cases':0,'missing_categories':[],'arms':{arm:{'target_above_random_median':0,'target_above_random_max':0,'target_zero_change':0,'mean_target_abs_change':0.,'mean_random_abs_change':0.} for arm in ARMS}}
            for category,i in chosen.items():
                if i is None:summary['missing_categories'].append(category);continue
                row=ev.loc[i];qfps,_,_,ok=featurize([row.smiles]);assert ok.all();sims=np.array(DataStructs.BulkTanimotoSimilarity(qfps[0],fps));nn=np.argsort(-sims)[:10];selected=[fps[j] for j in nn];pair=np.array([DataStructs.BulkTanimotoSimilarity(fp,selected) for fp in selected]);local_g=groups[nn];local_y=ref.y.to_numpy()[nn];local_sim=sims[nn]
                drop=masks(task,int(row.source_row),local_g);keep=[np.arange(10)]+[np.setdiff1d(np.arange(10),x) for x in drop];features=pd.DataFrame([evidence(row.to_dict(),local_y,pair,local_g,local_sim,k) for k in keep]);columns=['nn_sim','local_dens','nbr_disp','sali_mean','cb_disp','cb_sali','component_count','component_max_fraction'];delta=float(np.max(abs(features.loc[0,columns].to_numpy(dtype=float)-row[columns].to_numpy(dtype=float))));assert delta<=1e-10
                record={'task':task,'category':category,'source_row':int(row.source_row),'query':row.to_dict(),'rank_proxy_U':float(u[i]),'ranks':{arm:float(ranks[arm][i]) for arm in ARMS},'calibration_median_error':float(np.median(abs(cal.prediction-cal.y))),'feature_replay_max_abs':delta,'neighbors':[{'smiles':ref.iloc[j].smiles,'reference_y':float(ref.iloc[j].y),'component':int(groups[j]),'similarity':float(sims[j]),'reference_index':int(j)} for j in nn],'deleted_local_indices':[x.tolist() for x in drop],'arms':{}}
                for arm,model in models.items():
                    scores=model.predict(features[model.feature_names_in_.tolist()]);state['risk_predict_calls']+=1;state['risk_prediction_rows']+=len(scores);assert np.isfinite(scores).all() and abs(scores[0]-row[arm+'_risk'])<=1e-10
                    changes=scores[1:]-scores[0];random_abs=abs(changes[1:]);target_abs=float(abs(changes[0]));record['arms'][arm]={'scores':scores.tolist(),'signed_changes':changes.tolist(),'target_abs_change':target_abs,'random_abs_median':float(np.median(random_abs)),'random_abs_mean':float(np.mean(random_abs)),'random_abs_max':float(max(random_abs))}
                    s=summary['arms'][arm];s['target_above_random_median']+=int(target_abs>np.median(random_abs));s['target_above_random_max']+=int(target_abs>max(random_abs));s['target_zero_change']+=int(target_abs==0);s['mean_target_abs_change']+=target_abs;s['mean_random_abs_change']+=float(np.mean(random_abs))
                # ponytail: one static grid per case; interactive chemistry viewer only if needed for review.
                legends=[f'{category} | cached y={row.y:.3f}\npred={row.prediction:.3f} error={abs(row.prediction-row.y):.3f}']+[f'neighbor {j+1} | group {int(local_g[j])}\nsim={local_sim[j]:.3f} y={local_y[j]:.3f}' for j in range(10)]
                name=f'{task}_{category}';Draw.MolsToGridImage([Chem.MolFromSmiles(row.smiles)]+[Chem.MolFromSmiles(ref.iloc[j].smiles) for j in nn],molsPerRow=3,subImgSize=(360,260),legends=legends).save(str(dest/(name+'.png')))
                save(dest/(name+'.json'),record);cases.append(record);summary['cases']+=1;state['cases']+=1;assert state['cases']<=frozen['limits']['max_cases'];assert time.monotonic()-start<frozen['limits']['total_wall_seconds']
            for s in summary['arms'].values():
                for k in ['mean_target_abs_change','mean_random_abs_change']:s[k]/=summary['cases'] if summary['cases'] else 1
            aggregates.append(summary);state['tasks'].append({'task':task,'seconds':time.monotonic()-tick,'reference_rows':len(ref),'reference_components':len(set(groups))});save(dest/'execution.json',state);print('CASES',task,summary['cases'],flush=True)
        assert all(sha(f)==h for f,h in inputs.items());state.update(status='complete',seconds=time.monotonic()-start,input_hashes_unchanged=True);save(dest/'execution.json',state);save(dest/'cases.json',cases);save(dest/'aggregates.json',aggregates)
        html=['<meta charset="utf-8"><title>M39 complete case gallery</title><h1>M39 exploratory cases, cached activity scale</h1><p>All fixed cases; no chemical intervention, atom attribution or independent confirmation.</p>']
        for c in cases:
            name=c['task']+'_'+c['category'];html.append(f'<h2>{name}</h2><img src="{name}.png" width="1080"><pre>'+json.dumps({k:v for k,v in c.items() if k not in ['neighbors','query']},indent=2)+'</pre>')
        (dest/'gallery.html').write_text('\n'.join(html),encoding='utf-8')
    except Exception as error:state.update(status='failed',error=repr(error),seconds=time.monotonic()-start);save(dest/'execution.json',state);raise
if __name__=='__main__':main()
