"""Reuse 27 saved validation runs for a frozen, descriptive stratified comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import reuse_baselines as reuse
from graphcliff_pair.data import top1, FP

CONFIG=dict(datasets=['CHEMBL234_Ki','CHEMBL3979_EC50','CHEMBL4792_Ki'],models=['gcn','gat','mlp'],
     seeds=[42,43,44],activity_quantiles=[.2,.4,.6,.8],min_per_class_per_cell=3,
     min_retained_per_class=20,min_retained_fraction=.5,bootstrap_replicates=1000,bootstrap_seed=20261005,
     scope='Exploratory screening; fixed validation split, not causal or independent confirmation.')


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def source(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'frozen_rules.json',CONFIG)
    supports=[]
    for dataset in CONFIG['datasets']:
        folder=args.output/dataset
        reuse.DATASET=dataset
        reuse.prepare(argparse.Namespace(output=folder,baselines=args.baselines,project=ROOT,data=args.data))
        csv=args.data/f'{dataset}.csv'
        meta=pd.read_csv(csv,usecols=['smiles','split'])
        devrows=np.flatnonzero(meta['split'].eq('train'))
        rowset=set(devrows)
        dev=pd.read_csv(csv,usecols=['smiles','y','cliff_mol'],skiprows=lambda i:i>0 and i-1 not in rowset)
        dev.index=devrows
        valid=pd.read_csv(folder/'validation_rows.csv').source_row.to_numpy()
        train=np.array(sorted(rowset-set(valid)))
        mols={int(row):Chem.MolFromSmiles(record.smiles) for row,record in dev.iterrows()}
        assert all(mol is not None for mol in mols.values())
        canon={row:Chem.MolToSmiles(mol,canonical=True) for row,mol in mols.items()}
        assert not set(canon[int(r)] for r in train) & set(canon[int(r)] for r in valid), 'Canonical train/val overlap'
        fps={row:FP.GetFingerprint(mol) for row,mol in mols.items()}
        train_neighbors=pd.DataFrame(top1(train,train,canon,fps))
        valid_neighbors=pd.DataFrame(top1(valid,train,canon,fps)).set_index('query').loc[valid]
        median=float(np.median(train_neighbors.similarity))
        edges=np.quantile(dev.loc[train].y,CONFIG['activity_quantiles']).tolist()
        assert len(set(edges))==4, 'Duplicate training quantiles: do not change bins after observing results'
        frame=pd.read_csv(folder/'validation_rows.csv')
        frame['activity_bin']=np.searchsorted(edges,frame.y,side='right')
        frame['neighbor_similarity']=valid_neighbors.similarity.to_numpy()
        frame['coverage_bin']=(frame.neighbor_similarity>=median).astype(int)
        frame.to_csv(folder/'strata_rows.csv',index=False)
        train_neighbors.to_csv(folder/'train_neighbor_rows.csv',index=False)
        cells=[]
        for design,cols in [('activity',['activity_bin']),('activity_coverage',['activity_bin','coverage_bin'])]:
            cell_records=[]
            for key,sub in frame.groupby(cols,sort=True):
                key=key if isinstance(key,tuple) else (key,)
                n0=int(sub.cliff_mol.eq(0).sum()); n1=int(sub.cliff_mol.eq(1).sum())
                cell_records.append(dict(key=[int(k) for k in key],noncliff=n0,cliff=n1,
                    retained=min(n0,n1)>=CONFIG['min_per_class_per_cell'],weight_count=min(n0,n1)))
            totals={c:sum(r['cliff' if c else 'noncliff'] for r in cell_records if r['retained']) for c in [0,1]}
            eligible=all(totals[c]>=CONFIG['min_retained_per_class'] and
                          totals[c]/int(frame.cliff_mol.eq(c).sum())>=CONFIG['min_retained_fraction'] for c in [0,1])
            cells.append(dict(design=design,cells=cell_records,retained_noncliff=totals[0],retained_cliff=totals[1],
                  original_noncliff=int(frame.cliff_mol.eq(0).sum()),original_cliff=int(frame.cliff_mol.eq(1).sum()),eligible=eligible))
        support=dict(dataset=dataset,training_rows=len(train),validation_rows=len(frame),activity_edges=edges,
             train_loo_similarity_median=median,designs=cells)
        dump(folder/'support.json',support)
        supports.append(support)
        print(dataset,[(c['design'],c['eligible'],c['retained_cliff'],c['retained_noncliff']) for c in cells],flush=True)
    dump(args.output/'support_manifest.json',supports)
    print('All grouping rules and support counts saved before reading predictions.')


def analyze(args):
    assert json.loads((args.output/'frozen_rules.json').read_text())==CONFIG
    records=[]; sources={}; mismatches={}
    for dataset_idx,dataset in enumerate(CONFIG['datasets']):
        folder=args.output/dataset
        frame=pd.read_csv(folder/'strata_rows.csv').set_index('source_row')
        support=json.loads((folder/'support.json').read_text())
        for model in CONFIG['models']:
            errors=[]
            for seed in CONFIG['seeds']:
                path=args.baselines/f'results/{model}/{dataset}/seed_{seed}/validation_predictions.csv'
                manifest_path=path.parent/'manifest.json'
                manifest=json.loads(manifest_path.read_text())
                for rel,expected in manifest['artifacts'].items():
                    artifact=args.baselines/rel
                    assert source(artifact)==expected, f'Artifact mismatch: {artifact}'
                    sources[str(artifact)]=expected
                for rel,expected in manifest['identity']['files'].items():
                    artifact=args.baselines/rel
                    actual=source(artifact)
                    sources[str(artifact)]=actual
                    if actual!=expected: mismatches[str(artifact)]=dict(expected=expected,actual=actual)
                sources[str(manifest_path)]=source(manifest_path)
                pred=pd.read_csv(path)
                assert pred.source_row.is_unique and set(pred.source_row)==set(frame.index)
                pred=pred.set_index('source_row').loc[frame.index]
                assert np.array_equal(pred.smiles,frame.smiles)
                assert np.array_equal(pred.cliff_mol,frame.cliff_mol)
                assert np.allclose(pred.y,frame.y,atol=1e-12,rtol=0)
                assert np.isfinite(pred.y_pred).all()
                error=(pred.y_pred.to_numpy()-frame.y.to_numpy())**2
                errors.append(error)
                frame[f'{model}_{seed}_squared_error']=error
            mean_error=np.mean(errors,axis=0)
            raw_gap=float(mean_error[frame.cliff_mol.eq(1)] .mean()-mean_error[frame.cliff_mol.eq(0)].mean())
            for design_idx,design in enumerate(support['designs']):
                cols=['activity_bin'] if design['design']=='activity' else ['activity_bin','coverage_bin']
                retained=[c for c in design['cells'] if c['retained']]
                total_weight=sum(c['weight_count'] for c in retained)
                if not retained:
                    records.append(dict(dataset=dataset,model=model,design=design['design'],eligible=False,
                                        raw_mse_gap=raw_gap,matched_mse_gap=None,bootstrap_interval=None,seed_gaps=[]))
                    continue
                gap=0.; seed_gaps=np.zeros(3); bootstrap=np.zeros(CONFIG['bootstrap_replicates'])
                cell_rows=[]
                rng=np.random.default_rng(CONFIG['bootstrap_seed']+dataset_idx*10+design_idx)
                for cell in retained:
                    mask=np.ones(len(frame),dtype=bool)
                    for col,val in zip(cols,cell['key']): mask &= frame[col].to_numpy()==val
                    i1=np.flatnonzero(mask & frame.cliff_mol.eq(1).to_numpy())
                    i0=np.flatnonzero(mask & frame.cliff_mol.eq(0).to_numpy())
                    weight=cell['weight_count']/total_weight
                    delta=float(mean_error[i1].mean()-mean_error[i0].mean())
                    gap+=weight*delta
                    for i,error in enumerate(errors): seed_gaps[i]+=weight*(error[i1].mean()-error[i0].mean())
                    # Resample molecules within fixed strata/classes; same draws across models.
                    draws1=rng.choice(i1,size=(CONFIG['bootstrap_replicates'],len(i1)),replace=True)
                    draws0=rng.choice(i0,size=(CONFIG['bootstrap_replicates'],len(i0)),replace=True)
                    bootstrap+=weight*(mean_error[draws1].mean(axis=1)-mean_error[draws0].mean(axis=1))
                    cell_rows.append(dict(key=cell['key'],weight=weight,cliff_mse=float(mean_error[i1].mean()),
                                          noncliff_mse=float(mean_error[i0].mean()),mse_gap=delta))
                interval=np.quantile(bootstrap,[.025,.975]).tolist()
                np.savetxt(folder/f'{model}_{design["design"]}_bootstrap.csv',bootstrap,delimiter=',')
                records.append(dict(dataset=dataset,model=model,design=design['design'],eligible=design['eligible'],
                     retained_cliff=design['retained_cliff'],retained_noncliff=design['retained_noncliff'],
                     raw_mse_gap=raw_gap,matched_mse_gap=float(gap),seed_gaps=seed_gaps.tolist(),
                     bootstrap_interval=interval,cells=cell_rows,
                     positive_and_stable=bool(design['eligible'] and np.all(seed_gaps>0) and interval[0]>0)))
        frame.to_csv(folder/'aligned_squared_errors.csv')
    qualifying={model:[] for model in CONFIG['models']}
    for model in CONFIG['models']:
        for dataset in CONFIG['datasets']:
            pair=[r for r in records if r['model']==model and r['dataset']==dataset]
            if len(pair)==2 and all(r.get('positive_and_stable',False) for r in pair): qualifying[model].append(dataset)
    go=sum(len(v)>=2 for v in qualifying.values())>=2
    dump(args.output/'comparison.json',records)
    dump(args.output/'decision.json',dict(go=go,qualifying_tasks_by_model=qualifying,
           rule='At least two same models qualify on at least two tasks, in both frozen designs.',
           meaning='Information-value screening only; no causal, novelty, journal acceptance or training authorization implied.'))
    dump(args.output/'artifact_audit.json',dict(passed=True,historical_prediction_sets=27,
         official_test_labels_read=False,old_test_predictions_read=False,training_started=False,
         source_sha256=sources,training_source_mismatches=mismatches,
         strict_current_source_training_reproduction=not mismatches))
    print(json.dumps(dict(decision=go,qualifying=qualifying),indent=2))
    for r in records:
        print(r['dataset'],r['model'],r['design'],r['eligible'],r['matched_mse_gap'],r['bootstrap_interval'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','analyze'])
    parser.add_argument('--baselines',type=Path,default=Path('D:/WORK_SPACE/WORK_SPACE/my_work/graphcliff_baselines'))
    parser.add_argument('--data',type=Path,default=Path('D:/GraphCliff-main/benchmark_data'))
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/matched_cliff_20261005')
    args=parser.parse_args()
    {'prepare':prepare,'analyze':analyze}[args.mode](args)
