"""One bounded, validation-only case audit; descriptive and not a performance test."""
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import networkx as nx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graphcliff_pair.vendor.dataset_utils import smiles_to_graph


def main(output):
    source=Path('D:/GraphCliff-main/benchmark_data/CHEMBL234_Ki.csv')
    old=ROOT/'artifacts/baseline_reuse_234_20261005'
    out=output
    out.mkdir(parents=True,exist_ok=False)
    ranked=pd.read_csv(old/'cliff_case_ranking.csv')
    shared=pd.read_csv(old/'shared_hard_case_ranking.csv')
    selected=sorted(set(ranked.source_row.head(5)) | set(shared.source_row.head(5)))
    selection=dict(rule='First five SVM-better cliff cases and first five shared-hard cliff cases, union; fixed before case audit.',
                   svm_better_rows=ranked.source_row.head(5).tolist(),
                   shared_hard_rows=shared.source_row.head(5).tolist(), selected_rows=selected)
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    meta=pd.read_csv(source,usecols=['smiles','split'])
    dev_rows=np.flatnonzero(meta['split'].eq('train'))
    dev_set=set(dev_rows)
    dev=pd.read_csv(source,usecols=['smiles','y','cliff_mol'],skiprows=lambda i:i>0 and i-1 not in dev_set)
    dev.index=dev_rows
    validation=pd.read_csv(old/'validation_rows.csv').source_row.tolist()
    train=dev.loc[~dev.index.isin(validation)]
    assert len(train)==2632
    q10,q90=np.quantile(train.y,[.1,.9])
    groups=defaultdict(list)
    mols={}
    for row,record in dev.iterrows():
        mol=Chem.MolFromSmiles(record.smiles)
        assert mol is not None
        mols[row]=mol
        groups[Chem.MolToSmiles(mol,isomericSmiles=False)].append(int(row))
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    collision_groups=[]
    for rows in groups.values():
        if len(rows)<2: continue
        values=dev.loc[rows].y
        if values.max()-values.min()<1: continue
        iso={Chem.MolToSmiles(mols[r],isomericSmiles=True) for r in rows}
        if len(iso)<2: continue
        collision_groups.append(dict(rows=rows,label_range=float(values.max()-values.min()),
              validation_rows=[r for r in rows if r in validation]))
    wide=pd.read_csv(old/'aligned_validation_predictions.csv').set_index('source_row')
    cases=[]
    base=ranked.set_index('source_row')
    for row in selected:
        record=base.loc[row]
        ref=int(record.nearest_train_row)
        assert ref in train.index
        mol,refmol=mols[row],mols[ref]
        flat_equal=Chem.MolToSmiles(mol,isomericSmiles=False)==Chem.MolToSmiles(refmol,isomericSmiles=False)
        def graph(molecule):
            sample=smiles_to_graph(Chem.MolToSmiles(molecule,isomericSmiles=True),0)
            g=nx.Graph()
            for i,features in enumerate(sample.x.tolist()): g.add_node(i,features=tuple(features))
            for (a,b),features in zip(sample.edge_index.T.tolist(),sample.edge_attr.tolist()):
                g.add_edge(a,b,features=tuple(features))
            return g
        exact_equal=False
        if flat_equal:
            exact_equal=nx.is_isomorphic(graph(mol),graph(refmol),
                node_match=nx.algorithms.isomorphism.categorical_node_match('features',None),
                edge_match=nx.algorithms.isomorphism.categorical_edge_match('features',None))
        target=float(record.y)
        errors=[float(wide.at[row,f'graphcliff_{s}'])-target for s in [42,43,44]]
        cases.append(dict(source_row=int(row),nearest_train_row=ref,y=target,
          reference_y=float(train.at[ref,'y']),label_delta=target-float(train.at[ref,'y']),
          similarity=float(record.nearest_train_similarity),atom_count=mol.GetNumAtoms(),
          specified_stereo_centers=len(Chem.FindMolChiralCenters(mol,includeUnassigned=False)),
          same_connectivity_as_nearest=flat_equal,identical_graph_features=exact_equal,
          ecfp_equal=gen.GetFingerprint(mol)==gen.GetFingerprint(refmol),
          label_above_train_q90=bool(target>q90),label_outside_train_range=bool(target<train.y.min() or target>train.y.max()),
          graphcliff_signed_errors=errors,svm_signed_error=float(record.svm_prediction)-target,
          three_seeds_same_error_sign=bool(all(e>0 for e in errors) or all(e<0 for e in errors))))
    strata=[]
    for name,mask in [('low_tail',wide.y.lt(q10)),('middle',wide.y.between(q10,q90)),('high_tail',wide.y.gt(q90))]:
        for cliff_only in [False,True]:
            subset=wide.loc[mask & (wide.cliff_mol.eq(1) if cliff_only else True)]
            metrics={}
            for key in ['graphcliff_42','graphcliff_43','graphcliff_44','svm']:
                error=subset[key]-subset.y
                metrics[key]=dict(rmse=float(np.sqrt(np.mean(error**2))),signed_bias=float(error.mean()))
            strata.append(dict(stratum=name,cliff_only=cliff_only,n=len(subset),metrics=metrics))
    result=dict(dataset='CHEMBL234_Ki',official_test_labels_read=False,trained=False,
       train_rows=len(train),validation_rows=len(wide),train_label_min=float(train.y.min()),
       train_label_max=float(train.y.max()),train_q10=float(q10),train_q90=float(q90),
       selected_cases=cases,stereochemical_connectivity_groups_delta_at_least_one=collision_groups,
       tail_description=strata,source_sha256={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
       limitations=['Selected using validation errors, so exploratory only.','No assay-level provenance verification.',
                   'Nearest reference is structural, not a causal explanation.','Train-tail groups cannot establish a new method or generalization.'])
    (out/'case_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/hard_case_audit_234_20261005')
    main(parser.parse_args().output)
