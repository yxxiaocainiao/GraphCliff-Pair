"""Reuse finite single-cut MMP relations for a zero-fit development graph audit."""
import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from run_phase_a import ROOT, save, sha


def load_fragments(source,extended=True):
    tree=ast.parse(source.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['heavy','fragments']]
    assert len(functions)==2
    if extended:
        calls=[n for fn in functions for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='FragmentMol']
        assert len(calls)==1 and not any(k.arg=='maxCutBonds' for k in calls[0].keywords)
        calls[0].keywords.append(ast.keyword(arg='maxCutBonds',value=ast.Call(func=ast.Attribute(value=ast.Name(id='mol',ctx=ast.Load()),attr='GetNumBonds',ctx=ast.Load()),args=[],keywords=[])))
    module=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))
    env={'Chem':Chem,'rdMMPA':rdMMPA};exec(compile(module,str(source),'exec'),env)
    return env['fragments']


def star(ids):
    return [(ids[0],i) for i in ids[1:]] if ids else []


def mmp_spanning(variables):
    nonempty=[ids for ids in variables.values() if ids]
    if len(nonempty)<2:return []
    first,other=nonempty[0][0],nonempty[1][0]
    return [(first,i) for ids in nonempty[1:] for i in ids]+[(other,i) for i in nonempty[0][1:]]


def labels(n,edges):
    if not edges:return np.arange(n)
    a,b=np.array(sorted(edges),dtype=int).T
    graph=csr_matrix((np.ones(len(a),dtype=np.int8),(a,b)),shape=(n,n))
    return connected_components(graph,directed=False,return_labels=True)[1]


def describe(value,roles):
    counts=np.bincount(value);component_roles=defaultdict(set)
    for c,r in zip(value,roles):component_roles[int(c)].add(r)
    crossing={c for c,v in component_roles.items() if len(v)>1}
    return dict(components=len(counts),largest_rows=int(counts.max()),largest_fraction=float(counts.max()/len(value)),
        top_ten_component_rows=sorted(map(int,counts),reverse=True)[:10],
        cross_role_components=len(crossing),cross_role_rows=sum(int(counts[c]) for c in crossing),
        components_touching_role={r:sum(r in v for v in component_roles.values()) for r in sorted(set(roles))},
        necessary_three_roles=(len(counts)>=3 and counts.max()<=int(np.ceil(.6*len(value)))))


def self_check(source):
    old,new=load_fragments(source,False),load_fragments(source)
    for smiles in ['Cc1ccccc1','CCc1ccccc1','COc1ccccc1']:
        mol=Chem.MolFromSmiles(smiles);assert old(mol)==new(mol)
    long=Chem.MolFromSmiles('C'*24)
    assert old(long)=={} and new(long),'Default20 cutoff fixture must expose omission'
    variables={'a':[0,1],'b':[2,3],'c':[4]}
    edges=set(mmp_spanning(variables));full={(i,j) for a,ids in variables.items() for b,js in variables.items() if a!=b for i in ids for j in js}
    assert np.array_equal(labels(5,edges)[:,None]==labels(5,edges),labels(5,full)[:,None]==labels(5,full))
    assert mmp_spanning({'a':[0,1]})==[]
    print('PASS: small-molecule equivalence, default20 cutoff, exact component-preserving compression')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    pp=ROOT/'docs/research/joint_dependency_preflight_20261006/protocol.json'
    plan=json.loads(pp.read_text(encoding='utf-8'));source=Path(plan['reuse_source'])
    assert sha(source)==plan['reuse_sha256']
    if args.self_check:self_check(source);return
    assert args.output is not None and not args.output.exists(),'Refuse overwrite'
    inputs={(Path(p) if Path(p).is_absolute() else ROOT/p):h for p,h in plan['inputs'].items()}
    inputs.update({pp:sha(pp),source:sha(source),Path(__file__):sha(Path(__file__))})
    assert all(sha(p)==h for p,h in inputs.items())
    fragment=load_fragments(source);args.output.mkdir(parents=True)
    state=dict(status='running',attempts=1,fits=0,checkpoint_predictions=0,official_test_records_loaded=False,
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),tasks=[])
    save(args.output/'execution.json',state)
    try:
        results=[]
        for task in plan['tasks']:
            wall,cpu=time.monotonic(),time.process_time()
            def check_budget():
                assert time.monotonic()-wall<plan['limits']['task_wall_seconds'] and time.process_time()-cpu<plan['limits']['task_cpu_seconds'],'Task limit reached; no omitted edges accepted'
            manifest=json.loads((ROOT/plan['partitions']/(task+'.json')).read_text(encoding='utf-8'))
            role_map={i:r for r,ids in manifest['roles'].items() for i in ids};chosen=set(role_map)
            raw=Path(plan['data_root'])/(task+'.csv')
            f=pd.read_csv(raw,usecols=['smiles','split'],skiprows=lambda i:i>0 and i-1 not in chosen)
            assert len(f)==len(chosen) and f['split'].eq('train').all()
            nodes=[];canonical=defaultdict(list);scaffold=defaultdict(list);cores=defaultdict(lambda:defaultdict(list))
            empty=0;over20=0
            for i,(source_row,row) in enumerate(zip(sorted(chosen),f.itertuples(index=False))):
                check_budget();mol=Chem.MolFromSmiles(row.smiles);assert mol is not None
                canon=Chem.MolToSmiles(mol);scaf=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
                canonical[canon].append(i)
                if scaf:scaffold[scaf].append(i)
                else:empty+=1
                # Default cut SMARTS has one atom-pair match per cuttable bond.
                pattern=Chem.MolFromSmarts('[#6+0;!$(*=,#[!#6])]!@!=!#[*]')
                over20+=len(mol.GetSubstructMatches(pattern,uniquify=True,maxMatches=mol.GetNumBonds()+1))>20
                fragments=fragment(mol) if mol.GetNumBonds() else {}
                for core,variables in fragments.items():
                    for var in set(variables):cores[core][var].append(i)
                nodes.append(dict(source_row=source_row,role=role_map[source_row],canonical=canon,scaffold=scaf))
                if (i+1)%500==0:print(task,'fragmented',i+1,flush=True)
            check_budget()
            edges={kind:set() for kind in ['canonical','scaffold','mmp']}
            for kind,buckets in [('canonical',canonical),('scaffold',scaffold)]:
                for ids in buckets.values():edges[kind].update(tuple(sorted(e)) for e in star(ids))
            occurrences=0;active=0
            for variables in cores.values():
                check_budget()
                if len(variables)<2:continue
                active+=1;sizes=[len(v) for v in variables.values()];n=sum(sizes)
                occurrences+=(n*n-sum(k*k for k in sizes))//2
                edges['mmp'].update(tuple(sorted(e)) for e in mmp_spanning(variables))
            roles=[x['role'] for x in nodes];base_edges=edges['canonical']|edges['scaffold'];all_edges=base_edges|edges['mmp']
            base_labels=labels(len(nodes),base_edges);joint_labels=labels(len(nodes),all_edges)
            for buckets in [canonical,scaffold]:
                assert all(len({int(joint_labels[i]) for i in ids})==1 for ids in buckets.values())
            assert all(len({int(joint_labels[i]) for ids in v.values() for i in ids})==1 for v in cores.values() if len(v)>=2)
            check_budget();dest=args.output/task;dest.mkdir()
            save(dest/'nodes.json',nodes);save(dest/'edges.json',{k:sorted(v) for k,v in edges.items()})
            save(dest/'components.json',joint_labels.tolist())
            result=dict(dataset=task,selected_rows=len(nodes),role_rows={k:len(v) for k,v in manifest['roles'].items()},
                empty_scaffold_rows=empty,molecules_over_default20=over20,active_mmp_cores=active,
                mmp_pair_occurrences_not_unique=occurrences,spanning_edges_by_type={k:len(v) for k,v in edges.items()},
                canonical_scaffold=describe(base_labels,roles),joint=describe(joint_labels,roles),
                seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,complete_for_frozen_definition=True)
            results.append(result);state['tasks'].append(result);save(args.output/'execution.json',state)
            print('DONE',task,'components',result['joint']['components'],'largest',result['joint']['largest_rows'],flush=True)
        assert all(sha(p)==h for p,h in inputs.items())
        save(args.output/'results.json',dict(results=results,code_commit=state['code_commit'],rdkit_version=rdBase.rdkitVersion,
            input_sha256={str(p):h for p,h in inputs.items()},new_fits=0,checkpoint_predictions=0,scope=plan['scope'],
            formal_split_feasibility='unknown: no assignment, cliff counts, power or external dates evaluated'))
        state['status']='complete'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:save(args.output/'execution.json',state)


if __name__=='__main__':main()
