"""M40 read-only artifact checks; no training, model loading or score computation."""
import argparse,hashlib,json,subprocess,tempfile,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/research/publication_assets_20261006'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def check(path,expected=None):
    path=Path(path);exists=path.is_file();actual=sha(path) if exists else None
    return {'path':str(path),'exists':exists,'sha256':actual,'expected_sha256':expected,'matches':exists and (expected is None or actual==expected)}
def selfcheck():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'input';p.write_bytes(b'fixed');h=sha(p);assert check(p,h)['matches']
        p.write_bytes(b'changed');assert not check(p,h)['matches'] and not check(Path(d)/'missing')['matches']
    print('PASS: matching, changed and missing source guards')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['selfcheck','run']);a=ap.parse_args()
    if a.mode=='selfcheck':selfcheck();return
    start=time.monotonic();protocol=json.loads((DOC/'protocol.json').read_text());bound=[check(ROOT/p,h) for p,h in protocol['inputs'].items()];assert all(r['matches'] for r in bound)
    inventory=json.loads((ROOT/'experiments/reliability_base/local_reuse_inventory.json').read_text());old=Path(inventory['local_source']);historic=[]
    for item in inventory['chemprop_checkpoint_checks']:
        task=item['dataset'];manifest=old/'results/chemprop'/task/'manifest.json';m=json.loads(manifest.read_text());records=[check(manifest,item['manifest_sha256']),check(old/m['checkpoint'],item['checkpoint_sha256'])];assert m['dataset']==task and m['seed']==item['seed'];historic.append({'kind':'historic_chemprop','task':task,'seed':item['seed'],'checks':records,'evidence':'hash only, historical validation-selected; test predictions not read'})
    for item in inventory['neural_existence_checks']:
        task,model,seed=item['dataset'],item['model'],item['seed'];folder=old/f'results/{model}/{task}/seed_{seed}';manifest=folder/'manifest.json';m=json.loads(manifest.read_text());records=[check(manifest)]+[check(old/p,h) for p,h in m['artifacts'].items()];assert m['identity']['dataset']==task and m['identity']['model']==model and m['identity']['seed']==seed;historic.append({'kind':'historic_'+model,'task':task,'seed':seed,'checks':records,'evidence':'manifest-bound files only; validation was selected, no inference/score replay'})
    fixed=json.loads((ROOT/'experiments/reliability_base/fixed_protocol.json').read_text());current=[]
    for t in fixed['tasks']:
        task=t['dataset'];folder=ROOT/'artifacts/reliability_phase_a_20261005_v3'/task;weights=[check(folder/job/'training/model_0/best.pt') for job in ['fold0','fold1','fold2','full']];roles=[];sets={}
        for role in ['oof','calibration','evaluation']:
            p=folder/(role+'.csv');frame=pd.read_csv(p,usecols=['source_row','canonical'],float_precision='round_trip');sets[role]=set(frame.canonical);expected=t['role_rows']['fit' if role=='oof' else role];assert len(frame)==expected and frame.source_row.is_unique and frame.canonical.is_unique;roles.append(dict(check(p),role=role,rows=len(frame)))
        assert not any(sets[a]&sets[b] for a,b in [('oof','calibration'),('oof','evaluation'),('calibration','evaluation')]);current.append({'task':task,'weights':weights,'roles':roles,'canonical_role_overlap':0,'evidence':'existence/hash and identity only; no point inference or labels loaded'})
    pilot=json.loads((ROOT/'docs/research/component_roughness_pilot_20261006/results.json').read_text());private=ROOT/'artifacts/component_roughness_pilot_20261006';bindings=[check(private/p,h) for p,h in pilot['private_output_sha256'].items()]
    case=json.loads((ROOT/'docs/research/component_roughness_cases_20261006/protocol.json').read_text());bindings += [check(Path(p),h) for p,h in case['inputs'].items()]
    v=json.loads((ROOT/'docs/research/component_roughness_cases_20261006/verification.json').read_text());bindings += [check(ROOT/'artifacts/component_roughness_cases_20261006'/p,h) for p,h in v['artifact_sha256'].items()]
    all_checks=bound+[r for asset in historic for r in asset['checks']]+[r for asset in current for k in ['weights','roles'] for r in asset[k]]+bindings
    assert all(r['matches'] for r in all_checks);assert all(check(ROOT/p,h)['matches'] for p,h in protocol['inputs'].items())
    result={'status':'pass','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'runner_sha256':sha(Path(__file__)),'seconds':time.monotonic()-start,'source_bindings':bound,'historic_assets':historic,'current_assets':current,'M38_M39_bindings':bindings,'counts':{'old_chemprop':sum(r['kind']=='historic_chemprop' for r in historic),'old_neural':sum(r['kind']!='historic_chemprop' for r in historic),'current_point_weights':sum(len(r['weights']) for r in current),'current_role_caches':sum(len(r['roles']) for r in current),'checks':len(all_checks)},'fits':0,'predictions':0,'test_labels_read':0,'new_sources':0,'limit':'These checks do not certify prospective protocol, checkpoint forward replay, independent calibration, method novelty or publication acceptance.'}
    (DOC/'asset_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['counts'])
if __name__=='__main__':main()
