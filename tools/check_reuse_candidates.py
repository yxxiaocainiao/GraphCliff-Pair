"""Synthetic interface checks of frozen MIT upstream losses; no model fitting."""
import hashlib
import importlib.util
import ast
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    manifest=ROOT/'docs/research/reuse_candidates_downloads_20261005.json'
    files=json.loads(manifest.read_text(encoding='utf-8-sig'))
    for record in files:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    dir_loss=load('frozen_dir_loss',ROOT/'external/reuse_candidates/dir-a6fdc45/imdb-wiki-dir/loss.py')
    balanced=load('frozen_balanced_loss',ROOT/'external/reuse_candidates/balancedmse-c6f2f33/synthetic_benchmark/loss.py')
    results=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        pred=torch.tensor([[.2],[-.3],[1.2]],device=device,requires_grad=True)
        target=torch.tensor([[0.],[-.2],[1.]],device=device)
        weights=torch.ones_like(target)
        loss=dir_loss.weighted_mse_loss(pred,target,weights=weights)
        expected=torch.nn.functional.mse_loss(pred,target)
        assert torch.allclose(loss,expected,atol=1e-8,rtol=0)
        loss.backward()
        assert torch.isfinite(pred.grad).all()
        results.append(dict(method='DIR weighted_mse_loss',device=device,passed=True,
                            ones_weights_equal_mse=True))
        pred.grad=None
        criterion=balanced.BMCLossMD(init_noise_sigma=1.).to(device)
        try:
            loss=criterion(pred,target)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
            assert criterion.noise_sigma.grad is not None and torch.isfinite(criterion.noise_sigma.grad)
            results.append(dict(method='unmodified BalancedMSE BMCLossMD',device=device,passed=True,
                 loss=float(loss.detach()),finite_prediction_and_sigma_gradient=True))
        except RuntimeError as error:
            if device=='cpu': raise
            results.append(dict(method='unmodified BalancedMSE BMCLossMD',device=device,passed=False,
                 error=str(error),thin_device_adaptation_required=True))
    result=dict(torch_version=torch.__version__,synthetic_only=True,training_started=False,
          existing_model_or_loss_modified=False,upstream_files_unmodified=True,
          init_noise_sigma_note='1.0 is only a synthetic interface-check value, not a chosen experiment hyperparameter.',
          scope='Loss interfaces only; LDS/FDS, IA-MoE and end-to-end molecular integration not executed.',
          checks=results)
    # MIT upstream function stays byte-identical on disk; two device keywords only.
    source_path=ROOT/'external/reuse_candidates/balancedmse-c6f2f33/synthetic_benchmark/loss.py'
    tree=ast.parse(source_path.read_text(encoding='utf-8'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='bmc_loss_md')
    changed=0
    for node in ast.walk(function):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch'
                and node.func.attr in ['eye','arange']):
            assert not any(k.arg=='device' for k in node.keywords)
            node.keywords.append(ast.keyword(arg='device',value=ast.Attribute(
                value=ast.Name(id='pred',ctx=ast.Load()),attr='device',ctx=ast.Load())))
            changed+=1
    assert changed==2
    adapted_ast=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace=dict(balanced.__dict__)
    exec(compile(adapted_ast,str(source_path),'exec'),namespace)
    adapted=namespace['bmc_loss_md']
    adapted_checks=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        pred=torch.tensor([[.2],[-.3],[1.2]],device=device,requires_grad=True)
        target=torch.tensor([[0.],[-.2],[1.]],device=device)
        variance=torch.tensor(1.,device=device,requires_grad=True)
        loss=adapted(pred,target,variance)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(pred.grad).all() and torch.isfinite(variance.grad)
        adapted_checks.append(dict(device=device,passed=True,loss=float(loss.detach()),finite_gradients=True))
    assert all(abs(r['loss']-adapted_checks[0]['loss'])<1e-6 for r in adapted_checks)
    result['thin_device_adaptation']=dict(changes=['torch.eye device=pred.device','torch.arange device=pred.device'],
          formula_unchanged=True,upstream_file_unmodified=True,checks=adapted_checks)
    (ROOT/'docs/research/reuse_candidates_interface_20261005.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
