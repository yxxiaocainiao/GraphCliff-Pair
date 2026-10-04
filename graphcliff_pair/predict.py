"""只输入 SMILES 的预测入口；参考标签仅在 forward 后恢复活性。"""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from .data import FP, digest, top1
from .fingerprint import membership
from .model import PairRegressor
from .vendor.dataset_utils import smiles_to_graph
from .vendor.model import GraphCliffRegressor

def graph_and_identity(smiles, with_fp=False):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"无效或空分子: {smiles}")
    sample=smiles_to_graph(smiles,0.)
    graph=Data(x=sample.x,edge_index=sample.edge_index,edge_attr=sample.edge_attr)
    if with_fp:
        graph.atom_fp=membership(sample)
    return graph,Chem.MolToSmiles(mol,canonical=True),FP.GetFingerprint(mol)

def training_rows_only(source_csv,rows):
    if rows!=sorted(set(rows)) or not rows:
        raise ValueError('训练参考行号必须非空、唯一且有序')
    structures=pd.read_csv(source_csv,usecols=['smiles','split'])
    selected=structures.loc[rows].copy()
    if not selected['split'].eq('train').all():
        raise ValueError('参考行不属于官方train')
    row_set=set(rows)
    # skiprows保留header及指定训练记录，避免加载验证/test的y列值。
    labels=pd.read_csv(source_csv,usecols=['y'],skiprows=lambda line:line>0 and line-1 not in row_set)
    if len(labels)!=len(rows): raise ValueError('训练标签读取与行号不一致')
    selected['y']=pd.Series(labels.y.to_numpy(),index=rows)
    return selected

@torch.no_grad()
def infer(model,variant,query_smiles,bank,batch_size=32,device="cpu",with_fp=False):
    if not query_smiles or batch_size<1:
        raise ValueError("查询非空且 batch_size 为正")
    model.eval()
    query_graphs,canonical,fps={},{},{}
    for row,value in bank.items():
        canonical[row],fps[row]=value["canonical"],value["fp"]
    for i,smiles in enumerate(query_smiles):
        row=-i-1
        graph,can,fp=graph_and_identity(smiles,with_fp)
        query_graphs[row],canonical[row],fps[row]=graph,can,fp
    pairs=top1(query_graphs,bank,canonical,fps)
    # top1 返回原行号排序；按输入顺序恢复查询而非负行号排序。
    by_query={p["query"]:p for p in pairs}
    pairs=[by_query[-i-1] for i in range(len(query_smiles))]
    results=[]
    for start in range(0,len(pairs),batch_size):
        selected=pairs[start:start+batch_size]
        q=Batch.from_data_list([query_graphs[p["query"]] for p in selected]).to(device)
        if variant=="direct":
            edge_counts=torch.bincount(q.batch[q.edge_index[0]],minlength=q.num_graphs)
            if (edge_counts==0).any():
                output=torch.cat([model(g.x,g.edge_index,g.edge_attr,torch.zeros(g.num_nodes,dtype=torch.long,device=device))
                                  for g in q.to_data_list()])
            else:
                output=model(q.x,q.edge_index,q.edge_attr,q.batch)
            output=output.flatten().cpu().tolist()
            delta=None
        else:
            r=Batch.from_data_list([bank[p["reference"]]["graph"] for p in selected]).to(device)
            delta_tensor=model(q,r)
            reference_y=torch.tensor([bank[p["reference"]]["y"] for p in selected],dtype=delta_tensor.dtype,device=device).reshape(-1,1)
            output=(delta_tensor+reference_y).flatten().cpu().tolist()
            delta=delta_tensor.flatten().cpu().tolist()
        for index,pair in enumerate(selected):
            value=float(output[index])
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError("预测非有限")
            i=-pair["query"]-1
            results.append(dict(query_index=i,smiles=query_smiles[i],prediction=value,
                                reference_row=pair["reference"],similarity=pair["similarity"],
                                delta_prediction=delta[index] if delta is not None else None))
    return results

def run(run_dir,source_csv,query_csv,output,batch_size=None,device=None):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    run_dir=Path(run_dir)
    manifest=json.loads((run_dir.parents[2]/"manifest.json").read_text())
    pairing=json.loads((run_dir.parents[1]/"pairs.json").read_text())
    if digest(source_csv)!=pairing["input_sha256"]:
        raise ValueError("训练参考 CSV 哈希不匹配")
    config=manifest["config"]
    batch_size=config["batch_size"] if batch_size is None else batch_size
    arm=next(a for a in config["arms"] if a["name"]==run_dir.name)
    fp=arm["readout"]=="fppool"
    rows=pairing["train_rows"]
    frame=training_rows_only(source_csv,rows)
    bank={}
    for row in rows:
        graph,canonical,fingerprint=graph_and_identity(frame.at[row,"smiles"],fp)
        bank[row]=dict(graph=graph,canonical=canonical,fp=fingerprint,y=float(frame.at[row,"y"]))
    # 只读取 smiles；即使查询文件包含 y，也不会读取或传入模型。
    queries=pd.read_csv(query_csv,usecols=["smiles"]).smiles.tolist()
    if arm["variant"]=="direct":
        model=GraphCliffRegressor(38,13,hidden_size=config["hidden_size"],num_layers=config["num_layers"],dropout=0)
    else:
        model=PairRegressor(arm["variant"],config["hidden_size"],config["num_layers"],config["heads"],arm["readout"])
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    model.load_state_dict(torch.load(run_dir/"best.pt",map_location="cpu",weights_only=True)["model_state_dict"])
    model.to(device)
    output=Path(output)
    if output.exists():
        raise FileExistsError("预测输出已存在，拒绝覆盖")
    records=infer(model,arm["variant"],queries,bank,batch_size,device,fp)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output,index=False)
    print(f"PREDICTION_OK {len(records)} queries; query labels not read")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--source-csv",required=True)
    parser.add_argument("--query-csv",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--device",choices=["cpu","cuda"])
    args=parser.parse_args()
    run(args.run_dir,args.source_csv,args.query_csv,args.output,args.batch_size,args.device)

if __name__=="__main__":
    main()
