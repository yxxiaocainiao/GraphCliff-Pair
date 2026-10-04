"""只加载哈希匹配的外部官方源码。"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_fppool():
    target = ROOT / "external/fppool/pooling.py"
    if not target.exists():
        raise FileNotFoundError("先运行 python tools/fetch_fppool.py 下载固定官方版本")
    expected = json.loads((ROOT / "docs/sources.json").read_text())["fppool"]["sha256"]
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError("FPPool 来源哈希不匹配")
    spec = importlib.util.spec_from_file_location("official_fppool", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FingerprintPool
