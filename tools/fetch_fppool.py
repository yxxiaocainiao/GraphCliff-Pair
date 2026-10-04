"""下载固定版本的官方池化源码到忽略目录，不重新发布该快照。"""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = json.loads((ROOT / "docs/sources.json").read_text(encoding="utf-8"))["fppool"]
    target = ROOT / "external/fppool/pooling.py"
    content = urllib.request.urlopen(source["url"], timeout=30).read()
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise RuntimeError("官方源码哈希与固定版本不符")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise FileExistsError("已有不同版本源码，拒绝覆盖")
    target.write_bytes(content)
    print(f"FPPool verified: {source['commit']}")

if __name__ == "__main__":
    main()
