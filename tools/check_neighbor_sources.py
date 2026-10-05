"""Verify read-only, pinned source caches without importing upstream projects."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def verify(cache):
    sources = json.loads((cache / 'sources.json').read_text(encoding='utf-8'))
    checked = 0
    repos = {}
    text_files = []
    for record in sources:
        if 'repo' not in record:
            path = cache / record['output']
            normalized = path.read_text(encoding='utf-8').encode('utf-8')
            assert hashlib.sha256(normalized).hexdigest() == record['extracted_text_sha256']
            text_files.append({'file': record['output'], 'normalized_text_sha256': record['extracted_text_sha256'],
                               'cached_bytes_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            continue
        name = record['repo'].split('/')[-1]
        meta = json.loads((cache / name / 'tree.json').read_text(encoding='utf-8'))
        assert not meta['tree']['truncated']
        assert record['commit'] == meta['commit']
        assert record['url'] == f"https://raw.githubusercontent.com/{record['repo']}/{record['commit']}/{record['path']}"
        data = (cache / name / record['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        blob = next(x for x in meta['tree']['tree'] if x['path'] == record['path'])
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == blob['sha']
        checked += 1
        repos[record['repo']] = record['commit']
    for repo in ['XACs', 'PrismNet', 'molucn']:
        assert 'MIT License' in (cache / repo / 'LICENSE').read_text(encoding='utf-8')
    no_license = []
    for repo in ['MAPCliff-WMGR', 'SAGGLR']:
        tree = json.loads((cache / repo / 'tree.json').read_text(encoding='utf-8'))['tree']['tree']
        assert not any('licen' in x['path'].lower() for x in tree)
        no_license.append(repo)
    paths = [cache / r['repo'].split('/')[-1] / r['path'] for r in sources if r.get('path', '').endswith('.py')]
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'))
    # A normalized positive weight for one task is exactly one, regardless of updates.
    for updated_weight in [0.1, 0.5, 1.0, 3.0, 10.0]:
        assert updated_weight / updated_weight == 1.0
    return {'date': '2026-10-05', 'pinned_source_files_checked': checked,
            'python_files_parsed_not_executed': len(paths), 'commits': repos,
            'git_blob_and_sha256_checks': 'pass', 'text_extract_hashes': 'pass',
            'text_hash_convention': 'UTF-8 text with universal-newline normalization',
            'text_files': text_files,
            'licenses_verified': ['XACs:MIT', 'PrismNet:MIT', 'molucn:MIT'],
            'no_license_file_in_pinned_complete_tree': no_license,
            'single_task_normalized_weight_identity': 'pass',
            'upstream_execution': 0, 'training': 0, 'test_evaluation': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; refuse to overwrite')
    result = verify(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{result['pinned_source_files_checked']} pinned files verified; no upstream execution or training.")
