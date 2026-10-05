"""Read-only XLSX provenance check; no Excel or third-party dependency needed."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_sheets(path):
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid XLSX ZIP member')
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(x.itertext()) for x in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        targets = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        result = {}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).find('m:sheets', NS):
            target = targets[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            cells = {}
            for cell in ET.fromstring(archive.read(target)).findall('.//m:sheetData/m:row/m:c', NS):
                if cell.find('m:f', NS) is not None:
                    raise ValueError('Formula found: cached values are not sufficient for this audit')
                value = cell.find('m:v', NS)
                value = value.text if value is not None else ''
                if cell.attrib.get('t') == 's':
                    value = strings[int(value)]
                elif cell.attrib.get('t') == 'inlineStr':
                    value = ''.join(cell.find('m:is', NS).itertext())
                cells[cell.attrib['r']] = value
            result[sheet.attrib['name']] = cells
    return result


def audit(path):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    sheets = read_sheets(path)
    main, source = sheets['文献阅读表'], sheets['来源与核对']
    rows = sorted(int(k[1:]) for k in main if k.startswith('A') and str(main[k]).isdigit())
    assert rows == list(range(6, 35)), rows
    assert len({main[f'Q{r}'] for r in rows}) == 29
    records = []
    for row in rows:
        assert main[f'A{row}'] == source[f'A{row}']
        assert main[f'Q{row}'] == source[f'B{row}']
        records.append({'record': int(main[f'A{row}']), 'title': main[f'B{row}'],
                        'doi_as_recorded': source.get(f'G{row}', ''),
                        'workbook_row': row, 'reading_basis_as_recorded': main[f'O{row}']})
    assert main['Q17'] == '73QRCM96' and main['Q33'] == 'BCMQIEGP'
    assert '同一' in source['M33'] and '官方摘要' in main['O32']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    return {'date': '2026-10-05', 'source_filename': path.name, 'sha256': before,
            'sheet_names': list(sheets), 'records': records, 'record_count': 29,
            'unique_record_keys': 29, 'independent_studies_after_documented_version_merge': 28,
            'merged_records': [12, 28], 'reading_basis_counts_as_recorded': dict(Counter(main[f'O{r}'] for r in rows)),
            'abstract_only_record': 27, 'source_row_alignment_checks': 58,
            'source_unchanged': True, 'zip_integrity': 'pass',
            'scope': 'User-provided workbook; not a live aidd inventory or a new full-text review'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; refuse to overwrite')
    result = audit(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('29 records; 28 studies after documented merge; 58 source checks; input unchanged.')
