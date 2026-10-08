"""Metadata/source hashes only: no label/prediction/feature parsing, imports or fits."""
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/reliability_base'))
from prepare_conditional_pilot import jobs_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/'artifacts/conditional_pilot_inputs_20261008_v2'
    protocol_path = ROOT/'docs/research/conditional_pilot_prepare_20261008/protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    for name, expected in protocol['input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    prior = json.loads((ROOT/'docs/research/conditional_pilot_run_20261008/verification.json').read_text(encoding='utf-8'))
    for name, expected in prior['evidence_hashes'].items():
        assert sha(output/name) == expected, name
    prepared = json.loads((output/'inputs.json').read_text(encoding='utf-8'))
    for name, expected in prepared['files'].items():
        assert sha(output/name) == expected, name
    roles = json.loads((ROOT/protocol['private_roles_manifest']).read_text(encoding='utf-8'))
    jobs = json.loads((output/'jobs.json').read_text(encoding='utf-8'))
    assert jobs == jobs_for(roles, protocol, output)
    component = {r['source_row']: r['component'] for r in roles}
    calibration = {r['source_row'] for r in roles if r['role']=='calibration'}
    full = jobs[-1]
    assert full['name'] == 'full' and len(full['query']) == 585 and len(full['fit']) == 1403
    rows, cache_hashes = [], {}
    for job in jobs:
        reference, query = set(job['fit']), set(job['query'])
        assert reference.isdisjoint(query) and calibration.isdisjoint(reference | query | set(job['monitor']))
        assert reference <= set(full['fit'])
        overlap = query & set(full['query'])
        if job['name'] != 'full':
            assert not overlap
        rows.append(dict(job=job['name'], reference_rows=len(reference), query_rows=len(query),
             reference_components=len({component[i] for i in reference}),
             query_components=len({component[i] for i in query}),
             queries_shared_with_full=len(overlap)))
        cache = output/protocol['task']/job['name']/'features/cache/fixture.csv'
        assert cache.is_file()
        cache_hashes[job['name']] = sha(cache)  # Byte hash, never parse cached feature values.
    assert rows[0]['reference_rows'] == 935 and rows[1]['reference_rows'] == 935 and rows[2]['reference_rows'] == 936
    # Existing fold cache queries differ from evaluation; the required paired context is absent.
    assert all(r['queries_shared_with_full'] == 0 for r in rows[:3])
    # Elementary sanity examples, not a proposed algorithm or a chemical simulation.
    full_sim = [1-i*.01 for i in range(11)]
    subset_sim = full_sim[1:]
    assert max(full_sim) >= max(subset_sim) and mean(full_sim[:10]) >= mean(subset_sim[:10])
    assert pstdev([0]*9+[10]) > pstdev([0]*10)
    assert pstdev([0]*10) < pstdev([0]*9+[10])
    future = dict(query_rows=585, reference_rows=935, reference_job='fold0',
                  candidate_methods=0, chemprop_fits=0, chemprop_predictions=0,
                  auxiliary_rf_fits=1, auxiliary_feature_batches=1, auxiliary_query_rows=585,
                  risk_rf_fits=0, risk_predictions=0, kde_fits=0, scaler_fits=0, pca_fits=0,
                  feature_seconds_cap=300, whole_run_seconds_cap=600,
                  automatic_retries=0, target_label_reads=0, calibration_reads=0,
                  executed=False)
    result = dict(milestone='M67',status='passed',role_table=rows,
          paired_query_contexts_available=0,cache_hashes_observed_now=cache_hashes,
          metadata_and_prior_evidence_hashes_checked=True,
          new_fits=0,new_predictions=0,new_model_loads=0,real_feature_recomputations=0,
          label_value_parsing=False,prediction_value_parsing=False,feature_value_parsing=False,
          official_test_record_parsing=False,calibration_record_parsing=False,
          future_minimal_check=future,
          audit_history=['metadata check passed', 'added synthetic sanity examples; expanded check passed',
                         'source access records added; final metadata check passed'],
          access_limits=['expression-skill missing', 'guessed refinement report path missing; located actual records with rg',
                         'PMC full text blocked by captcha; used author arXiv abstract'],
          literature=[dict(url='https://www.jmlr.org/beta/papers/v8/sugiyama07a.html',scope='official abstract'),
                      dict(url='https://arxiv.org/abs/2104.00673',scope='author abstract and metadata only')],
          synthetic_checks=dict(top_k_similarity_monotonic_example=True,
                                roughness_can_increase_or_decrease=True,estimator_fits=0),
          input_hashes=dict(protocol=sha(protocol_path),jobs=sha(output/'jobs.json'),
              roles=sha(ROOT/protocol['private_roles_manifest']),runner=sha(ROOT/'experiments/reliability_base/run_conditional_pilot.py')))
    (Path(__file__).parent/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(role_table=rows,paired_query_contexts_available=0,new_fits=0,new_predictions=0)))


if __name__ == '__main__':
    main()
