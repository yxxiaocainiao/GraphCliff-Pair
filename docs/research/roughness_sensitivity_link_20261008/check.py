"""Four predeclared descriptive correlations; no fits, predictions or new features."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/reliability_base'))
from run_conditional_pilot import read_selected, SOURCE, SOURCE_SHA
from run_phase_a import sha, save

PUBLIC = Path(__file__).parent
BASE = ROOT/'artifacts/conditional_pilot_inputs_20261008_v2'
CONTEXT = ROOT/'artifacts/reference_context_check_20261008'
OUTPUT = ROOT/'artifacts/roughness_sensitivity_link_20261008'


def correlation(a,b):
    if min(a.nunique(),b.nunique()) < 2:
        return None
    return float(a.corr(b,method='spearman'))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    state=dict(status='checking',fits=0,predictions=0,target_rows_read=0)
    save(OUTPUT/'state.json',state)
    try:
        protocol=json.loads((PUBLIC/'protocol.json').read_text(encoding='utf-8'))
        for name,expected in protocol['input_sha256'].items():
            assert sha(ROOT/name)==expected,name
        assert sha(SOURCE)==SOURCE_SHA
        prior=json.loads((ROOT/'docs/research/conditional_pilot_run_20261008/verification.json').read_text(encoding='utf-8'))
        for name,expected in prior['evidence_hashes'].items():
            assert sha(BASE/name)==expected,name
        context=json.loads((ROOT/'docs/research/reference_context_check_20261008/verification.json').read_text(encoding='utf-8'))
        for key,path in [('new_cache',CONTEXT/'features/cache/fixture.csv'),
                         ('full_cache',BASE/'CHEMBL234_Ki/full/features/cache/fixture.csv'),
                         ('fixed_predictions',CONTEXT/'fixed_predictions.csv')]:
            assert sha(path)==context['evidence_hashes'][key],key
        roles=json.loads((BASE/'jobs.json').read_text(encoding='utf-8'))[-1]
        scores=pd.read_csv(BASE/'scores.csv',index_col='source_row',usecols=['source_row','RF7','IW-COND7'])
        assert scores.index.tolist()==roles['query'] and len(scores)==585 and np.isfinite(scores).all().all()
        ids=pd.read_csv(BASE/'identities.csv',index_col='source_row',usecols=['source_row','smiles','canonical'])
        original=pd.read_csv(BASE/'CHEMBL234_Ki/full/features/cache/fixture.csv',usecols=['smiles']+protocol['features'])
        changed=pd.read_csv(CONTEXT/'features/cache/fixture.csv',usecols=['smiles']+protocol['features'])
        prediction=pd.read_csv(CONTEXT/'fixed_predictions.csv',usecols=['smiles','y'])
        expected_smiles=ids.loc[scores.index,'smiles'].tolist()
        assert original.smiles.tolist()==changed.smiles.tolist()==prediction.smiles.tolist()==expected_smiles
        sensitivity=(changed[protocol['features']]-original[protocol['features']]).abs()
        sensitivity.index=scores.index
        assert np.isfinite(sensitivity).all().all()
        # Hashes/protocol/scores fixed above; only the already-used evaluation truth is opened.
        truth=read_selected(SOURCE,scores.index.tolist(),['y']).y
        state['target_rows_read']=585
        save(OUTPUT/'state.json',state)
        error2=pd.Series(np.square(prediction.y.to_numpy()-truth.to_numpy()),index=scores.index)
        assert np.isfinite(error2).all()
        contribution=pd.Series(0.,index=scores.index)
        originals=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
        deltas=[]
        for k,fraction in enumerate(protocol['coverage_grid']):
            count=math.ceil(fraction*len(scores))
            accepts={}
            for arm in ['RF7','IW-COND7']:
                order=sorted(scores.index,key=lambda i:(scores.loc[i,arm],ids.loc[i,'canonical'],int(i)))
                accepts[arm]=pd.Series(scores.index.isin(order[:count]).astype(int),index=scores.index)
                rmse=float(np.sqrt(error2.loc[order[:count]].mean()))
                assert np.isclose(rmse,originals[arm]['curve_rmse'][k],atol=1e-12,rtol=0)
            term=error2*(accepts['IW-COND7']-accepts['RF7'])/count
            delta=originals['IW-COND7']['curve_rmse'][k]**2-originals['RF7']['curve_rmse'][k]**2
            assert np.isclose(term.sum(),delta,atol=1e-12,rtol=0)
            if fraction==1:
                assert term.eq(0).all()
            contribution+=term/len(protocol['coverage_grid'])
            deltas.append(delta)
        assert np.isclose(contribution.sum(),np.mean(deltas),atol=1e-12,rtol=0)
        rows=[]
        for feature in protocol['features']:
            for endpoint,values in [('signed_exchange_contribution',contribution),('point_squared_error',error2)]:
                rows.append(dict(feature=feature,endpoint=endpoint,rows=585,
                                 spearman=correlation(sensitivity[feature],values)))
        private=sensitivity.assign(error2=error2,signed_exchange_contribution=contribution)
        private.to_csv(OUTPUT/'pairs.csv',index_label='source_row')
        state['status']='completed'
        save(OUTPUT/'state.json',state)
        result=dict(milestone='M69',status='passed',correlations=rows,
             summed_contribution=float(contribution.sum()),mean_six_mse_difference=float(np.mean(deltas)),
             six_mse_differences=deltas,zero_contribution_rows=int(contribution.eq(0).sum()),
             positive_contribution_rows=int(contribution.gt(0).sum()),negative_contribution_rows=int(contribution.lt(0).sum()),
             target_rows_read=585,new_fits=0,new_predictions=0,model_loads=0,real_feature_recomputations=0,
             official_test_record_reads=0,calibration_record_reads=0,threshold_searches=0,
             checks=dict(curves_replayed=True,contribution_identity=True,full_acceptance_zero=True,
                         score_config_hashes=True,query_order=True),
             protocol_sha256=sha(PUBLIC/'protocol.json'),runner_sha256=sha(Path(__file__)),
             private_pairs_sha256=sha(OUTPUT/'pairs.csv'))
        save(PUBLIC/'verification.json',result)
        print(json.dumps(result))
    except BaseException as error:
        state.update(status='failed_or_interrupted',error=repr(error))
        save(OUTPUT/'state.json',state)
        raise


if __name__=='__main__':
    # Average ranks (ties), an uninformative constant, and opposite signed effects.
    assert np.isclose(correlation(pd.Series([0,0,1]),pd.Series([1,1,0])),-1)
    assert correlation(pd.Series([0,0,0]),pd.Series([0,1,2])) is None
    main()
