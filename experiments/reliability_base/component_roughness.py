"""Prospective local component aggregation; no query activity argument.
Equal-component averaging is established, not claimed as a new weighting theorem.
"""
from collections import Counter,defaultdict
import numpy as np

def summarize(neighbor_activity,pair_similarity,component_ids):
    y=np.asarray(neighbor_activity,dtype=float);s=np.asarray(pair_similarity,dtype=float);g=np.asarray(component_ids)
    assert y.ndim==1 and len(y)>=2 and g.shape==y.shape and s.shape==(len(y),len(y))
    assert np.issubdtype(g.dtype,np.integer), "component IDs must be integer labels"
    assert np.isfinite(y).all() and np.isfinite(s).all() and ((s>=0)&(s<=1)).all() and np.allclose(s,s.T)
    _,group=np.unique(g,return_inverse=True);counts=Counter(group.tolist());number=len(counts)
    weights=np.array([1/(number*counts[c]) for c in group]);assert np.isclose(weights.sum(),1)
    mean=weights@y;dispersion=float(np.sqrt(max(0.,weights@((y-mean)**2))))
    blocks=defaultdict(list);values=[]
    for a in range(len(y)):
        for b in range(a+1,len(y)):
            value=abs(y[a]-y[b])/max(1-s[a,b],1e-3)
            blocks[tuple(sorted((int(group[a]),int(group[b]))))].append(value);values.append(value)
    return {'cb_disp':dispersion,'cb_sali':float(np.mean([np.mean(v) for v in blocks.values()])),
            'component_count':number,'component_max_fraction':max(counts.values())/len(y),
            'raw_disp':float(np.std(y)),'raw_sali':float(np.mean(values))}

def selfcheck():
    y=np.array([0.,0.,0.,0.,10.]);s=np.eye(5);mixed=summarize(y,s,[0,0,0,0,1])
    assert np.isclose(mixed['cb_disp'],5.) and np.isclose(mixed['raw_disp'],4.)
    assert np.isclose(mixed['cb_sali'],5.) and np.isclose(mixed['raw_sali'],4.)
    for groups in [np.zeros(5,dtype=int),np.arange(5)]:
        result=summarize(y,s,groups)
        assert np.isclose(result['cb_disp'],result['raw_disp']) and np.isclose(result['cb_sali'],result['raw_sali'])
    permutation=np.array([4,2,0,3,1]);g=np.array([0,0,0,0,1]);changed=summarize(y[permutation]+9,s[np.ix_(permutation,permutation)],g[permutation]+100)
    assert all(np.isclose(mixed[k],changed[k]) for k in mixed)
    try:summarize([np.nan,1],np.eye(2),[0,1])
    except AssertionError:pass
    else:raise AssertionError('nonfinite trust boundary not checked')
    print('PASS: exact mixed-component fixture, raw limits, permutation/label/offset invariance, finite input guard')

if __name__=='__main__':selfcheck()
