"""M59 pure finite-array checks. Stdlib only; no real data or model imports."""
from fractions import Fraction as F
from itertools import combinations
import json


def difference(values):
    return [[a-b for b in values] for a in values]


def project(h):
    n=len(h)
    assert n>0 and all(len(row)==n for row in h)
    assert all(h[i][j]==-h[j][i] for i in range(n) for j in range(n))
    return [sum(row,F(0))/n for row in h]


def objective(h,s):
    return sum((s[i]-s[j]-h[i][j])**2 for i,j in combinations(range(len(s)),2))


def main():
    count=0
    for n in range(2,9):
        g=[F((i*7)%11,3) for i in range(n)]
        h=difference(g);s=project(h);center=sum(g)/n
        assert s==[x-center for x in g] and sum(s)==0 and objective(h,s)==0
        for m in range(1,n+1):
            assert sorted(range(n),key=lambda i:(g[i],i))[:m]==sorted(range(n),key=lambda i:(s[i],i))[:m]
        # Pairwise squared fitting of an additive score is centered pointwise fitting.
        loss=[F(i*i,5) for i in range(n)];err=[g[i]-loss[i] for i in range(n)]
        pair_loss=sum((err[i]-err[j])**2 for i,j in combinations(range(n),2))
        mean=sum(err)/n
        assert pair_loss==n*sum((e-mean)**2 for e in err)
        assert pair_loss==n*sum(e*e for e in err)-sum(err)**2
        count+=1
    # Pure circulation survives antisymmetry but yields no point score.
    cycle=[[F(0),F(3),F(-3)],[F(-3),F(0),F(3)],[F(3),F(-3),F(0)]]
    assert project(cycle)==[0,0,0] and objective(cycle,[0,0,0])==27
    truth=[F(0),F(1),F(4)];h=difference(truth)
    plus=[[h[i][j]+cycle[i][j] for j in range(3)] for i in range(3)]
    assert project(plus)==project(h)
    assert all(sum(plus[i][j]-(project(plus)[i]-project(plus)[j]) for j in range(3))==0 for i in range(3))
    for t in [F(-2),F(-1),F(1),F(2)]:
        perturbed=[project(plus)[0]+t,project(plus)[1]-t,project(plus)[2]]
        assert objective(plus,project(plus))<=objective(plus,perturbed)
    # A cycle can produce harmful tie-based acceptance; no universal improvement.
    keys=['z','y','a'];m=1
    accepted=sorted(range(3),key=lambda i:(project(cycle)[i],keys[i]))[:m]
    baseline=sorted(range(3),key=lambda i:(truth[i],keys[i]))[:m]
    assert accepted==[2] and baseline==[0] and truth[accepted[0]]>truth[baseline[0]]
    # Within-fold pairs do not identify free score offsets between components.
    labels=[F(0),F(1),F(4),F(5)]; shifted=[F(10),F(11),F(4),F(5)]
    assert all(shifted[i]-shifted[j]==labels[i]-labels[j] for i,j in [(0,1),(2,3)])
    assert sorted(range(4),key=lambda i:shifted[i])[:2]==[2,3]
    assert sorted(range(4),key=lambda i:labels[i])[:2]==[0,1]
    # Correlated finite loss worlds: common conditioning still gives additive gains.
    worlds=[[F(0),F(4),F(1)],[F(4),F(0),F(1)]]
    mu=[sum(w[i] for w in worlds)/2 for i in range(3)]
    expected_pair=[[sum(w[i]-w[j] for w in worlds)/2 for j in range(3)] for i in range(3)]
    assert expected_pair==difference(mu)
    for m in [1,2,3]:
        chosen=sorted(range(3),key=lambda i:(mu[i],i))[:m]
        best=min(sum(sum(w[i] for i in subset)/m for w in worlds)/2 for subset in combinations(range(3),m))
        assert sum(mu[i] for i in chosen)/m==best
    print(json.dumps(dict(status='passed',exact_fraction_array_cases=count,
        checks=['additive projection reproduces ordering and ties','centered pointwise loss identity',
                'cycle cancellation and least-squares normal equations','harmful cycle tie counterexample',
                'disconnected fold offsets','conditional MSE separability without independence'],
        real_data_rows=0,training_fits=0,model_loads=0,predict_calls=0,official_test_rows_read=0)))


if __name__=='__main__':main()
