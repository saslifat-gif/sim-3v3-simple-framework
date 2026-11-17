import copy
import unittest
import numpy as np
from policy_gradient import policy_gradient,softmax,update


class GradientTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(3)
        w=rng.normal(0,.1,(5,12));x=rng.normal(0,.2,(9,12))
        a=np.array([0,1,2,3,4,0,1,2,3]);adv=rng.normal(size=9)
        analytic=policy_gradient(w,x,a,adv)
        def objective(weights):
            p=softmax(x@weights.T)
            return (np.log(p[np.arange(len(a)),a])*adv).mean()+.01*(-p*np.log(p)).sum(axis=1).mean()
        numeric=np.zeros_like(w)
        for i in range(5):
            for j in range(12):
                hi=w.copy();lo=w.copy();hi[i,j]+=1e-6;lo[i,j]-=1e-6
                numeric[i,j]=(objective(hi)-objective(lo))/2e-6
        np.testing.assert_allclose(analytic,numeric,atol=1e-7)

    def test_actual_update_is_finite_bounded_and_on_policy(self):
        model=dict(weights=[[0.]*12 for _ in range(5)],value_weights=[0.]*12)
        rows=[]
        for i in range(40):
            x=[1.,i/40]+[0.]*10
            rows.append(dict(features=x,action=i%5,probabilities=[.2]*5,
                             reward=10. if i==30 else 0.,duration_sim_seconds=2,done=i==39))
        learned,stats=update(model,rows)
        self.assertGreater(stats['weight_change_l2'],0)
        self.assertLessEqual(stats['kl'],.02)
        self.assertTrue(np.isfinite(learned['weights']).all())
        broken=copy.deepcopy(rows);broken[0]['probabilities']=[1,0,0,0,0]
        with self.assertRaises(ValueError):update(model,broken)


if __name__=='__main__':unittest.main()
