import copy
import json
import math
import unittest
from unittest.mock import patch
import test_rl_policy
from src.rl_policy import TeamPolicy,probabilities
from src.online_learning import update_chunk


class OnlineLearningTests(unittest.TestCase):
    def rows(self):
        model=dict(weights=[[0.]*12 for _ in range(5)],value_weights=[.1]*12,updates=5)
        rows=[]
        for i in range(20):
            x=[1.,i/20]+[0.]*10
            rows.append(dict(features=x,probabilities=probabilities(model['weights'],x),
                             action=i%5,reward=1 if i%5==0 else -.1,done=False,duration_sim_seconds=2))
        return model,rows

    def test_on_policy_update_changes_weights_and_limits_step(self):
        model,rows=self.rows();saved=copy.deepcopy(model)
        learned,stats=update_chunk(model,rows,[1.]*12)
        self.assertEqual(model,saved)
        self.assertEqual(learned['updates'],6)
        self.assertGreater(stats['weight_change_l2'],0)
        self.assertLessEqual(stats['kl'],.01)
        self.assertTrue(all(math.isfinite(v) for row in learned['weights'] for v in row))
        with self.assertRaises(ValueError):update_chunk(learned,rows)

    def test_bootstrap_changes_value_targets(self):
        model,rows=self.rows()
        _,terminal=update_chunk(model,rows)
        _,continuing=update_chunk(model,rows,[1.]*12)
        self.assertGreater(continuing['return_mean'],terminal['return_mean'])

    def test_updates_at_wall_clock_boundary_and_freezes_evaluation(self):
        factory=test_rl_policy.RLPolicyTests()
        for training in (True,False):
            model=factory.model();model.update(training=training,update_interval_seconds=20)
            policy=TeamPolicy(model)
            with patch('src.rl_policy.time.monotonic',return_value=100):policy.step(factory.context())
            for i in range(1,10):
                with patch('src.rl_policy.time.monotonic',return_value=100+i):
                    policy.step(factory.context(600-2*i))
            self.assertEqual(policy.model.get('updates',0),0)
            with patch('src.rl_policy.time.monotonic',return_value=120.1):
                policy.step(factory.context(580,score=1))
            self.assertEqual(policy.model.get('updates',0),1 if training else 0)
            if training:
                self.assertEqual(policy.batch,[])
                self.assertEqual(policy.pending['probabilities'],probabilities(policy.model['weights'],policy.pending['features']))

    def test_terminal_tail_is_not_discarded(self):
        model,rows=self.rows()
        tail=rows[-2:];tail[-1]['done']=True
        learned,stats=update_chunk(model,tail)
        self.assertEqual(learned['updates'],6)
        self.assertEqual(stats['transitions'],2)
