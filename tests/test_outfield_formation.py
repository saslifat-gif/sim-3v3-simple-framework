"""Exercise actual match role assignment without the Docker-only SDK."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D

sdk = SimpleNamespace(AgentBase=type('AgentBase', (), {}), AgentFeatures=object)
spec = importlib.util.spec_from_file_location('src._formation_test_entry',
                                             Path(__file__).resolve().parents[1]/'src/main.py')
entry = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, booster_agent_framework=sdk):
    spec.loader.exec_module(entry)


class OutfieldFormationTests(unittest.TestCase):
    def setUp(self):
        self.ctx = Context(1, .03, 1, ADULT_FIELD_DIMENSIONS, ball=BallState())
        self.players = [Mock(id=i, pose=Pose2D(x,y,0), is_fallen=False)
                        for i,x,y in [(1,-.3,0),(2,-6,-1),(3,-2,1)]]

    def test_normal_uses_two_separate_support_lanes_and_no_keeper(self):
        entry._act_normal(self.ctx,self.players,SimpleNamespace(normal_attacker=None))
        self.players[0].attack.assert_called_once()
        offsets = [p.support.call_args.kwargs['lateral_offset'] for p in self.players[1:]]
        self.assertLess(min(offsets),0)
        self.assertGreater(max(offsets),0)
        for p in self.players:
            p.guard.assert_not_called()

    def test_ready_and_opponent_kickoff_put_everyone_in_midfield(self):
        for action in [entry._act_ready,entry._act_opp_kickoff]:
            for p in self.players:p.reset_mock()
            action(self.ctx,self.players)
            targets=[]
            for p in self.players:
                p.guard.assert_not_called()
                p.walk_to.assert_called_once()
                target=p.walk_to.call_args.args[0]
                self.assertLess(target[0],0)
                self.assertGreater(target[0],-4)
                targets.append(target)
            self.assertEqual(len(set(targets)),3)

    def test_our_kickoff_releases_both_supporters_from_goal(self):
        store=SimpleNamespace(prev_phase=None,kickoff_taker=None)
        entry._act_our_kickoff(self.ctx,self.players,store)
        self.players[0].kick.assert_called_once()
        for p in self.players[1:]:
            p.guard.assert_not_called()
            p.walk_to.assert_called_once()
            self.assertGreater(p.walk_to.call_args.args[0][0],-4)

    def test_opponent_restart_keeps_all_three_outside_exclusion_circle(self):
        entry._act_opp_set_play(self.ctx,self.players,SimpleNamespace())
        for p in self.players:
            p.guard.assert_not_called()
            p.walk_to.assert_called_once()
            x,y=p.walk_to.call_args.args[0]
            self.assertGreaterEqual((x*x+y*y)**.5,2)


if __name__=='__main__':
    unittest.main()
