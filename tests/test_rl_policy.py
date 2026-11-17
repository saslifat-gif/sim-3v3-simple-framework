import json
import math
from dataclasses import replace
import unittest
from src.framework.types import (ADULT_FIELD_DIMENSIONS,BallState,Context,GameControlState,
                                 GameState,Pose2D,RobotState,TeamState,PlayerState,Penalty)
from src.match_policy import parameter,set_rl_parameters
from src.rl_policy import TeamPolicy,features,probabilities


class RLPolicyTests(unittest.TestCase):
    def tearDown(self):set_rl_parameters({})

    def context(self,remaining=600,state=GameState.PLAYING,score=0):
        game=GameControlState(state=state,secs_remaining=remaining,
                              teams=(TeamState(team_number=1,score=score),TeamState(team_number=2)))
        return Context(1,.03,1,ADULT_FIELD_DIMENSIONS,game=game,ball=BallState(),
                       teammates={i:RobotState(i,Pose2D(-i,0)) for i in [1,2,3]},
                       opponents={i:RobotState(i,Pose2D(i,0)) for i in [1,2,3]})

    def model(self):
        weights=[[0.]*12 for _ in range(5)]
        weights[1][0]=2
        return dict(enabled=True,training=False,weights=weights,episode='test',seed=1)

    def test_features_and_stable_softmax(self):
        x=features(self.context())
        self.assertEqual(len(x),12)
        probs=probabilities([[1000.]*12 for _ in range(5)],x)
        self.assertAlmostEqual(sum(probs),1)
        self.assertTrue(all(math.isfinite(p) for p in probs))
        with self.assertRaises(ValueError):probabilities([[0]],x)

    def test_real_goal_reward_and_simulation_time_decision(self):
        policy=TeamPolicy(self.model())
        policy.step(self.context())
        self.assertEqual(parameter('near_power',5),8)
        with self.assertLogs('src.rl_policy',level='INFO') as captured:
            policy.step(self.context(598,score=1))
        row=json.loads(captured.output[0].split('RL_TRANSITION ',1)[1])
        self.assertEqual(row['reward'],10)
        self.assertEqual(row['action'],1)
        self.assertEqual(row['duration_sim_seconds'],2)

    def test_terminal_marker_and_missing_observations(self):
        policy=TeamPolicy(self.model())
        self.assertIsNone(features(replace(self.context(),opponents={})))
        policy.step(self.context())
        with self.assertLogs('src.rl_policy',level='INFO') as captured:
            policy.step(self.context(0,GameState.FINISHED))
        self.assertTrue(any('RL_EPISODE_END ' in line for line in captured.output))
        self.assertTrue(policy.finished)

    def test_previous_match_finished_packet_does_not_end_new_rollout(self):
        policy=TeamPolicy(self.model())
        policy.step(self.context(0,GameState.FINISHED,score=2))
        self.assertFalse(policy.finished)
        self.assertFalse(policy.started)
        policy.step(self.context())
        self.assertTrue(policy.started)
        with self.assertLogs('src.rl_policy',level='INFO') as captured:
            policy.step(self.context(598,score=1))
        self.assertTrue(any('RL_TRANSITION ' in line for line in captured.output))

    def test_goal_and_penalty_during_restart_are_not_dropped(self):
        policy=TeamPolicy(self.model())
        policy.step(self.context())
        policy.step(self.context(599,GameState.READY))
        ready=self.context(598,GameState.READY,score=1)
        ours=replace(ready.game.teams[0],players=(PlayerState(penalty=Penalty.MOTION_IN_SET),)*1+(PlayerState(),)*19)
        ready=replace(ready,game=replace(ready.game,teams=(ours,ready.game.teams[1])))
        policy.step(ready)
        with self.assertLogs('src.rl_policy',level='INFO') as captured:
            policy.step(self.context(580,GameState.FINISHED,score=1))
        row=json.loads(next(s.split('RL_TRANSITION ',1)[1] for s in captured.output if 'RL_TRANSITION ' in s))
        self.assertEqual(row['environment_reward'],9.5)
        self.assertEqual(policy.observed_reward,policy.emitted_reward)
        self.assertEqual(policy.unassigned_reward,0)
        self.assertTrue(row['done'])

    def test_missing_observations_keep_prior_action_until_recovery(self):
        policy=TeamPolicy(self.model());policy.step(self.context())
        policy.step(replace(self.context(598,score=1),ball=None))
        self.assertEqual(policy.sequence,0)
        with self.assertLogs('src.rl_policy',level='INFO') as captured:policy.step(self.context(596,score=1))
        row=json.loads(captured.output[0].split('RL_TRANSITION ',1)[1])
        self.assertEqual(row['environment_reward'],10)

    def test_reward_before_first_action_is_carried_forward(self):
        policy=TeamPolicy(self.model());policy.step(self.context(state=GameState.READY))
        policy.step(self.context(599,GameState.READY,score=1))
        self.assertEqual(policy.unassigned_reward,10)
        policy.step(self.context(598,score=1))
        self.assertEqual(policy.pending['reward'],10)
        self.assertEqual(policy.unassigned_reward,0)
