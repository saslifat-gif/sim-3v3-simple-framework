import math
import unittest
from unittest.mock import Mock, patch
from src.framework.config import SoccerConfig
from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D, RobotState
from src.models.match_policy import MODEL
from src.match_policy import parameter, shot_target
from src.player import Player
from src.utils.tactics import guard_target


class MatchPolicyTests(unittest.TestCase):
    def context(self, ball=None):
        return Context(1, .03, 1, ADULT_FIELD_DIMENSIONS,
                       ball=ball if ball is not None else BallState(),
                       teammates={1: RobotState(1, Pose2D(-.5, 0, 0))})

    def test_disabled_checkpoint_preserves_prior(self):
        with patch.dict(MODEL, enabled=False):
            self.assertEqual(shot_target(self.context(), 1, (7.25, .4)), (7.25, .4))
            self.assertEqual(parameter('near_power', 5), 5)

    def test_learned_shot_stays_inside_posts(self):
        with patch.dict(MODEL, enabled=True, parameters={'center_bias': .1, 'imitation_bonus': .2}):
            for bx in (-6, 0, 6.9):
                for by in (-4, 0, 4):
                    target = shot_target(self.context(BallState(bx, by)), 1, (7.25, 0))
                    crossing = by + (target[1] - by) * (7 - bx) / (target[0] - bx)
                    self.assertLessEqual(abs(crossing), 1 + 1e-8)

    def test_learned_power_reaches_backend(self):
        backend = Mock()
        player = Player(1, SoccerConfig(), backend)
        player.context = self.context()
        with patch.dict(MODEL, enabled=True, parameters={'near_power': 8, 'far_power': 9}):
            player.attack((7.25, 0))
        self.assertEqual(backend.kick.call_args.args[1], 8)

    def test_keeper_depth_and_invalid_parameter(self):
        with patch.dict(MODEL, enabled=True, parameters={'keeper_depth': .8, 'near_power': math.nan}):
            self.assertAlmostEqual(guard_target(self.context())[0], -6.2)
            self.assertEqual(parameter('near_power', 5), 5)


if __name__ == '__main__':
    unittest.main()
