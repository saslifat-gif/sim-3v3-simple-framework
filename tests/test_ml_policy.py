import math
import unittest
from unittest.mock import patch

from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D, RobotState
from src.ml_policy import FEATURE_NAMES, encode_state, learned_shot_target, predict_tree
from src.framework.config import SoccerConfig
from src.player import Player
from unittest.mock import Mock


class MLPolicyTests(unittest.TestCase):
    def setUp(self):
        from src.models.match_policy import MODEL
        self.policy_patch = patch.dict(MODEL, enabled=False)
        self.policy_patch.start()
        self.addCleanup(self.policy_patch.stop)

    def test_features_do_not_depend_on_dictionary_order(self):
        state = dict(ball=[1, 2], length=14, width=9, goal_width=2.6, opponents=[[5, 3], [-2, 0], [0, 0]],
                     teammates=[[-5, 0], [2, 3]])
        expected = encode_state(state)
        state['opponents'].reverse()
        state['teammates'].reverse()
        self.assertEqual(expected, encode_state(state))
        self.assertEqual(len(expected), len(FEATURE_NAMES))

    def test_tree_matches_float32_threshold_semantics(self):
        model = {'nodes': [[0, 1.0, 1, 2, 4, 1], [-2, -2, -1, -1, 0, 1], [-2, -2, -1, -1, 8, 1]]}
        self.assertEqual(predict_tree(model, [1.00000001])[0], 0)
        self.assertEqual(predict_tree(model, [1.01])[0], 8)

    def context(self):
        return Context(0, .03, 1, ADULT_FIELD_DIMENSIONS, ball=BallState(0, 3),
                       teammates={i: RobotState(i, Pose2D(-i, 0)) for i in [1, 2, 3]},
                       opponents={i: RobotState(i, Pose2D(i, 0)) for i in [4, 5, 6]})

    def test_low_confidence_prediction_falls_back(self):
        model = dict(min_confidence=.85, nodes=[[-2, -2, -1, -1, 8, .5]])
        with patch('src.ml_policy.load_model', return_value=model):
            self.assertIsNone(learned_shot_target(self.context(), 1))

    def test_missing_opponent_falls_back(self):
        ctx = self.context()
        ctx.opponents.pop(4)
        self.assertIsNone(learned_shot_target(ctx, 1))

    def test_learned_lane_crosses_inside_goal(self):
        model = dict(min_confidence=.85, nodes=[[-2, -2, -1, -1, 8, 1]])
        ctx = self.context()
        with patch('src.ml_policy.load_model', return_value=model):
            tx, ty = learned_shot_target(ctx, 1)
        crossing = ctx.ball.y + (ty-ctx.ball.y) * (7-ctx.ball.x)/(tx-ctx.ball.x)
        self.assertAlmostEqual(crossing, 1.0)
        self.assertTrue(math.isfinite(ty))

    def test_player_uses_model_when_enabled(self):
        player = Player(1, SoccerConfig(), Mock())
        player.context = self.context()
        with patch('src.player.USE_LEARNED_SHOOTING', True), \
                patch('src.player.policy_shot_target', return_value=(7.25, 1.0)), \
                patch('src.player.choose_shot_target') as teacher:
            direction, _ = player.plan_kick()
        teacher.assert_not_called()
        self.assertAlmostEqual(direction, math.atan2(-2, 7.25))

    def test_player_falls_back_and_preserves_explicit_target(self):
        player = Player(1, SoccerConfig(), Mock())
        player.context = self.context()
        with patch('src.player.USE_LEARNED_SHOOTING', True), \
                patch('src.player.policy_shot_target', return_value=None) as learned, \
                patch('src.player.choose_shot_target', return_value=(7.25, 0)) as teacher:
            player.plan_kick()
            learned.assert_called_once()
            teacher.assert_called_once()
            learned.reset_mock()
            player.plan_kick((3, 3))
            learned.assert_not_called()


if __name__ == '__main__':
    unittest.main()
