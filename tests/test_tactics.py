import math
import unittest
from unittest.mock import Mock, patch

from src.framework.config import SoccerConfig
from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D, RobotState, GameControlState, SetPlay
from src.param import GUARD_POST_MARGIN_M, SHOT_POST_MARGIN_M
from src.player import Player
from src.utils.tactics import choose_shot_target, guard_target, restart_target
from src.utils.obstacles import collect_obstacles


def context(ball=BallState(), opponents=None):
    return Context(
        now=1.0, dt=1 / 30, team_id=1, field=ADULT_FIELD_DIMENSIONS,
        ball=ball, teammates={1: RobotState(1, Pose2D(-0.5, 0, 0))},
        opponents=opponents or {},
    )


class TacticsTests(unittest.TestCase):
    def setUp(self):
        # These regressions cover the original coded tactics independently of
        # whichever candidate the training process has currently exported.
        from src.models.match_policy import MODEL
        self.policy_patch = patch.dict(MODEL, enabled=False)
        self.policy_patch.start()
        self.addCleanup(self.policy_patch.stop)

    def test_missing_ball_stops_support_and_releases_kick(self):
        backend = Mock()
        player = Player(1, SoccerConfig(), backend)
        player.context = context(ball=None)
        player._kicking = True
        player.support()
        backend.release_kick.assert_called_once()
        backend.set_velocity.assert_called_once_with(0.0, 0.0, 0.0)
        self.assertFalse(player.is_kicking)

    def test_open_goal_prefers_center(self):
        ctx = context()
        self.assertAlmostEqual(choose_shot_target(ctx, 1)[1], 0.0)

    def test_central_keeper_causes_off_center_shot(self):
        ctx = context(opponents={4: RobotState(4, Pose2D(6.5, 0, 0))})
        self.assertGreater(abs(choose_shot_target(ctx, 1)[1]), 0.5)

    def test_robot_behind_ball_does_not_change_shot(self):
        ctx = context(opponents={4: RobotState(4, Pose2D(-0.1, 0, 0))})
        self.assertAlmostEqual(choose_shot_target(ctx, 1)[1], 0.0)

    def test_teammate_blocks_lane_but_shooter_does_not(self):
        ctx = context()
        ctx.teammates[2] = RobotState(2, Pose2D(6.5, 0, 0))
        self.assertGreater(abs(choose_shot_target(ctx, 1)[1]), 0.5)

    def test_off_axis_shot_crosses_inside_posts(self):
        for bx in (-5.0, 0.0, 6.9):
            for by in (-4.0, 4.0):
                ctx = context(BallState(bx, by), {4: RobotState(4, Pose2D(6.8, 0, 0))})
                tx, ty = choose_shot_target(ctx, 1)
                goal_x = ctx.field.length / 2
                crossing_y = by + (ty - by) * (goal_x - bx) / (tx - bx)
                self.assertLessEqual(abs(crossing_y), ctx.field.goal_width / 2 - SHOT_POST_MARGIN_M + 1e-8)

    def test_explicit_attack_target_reaches_backend(self):
        backend = Mock()
        player = Player(1, SoccerConfig(), backend)
        player.context = context()
        player.attack((4.0, 2.0))
        self.assertAlmostEqual(backend.kick.call_args.args[0], math.atan2(2, 4))

    def test_guard_tracks_both_sides_and_stays_inside_posts(self):
        for bx in (-8.0, -6.5, -6.0, 0.0, 5.0):
            for by in (-4.0, 4.0):
                ctx = context(BallState(bx, by))
                x, y = guard_target(ctx)
                self.assertGreater(x, -ctx.field.length / 2)
                self.assertGreater(y * by, 0)
                self.assertLessEqual(abs(y), ctx.field.goal_width / 2 - GUARD_POST_MARGIN_M)

    def test_guard_returns_home_when_ball_missing(self):
        self.assertEqual(guard_target(context(ball=None)), (-6.5, 0.0))

    def test_restart_targets_clear_ball_even_near_corners(self):
        for bx in (-6.7, 0, 6.7):
            for by in (-4.3, 0, 4.3):
                ctx = context(BallState(bx, by))
                x, y = restart_target(ctx, (bx, by), (bx - 0.1, by))
                self.assertLessEqual(abs(x), 6.65)
                self.assertLessEqual(abs(y), 4.15)
                self.assertGreaterEqual(math.hypot(x - bx, y - by), 2.0)

    def test_goal_kick_retreat_leaves_opponent_penalty_area(self):
        from dataclasses import replace
        ctx = replace(context(BallState(6, 2)), game=GameControlState(set_play=SetPlay.GOAL_KICK))
        for current in ((5.5, 2), (6.4, 2), (5.5, -2), (6, 2)):
            x, y = restart_target(ctx, (5, 1), current)
            self.assertFalse(x >= 3.65 and abs(y) <= 3.35)
            self.assertGreaterEqual(math.hypot(x - 6, y - 2), 2)

    def test_retreat_does_not_walk_through_ball(self):
        ctx = context(BallState(0, 0))
        for current in ((0.2, 0), (-0.2, 0), (0, 0.2)):
            target = restart_target(ctx, (-3, 0), current)
            radial = sum(current[i] * (target[i] - current[i]) for i in (0, 1))
            self.assertGreaterEqual(radial, 0)

    def test_legal_restart_station_is_kept(self):
        self.assertEqual(restart_target(context(), (-3, 1), (-4, 1)), (-3, 1))

    def test_restart_without_ball_keeps_target_in_bounds(self):
        self.assertEqual(restart_target(context(ball=None), (-20, 20), (0, 0)), (-6.65, 4.15))

    def test_restart_obstacle_uses_full_avoidance_radius(self):
        obstacles = collect_obstacles(context(), 1, ball=True, robots=False, ball_radius=2.0)
        self.assertEqual(obstacles[0].radius, 2.0)


if __name__ == '__main__':
    unittest.main()
