"""Pure geometry for shot selection and conservative goalkeeper positioning."""

from __future__ import annotations

import math

from ..framework.types import Context, SetPlay
from ..param import (
    GOAL_TARGET_DEPTH_M, GUARD_POST_MARGIN_M, OPPONENT_RADIUS,
    SHOT_CENTER_BIAS, SHOT_CLEARANCE_CAP_M, SHOT_POST_MARGIN_M,
    SHOT_TARGET_COUNT, TEAMMATE_RADIUS, RESTART_CLEARANCE_MARGIN_M,
)
from .geom import clamp, opponent_goal, own_goal_area_center


def choose_shot_target(ctx: Context, shooter_id: int) -> tuple[float, float]:
    """Choose a goal-mouth lane using currently observed robots only.

    Clearance is measured along the finite ball-to-goal segment. Prefer the
    center once a lane has ample space; robots behind the ball cannot block it.
    Extend the chosen ray beyond the line without changing its crossing point.
    """
    ball = ctx.ball
    goal_x = ctx.field.length / 2.0
    if ball is None or ball.x >= goal_x - 1e-6:
        return opponent_goal(ctx)
    half = max(0.0, ctx.field.goal_width / 2.0 - SHOT_POST_MARGIN_M)
    blockers = [
        (r.pose, radius)
        for robots, radius in (
            (ctx.opponents.values(), OPPONENT_RADIUS),
            ((r for pid, r in ctx.teammates.items() if pid != shooter_id), TEAMMATE_RADIUS),
        )
        for r in robots if r.pose is not None
    ]
    count = max(3, SHOT_TARGET_COUNT | 1)
    candidates = [half * (2.0 * i / (count - 1) - 1.0) for i in range(count)]

    def score(y: float) -> float:
        dx, dy = goal_x - ball.x, y - ball.y
        length2 = dx * dx + dy * dy
        clearance = SHOT_CLEARANCE_CAP_M
        for pose, radius in blockers:
            t = ((pose.x - ball.x) * dx + (pose.y - ball.y) * dy) / length2
            if t <= 0.0 or t > 1.0:
                continue
            gap = math.hypot(pose.x - ball.x - t * dx, pose.y - ball.y - t * dy) - radius
            clearance = min(clearance, gap)
        return clearance - SHOT_CENTER_BIAS * abs(y) / max(half, 1e-6)

    target_y = max(candidates, key=lambda y: (score(y), -abs(y)))
    scale = (goal_x + GOAL_TARGET_DEPTH_M - ball.x) / (goal_x - ball.x)
    return goal_x + GOAL_TARGET_DEPTH_M, ball.y + (target_y - ball.y) * scale


def guard_target(ctx: Context) -> tuple[float, float]:
    """Cover the ball-to-goal-center line while staying near the goal mouth."""
    home_x, _ = own_goal_area_center(ctx)
    from ..match_policy import parameter
    home_x = -ctx.field.length / 2 + parameter('keeper_depth', home_x + ctx.field.length / 2)
    ball = ctx.ball
    if ball is None:
        return home_x, 0.0
    goal_x = -ctx.field.length / 2.0
    if ball.x > home_x:
        y = ball.y * (home_x - goal_x) / (ball.x - goal_x)
    else:
        y = ball.y
    half = max(0.0, ctx.field.goal_width / 2.0 - GUARD_POST_MARGIN_M)
    return home_x, clamp(y, -half, half)


def restart_target(ctx: Context, desired: tuple[float, float],
                   current: tuple[float, float]) -> tuple[float, float]:
    """Keep opponent-restart targets in bounds, clear of ball and goal-kick area.

    If already too close, prefer the nearest legal point reached by moving
    away from the ball. This avoids walking through the ball to get behind it.
    """
    field, ball = ctx.field, ctx.ball
    hx, hy = field.length / 2 - 0.35, field.width / 2 - 0.35
    desired = (clamp(desired[0], -hx, hx), clamp(desired[1], -hy, hy))
    if ball is None:
        return desired
    radius = field.circle_radius + RESTART_CLEARANCE_MARGIN_M
    goal_kick = ctx.game is not None and ctx.game.set_play == SetPlay.GOAL_KICK

    def legal(point):
        x, y = point
        return (abs(x) <= hx and abs(y) <= hy
                and math.hypot(x - ball.x, y - ball.y) >= radius
                and not (goal_kick and x >= field.length / 2 - field.penalty_area_length - 0.35
                         and abs(y) <= field.penalty_area_width / 2 + 0.35))

    retreating = not legal(current)
    if not retreating and legal(desired):
        return desired
    ring = radius + 0.2
    candidates = [desired] + [
        (ball.x + ring * math.cos(i * math.tau / 48),
         ball.y + ring * math.sin(i * math.tau / 48)) for i in range(48)
    ] + [(hx * (i / 4 - 1), hy * (j / 4 - 1)) for i in range(9) for j in range(9)]
    candidates = [point for point in candidates if legal(point)]
    if retreating:
        outward = [point for point in candidates
                   if ((current[0] - ball.x) * (point[0] - current[0])
                       + (current[1] - ball.y) * (point[1] - current[1])) >= 0]
        candidates = outward or candidates
    reference = current if retreating else desired
    return min(candidates, key=lambda point: math.dist(point, reference)) if candidates else desired
