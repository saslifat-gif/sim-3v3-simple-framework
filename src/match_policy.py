"""Small, dependency-free policy trained by simulator match rewards."""
from __future__ import annotations
import math
from .models.match_policy import MODEL
_rl_parameters = {}


def set_rl_parameters(values):
    global _rl_parameters
    _rl_parameters = dict(values)


def parameter(name, default):
    if name in _rl_parameters:
        value = float(_rl_parameters[name])
        return value if math.isfinite(value) else default
    if not MODEL.get('enabled', False):
        return default
    value = float(MODEL['parameters'].get(name, default))
    return value if math.isfinite(value) else default


def shot_target(ctx, shooter_id, imitation_target):
    if not MODEL.get('enabled', False) or ctx.ball is None:
        return imitation_target
    from .param import GOAL_TARGET_DEPTH_M, SHOT_POST_MARGIN_M
    gx, bx, by = ctx.field.length / 2, ctx.ball.x, ctx.ball.y
    if bx >= gx - 0.05:
        return imitation_target
    half = max(0.0, ctx.field.goal_width / 2 - SHOT_POST_MARGIN_M)
    blockers = [(r.pose, .55) for r in ctx.opponents.values() if r.pose is not None]
    blockers += [(r.pose, .48) for pid, r in ctx.teammates.items()
                 if pid != shooter_id and r.pose is not None]
    prior_y = None
    if imitation_target is not None and imitation_target[0] > bx:
        prior_y = by + (imitation_target[1] - by) * (gx - bx) / (imitation_target[0] - bx)

    def score(y):
        dx, dy = gx - bx, y - by
        norm = dx * dx + dy * dy
        clearance = 1.2
        for pose, radius in blockers:
            t = ((pose.x - bx) * dx + (pose.y - by) * dy) / norm
            if 0 < t <= 1:
                clearance = min(clearance, math.hypot(pose.x - bx - t * dx,
                                                     pose.y - by - t * dy) - radius)
        prior = max(0.0, 1 - abs(y - prior_y) / max(half, .01)) if prior_y is not None else 0
        return clearance - parameter('center_bias', .08) * abs(y) / max(half, .01) + parameter('imitation_bonus', 0) * prior

    crossing = max([half * (i / 4 - 1) for i in range(9)], key=lambda y: (score(y), -abs(y)))
    scale = (gx + GOAL_TARGET_DEPTH_M - bx) / (gx - bx)
    return gx + GOAL_TARGET_DEPTH_M, by + (crossing - by) * scale
