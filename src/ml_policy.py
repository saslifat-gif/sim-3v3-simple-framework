"""Portable inference for an imitation-trained shooting policy (no ML dependencies)."""
from __future__ import annotations

import logging
import math
import struct
from functools import lru_cache

from .param import (GOAL_TARGET_DEPTH_M, SHOT_POST_MARGIN_M, SHOT_TARGET_COUNT,
                    OPPONENT_RADIUS, TEAMMATE_RADIUS, SHOT_CLEARANCE_CAP_M)

FEATURE_NAMES = ['ball_x', 'ball_y'] + [
    f'{kind}{i}_{axis}' for kind, count in [('opponent', 3), ('teammate', 2)]
    for i in range(count) for axis in ['dx', 'dy', 'visible']
] + [f'lane{i}_clearance' for i in range(SHOT_TARGET_COUNT)] + [
    f'lane{i}_minus_lane{j}' for i in range(SHOT_TARGET_COUNT) for j in range(i+1, SHOT_TARGET_COUNT)
]
_log = logging.getLogger(__name__)
_decisions = 0
_accepted = 0


def state_from_context(ctx, shooter_id):
    return dict(ball=[ctx.ball.x, ctx.ball.y], length=ctx.field.length,
                width=ctx.field.width, goal_width=ctx.field.goal_width,
                opponents=[[r.pose.x, r.pose.y] for r in ctx.opponents.values() if r.pose is not None],
                teammates=[[r.pose.x, r.pose.y] for pid, r in ctx.teammates.items()
                           if pid != shooter_id and r.pose is not None])


def encode_state(state):
    bx, by = state['ball']
    length, width = state['length'], state['width']
    features = [2 * bx / length, 2 * by / width]
    for kind, count in [('opponents', 3), ('teammates', 2)]:
        robots = sorted(state[kind], key=lambda p: ((p[0] - bx)**2 + (p[1] - by)**2, p[0], p[1]))
        for i in range(count):
            if i < len(robots):
                features.extend([(robots[i][0] - bx) / length, (robots[i][1] - by) / width, 1.0])
            else:
                features.extend([0.0, 0.0, 0.0])
    # Geometric features describe each finite shot corridor, not its expert
    # action label. Pairwise differences let a small tree learn lane ranking.
    goal_x = length/2
    half = max(0.0, state['goal_width']/2-SHOT_POST_MARGIN_M)
    gaps = []
    blockers = [(p, radius) for kind, radius in [('opponents', OPPONENT_RADIUS), ('teammates', TEAMMATE_RADIUS)]
                for p in state[kind]]
    for i in range(SHOT_TARGET_COUNT):
        y = half*(2*i/(SHOT_TARGET_COUNT-1)-1)
        dx, dy = goal_x-bx, y-by
        length2 = max(dx*dx+dy*dy, 1e-12)
        gap = SHOT_CLEARANCE_CAP_M
        for (px, py), radius in blockers:
            t = ((px-bx)*dx+(py-by)*dy)/length2
            if 0 < t <= 1:
                gap = min(gap, math.hypot(px-bx-t*dx, py-by-t*dy)-radius)
        gaps.append(gap)
    features.extend(gaps)
    features.extend(gaps[i]-gaps[j] for i in range(SHOT_TARGET_COUNT) for j in range(i+1, SHOT_TARGET_COUNT))
    return features


def predict_tree(model, features):
    # sklearn trees compare float32 input with float64 thresholds.
    x = [struct.unpack('f', struct.pack('f', v))[0] for v in features]
    i = 0
    for _ in range(len(model['nodes'])):
        feature, threshold, left, right, label, confidence = model['nodes'][i]
        if left == -1:
            return int(label), float(confidence)
        i = left if x[feature] <= threshold else right
    raise ValueError('Cyclic policy tree')


@lru_cache(maxsize=1)
def load_model():
    try:
        from .models.shot_policy import MODEL
        model = MODEL
        if model['format_version'] != 1 or model['feature_names'] != FEATURE_NAMES:
            raise ValueError('Unsupported policy feature schema')
        _log.info('Imitation shooting policy loaded: %s (%d nodes)', model['model_id'], len(model['nodes']))
        return model
    except (ImportError, ValueError, KeyError) as error:
        _log.warning('Imitation model unavailable; using teacher: %s', error)
        return None


def learned_shot_target(ctx, shooter_id):
    if ctx.ball is None or ctx.ball.x >= ctx.field.length / 2 - 0.05:
        return None
    state = state_from_context(ctx, shooter_id)
    # Missing robots are outside this first model's fully observed training domain.
    if len(state['opponents']) != 3 or len(state['teammates']) != 2:
        return None
    model = load_model()
    if model is None:
        return None
    if model.get('field', [ctx.field.length, ctx.field.width, ctx.field.goal_width]) != [ctx.field.length, ctx.field.width, ctx.field.goal_width]:
        return None
    features = encode_state(state)
    if not all(math.isfinite(v) for v in features):
        return None
    label, confidence = predict_tree(model, features)
    if confidence < model['min_confidence'] or not 0 <= label < SHOT_TARGET_COUNT:
        return None
    half = max(0, ctx.field.goal_width / 2 - SHOT_POST_MARGIN_M)
    crossing_y = half * (2 * label / (SHOT_TARGET_COUNT - 1) - 1)
    goal_x = ctx.field.length / 2
    scale = (goal_x + GOAL_TARGET_DEPTH_M - ctx.ball.x) / (goal_x - ctx.ball.x)
    return goal_x + GOAL_TARGET_DEPTH_M, ctx.ball.y + (crossing_y - ctx.ball.y) * scale


def policy_shot_target(ctx, shooter_id):
    """Fail back to the teacher on model errors and report actual inference use."""
    global _decisions, _accepted
    try:
        target = learned_shot_target(ctx, shooter_id)
    except (ValueError, KeyError, IndexError, OverflowError) as error:
        _log.warning('Imitation inference failed; using teacher: %s', error)
        target = None
    _decisions += 1
    _accepted += target is not None
    if _decisions % 300 == 0:
        _log.info('Imitation policy decisions=%d learned=%d fallback=%d',
                  _decisions, _accepted, _decisions-_accepted)
    return target
