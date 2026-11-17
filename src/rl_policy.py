"""Centralized team tactics policy; standard-library inference and rollout logs.

Walking, kicking primitives and referee legality remain in the existing agent.
Training uses wall-clock chunks; evaluation holds the policy fixed.
"""
from __future__ import annotations
import json
import logging
import math
import random
import copy
import time
from .framework.types import GameState, Penalty
from .match_policy import set_rl_parameters
from .models.rl_policy import MODEL

FEATURE_COUNT=12
PROFILES=[
    dict(name='balanced',near_power=7.,far_power=8.,support_distance=1.6,support_lane_spacing=1.8,pressure_radius=1.),
    dict(name='press',near_power=8.,far_power=9.,support_distance=.9,support_lane_spacing=1.4,pressure_radius=1.6),
    dict(name='wide',near_power=7.5,far_power=8.5,support_distance=1.2,support_lane_spacing=3.,pressure_radius=.8),
    dict(name='patient',near_power=5.5,far_power=7.,support_distance=2.2,support_lane_spacing=2.,pressure_radius=.5),
    dict(name='counter',near_power=9.,far_power=9.5,support_distance=2.4,support_lane_spacing=2.4,pressure_radius=1.3),
]
_log=logging.getLogger(__name__)


def probabilities(weights, features):
    if len(features)!=FEATURE_COUNT or len(weights)!=len(PROFILES):
        raise ValueError('RL checkpoint shape mismatch')
    if any(len(row)!=FEATURE_COUNT for row in weights):
        raise ValueError('RL checkpoint feature mismatch')
    logits=[sum(w*x for w,x in zip(row,features)) for row in weights]
    if not all(math.isfinite(v) for v in logits):raise ValueError('Non-finite RL inference')
    exp=[math.exp(v-max(logits)) for v in logits]
    return [v/sum(exp) for v in exp]


def features(ctx):
    if ctx.ball is None:return None
    own=[r.pose for r in ctx.teammates.values() if r.pose is not None]
    opp=[r.pose for r in ctx.opponents.values() if r.pose is not None]
    if len(own)!=3 or len(opp)!=3:return None
    game=ctx.game
    ours=game.get_team_state(ctx.team_id) if game else None
    theirs=next((t for t in game.teams if t.team_number!=ctx.team_id),None) if game else None
    bx,by=ctx.ball.x,ctx.ball.y
    own_pen=sum(p.penalty!=Penalty.NONE for p in ours.players[:3]) if ours else 0
    opp_pen=sum(p.penalty!=Penalty.NONE for p in theirs.players[:3]) if theirs else 0
    values=[1.,bx/7,by/4.5,math.hypot(7-bx,by)/14,
            min(math.hypot(p.x-bx,p.y-by) for p in own)/14,
            min(math.hypot(p.x-bx,p.y-by) for p in opp)/14,
            sum(p.x for p in own)/21,sum(p.x for p in opp)/21,
            max(-1,min(1,((ours.score if ours else 0)-(theirs.score if theirs else 0))/5)),
            own_pen/3,opp_pen/3,(game.secs_remaining if game else 600)/600]
    return values if all(math.isfinite(v) for v in values) else None


class TeamPolicy:
    def __init__(self, model=None):
        self.model=copy.deepcopy(MODEL if model is None else model)
        self.batch=[]
        self.last_update_wall=None
        self.rng=random.Random(self.model.get('seed',314))
        self.pending=None
        self.last_scores=None
        self.last_penalties={}
        self.sequence=0
        self.finished=False
        self.started=False
        self.unassigned_reward=0.0
        self.observed_reward=0.0
        self.emitted_reward=0.0
        self.observed_penalties=0
        set_rl_parameters({})
        _log.info('RL checkpoint episode=%s enabled=%s training=%s updates=%s',
                  self.model.get('episode'),self.model.get('enabled'),self.model.get('training'),self.model.get('updates',0))

    def _emit(self, done, remaining, potential):
        if self.pending is None:return
        row=self.pending
        row['environment_reward']=row['reward']
        self.emitted_reward+=row['environment_reward']
        row['reward']+=.1*(potential-row.pop('potential'))
        row['duration_sim_seconds']=max(0,row.pop('remaining')-remaining)
        row.update(done=done,episode=self.model.get('episode'),sequence=self.sequence)
        self.sequence+=1
        if self.model.get('training') and self.model.get('update_interval_seconds'):
            self.batch.append(dict(row))
        _log.info('RL_TRANSITION %s',json.dumps(row,separators=(',',':')))
        self.pending=None

    def step(self,ctx):
        if not self.model.get('enabled') or ctx.game is None or self.finished:return
        game=ctx.game
        if game.state==GameState.INITIAL:return
        ours=game.get_team_state(ctx.team_id)
        theirs=next((t for t in game.teams if t.team_number!=ctx.team_id),None)
        if ours is None or theirs is None:return
        scores=(ours.score,theirs.score)
        # A freshly launched agent first receives the previous match's latched
        # FINISHED packet. Arm only after the new match has actually started.
        if not self.started:
            if game.state not in (GameState.READY,GameState.SET,GameState.PLAYING) or scores!=(0,0):
                return
            self.started=True
            self.last_update_wall=time.monotonic()
        if self.last_scores is None:self.last_scores=scores
        reward=10*((scores[0]-self.last_scores[0])-(scores[1]-self.last_scores[1]))
        self.last_scores=scores
        for i,p in enumerate(ours.players[:3]):
            if p.penalty!=Penalty.NONE and self.last_penalties.get(i,Penalty.NONE)==Penalty.NONE:
                reward-=.5
                self.observed_penalties+=1
            self.last_penalties[i]=p.penalty
        self.observed_reward+=reward
        if self.pending is not None:
            self.pending['reward']+=reward
        else:
            # Before the first actionable observation, keep an explicit ledger.
            # Never silently discard a referee reward during READY/SET or a gap.
            self.unassigned_reward+=reward
        remaining=game.secs_remaining
        potential=ctx.ball.x/7 if ctx.ball is not None else (self.pending['potential'] if self.pending else 0.)
        done=game.state==GameState.FINISHED
        playing=game.state==GameState.PLAYING and not game.stopped
        x=features(ctx) if playing else None
        actionable=playing and x is not None
        due=bool(self.model.get('training') and self.model.get('update_interval_seconds') and
                 time.monotonic()-self.last_update_wall>=self.model['update_interval_seconds'])
        # Keep the preceding action open through pauses and observation gaps.
        # Close it only at a terminal state or when we can select the next action.
        if self.pending and (done or (actionable and (due or self.pending['remaining']-remaining>=2))):
            self._emit(done,remaining,potential)
        if ((due and actionable) or done) and self.model.get('training') and self.model.get('update_interval_seconds') and len(self.batch)>=(1 if done else 4):
            from .online_learning import update_chunk
            next_features=None if done else features(ctx)
            if done or next_features is not None:
                self.model,stats=update_chunk(self.model,self.batch,next_features)
                now=time.monotonic()
                stats.update(interval_wall_seconds=now-self.last_update_wall,wall_time=time.time(),
                             update_index=self.model['updates'],episode=self.model['episode'],terminal=done)
                _log.info('RL_ONLINE_UPDATE %s',json.dumps(dict(stats=stats,model=self.model),separators=(',',':')))
                self.batch=[]
                self.last_update_wall=now
        if done:
            # End-marker covers a match that finishes during READY/SET after a goal.
            _log.info('RL_EPISODE_END %s',json.dumps(dict(episode=self.model.get('episode'),
                      transitions=self.sequence,score=scores,training=self.model.get('training',False),
                      observed_reward=self.observed_reward,emitted_reward=self.emitted_reward,
                      unassigned_reward=self.unassigned_reward,observed_penalties=self.observed_penalties)))
            self.finished=True
            return
        if not playing or self.pending is not None:return
        if x is None:return
        probs=probabilities(self.model['weights'],x)
        if self.model.get('training'):
            action=self.rng.choices(range(len(PROFILES)),weights=probs,k=1)[0]
        else:
            action=max(range(len(PROFILES)),key=lambda i:probs[i])
        set_rl_parameters({k:v for k,v in PROFILES[action].items() if k!='name'})
        self.pending=dict(features=x,action=action,probabilities=probs,reward=self.unassigned_reward,remaining=remaining,potential=potential)
        self.unassigned_reward=0.0
