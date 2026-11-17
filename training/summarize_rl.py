"""Audit complete match evidence and export a concise RL pilot result."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/rl-training'
report=json.loads((OUT/'report.json').read_text())
assert report.get('complete') and not report.get('error') and not report.get('restore_error')
training=[e for e in report['episodes'] if e.get('update')]
evaluation=next(e for e in reversed(report['episodes']) if not e['training'] and e.get('transitions'))
assert len(training)==4 and report['trained_updates']==4
penalties={}
for episode in training+[evaluation]:
    assert episode['finish_reason']=='time_expired' and episode['python_errors']==0
    assert hashlib.sha256((ROOT/episode['package']).read_bytes()).hexdigest()==episode['sha256']
    events=[json.loads(s) for s in (OUT/f'{episode["name"]}-events.jsonl').read_text().splitlines()]
    events=[e for e in events if episode['started']-.5 <= e['wallTime'] <= episode['finished']]
    goals=[e for e in events if e['type']=='goal']
    assert [sum(e['actor']['side']==side for e in goals) for side in ['home','away']]==episode['score']
    assert any(e['type']=='match_finished' for e in events)
    penalties[episode['name']]=dict(collections.Counter(e.get('outcome',{}).get('penalty') for e in events
        if e['type']=='penalty' and e.get('actor',{}).get('side')=='home'))
    assert episode['speed_probe']['reported_target']==episode['requested_speed']
model=json.loads((OUT/'evaluation-1x-model.json').read_text())
assert model['training'] is False and model['updates']==4
rows=[json.loads(s) for s in (OUT/'evaluation-1x-rollout.jsonl').read_text().splitlines()]
for row in rows:
    logits=[sum(w*x for w,x in zip(weights,row['features'])) for weights in model['weights']]
    assert row['action']==max(range(5),key=lambda i:logits[i])
initial=[[0.]*12 for _ in range(5)];initial[0][0]=.25
weight_change=math.sqrt(sum((model['weights'][i][j]-initial[i][j])**2 for i in range(5) for j in range(12)))
summary=dict(method='Vanilla policy gradient; centralized linear-softmax team policy with a learned value baseline',
    formation='Three outfield players; no dedicated goalkeeper',requested_training_speed=10,
    measured_game_clock_speedup_excluding_builds=sum(600 for _ in training)/sum(e['finished']-e['started'] for e in training),
    valid_training_matches=len(training),training_scores=[e['score'] for e in training],
    training_transitions=sum(e['transitions'] for e in training),policy_updates=4,
    weight_change_l2_from_initial=weight_change,update_statistics=[e['update'] for e in training],
    evaluation=dict(version=evaluation['version'],score=evaluation['score'],speed_target=1,
        measured_physics_speed=evaluation['speed_probe']['measured_sim_seconds_per_wall_second'],
        exploration_disabled=True,used_for_training=False,action_counts=evaluation['action_counts'],
        package=evaluation['package'],sha256=evaluation['sha256']),
    penalties=penalties,python_exceptions=0,referee_events_verified=True,
    final_speed_target=report['restored_speed']['reported_target'],
    limitations=['One normal-speed evaluation; no reliable estimate of superiority over the baseline.',
                 'Accelerated simulation changes wall-clock controller timing; normal-speed evaluation is reported separately.',
                 'One empty rollout was rejected after a stale match-end packet; it did not contribute a gradient update.'])
path=ROOT/'training/artifacts/rl-pilot-summary.json'
path.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
