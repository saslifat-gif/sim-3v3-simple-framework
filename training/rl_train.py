"""Four accelerated on-policy training matches, followed by a frozen 1x test."""
from __future__ import annotations
import collections
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from policy_gradient import update
from tensorboard_metrics import Metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/rl-training'
CONTAINER=os.environ.get('BOOSTER_ROBOT','7gofn')+'-k1'
RUNNER='/tmp/booster_agent_deploy/football3v3_runner'
INSTALL='/opt/booster/booster_agent_data/data/agents/extract/com.example.lqinternalmatch'
AGENT='com.example.lqinternalmatch'
parser=argparse.ArgumentParser()
parser.add_argument('--additional-episodes',type=int,default=0)
parser.add_argument('--update-seconds',type=float,default=20)
parser.add_argument('--run-directory',type=Path,default=OUT)
parser.add_argument('--initial-model',type=Path)
parser.add_argument('--training-speed',type=float,default=1)
parser.add_argument('--version-prefix',default='1.7')
parser.add_argument('--tensorboard-directory',type=Path)
args=parser.parse_args()
if args.additional_episodes<0:parser.error('additional episodes must be nonnegative')
if args.update_seconds<5:parser.error('update interval must be at least five seconds')
if not 0<args.training_speed<=10:parser.error('speed must be in (0,10]')
OUT=args.run_directory.resolve()
OUT.mkdir(parents=True,exist_ok=True)
IPC=OUT/'native'
REPORT=OUT/'report.json'
state=json.loads(REPORT.read_text()) if REPORT.exists() else dict(method='vanilla policy gradient with a linear value baseline',
    requested_speed=args.training_speed,formation='three outfield players, no dedicated goalkeeper',episodes=[],complete=False)
state['requested_speed']=args.training_speed
builder=None
restore_archive=None
port=None
metrics=Metrics(args.tensorboard_directory or OUT/'tensorboard',event_directory=OUT)
for historical in state['episodes']:metrics.episode(historical)
if args.additional_episodes:
    state['target_updates']=sum(bool(e.get('update')) for e in state['episodes'])+args.additional_episodes
    state['complete']=False
    state['continuation_started']=time.time()
TRAINING_EPISODES=state.get('target_updates',4)


def save():
    temp=REPORT.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2))
    temp.replace(REPORT)


def command(*args,timeout=180):
    p=subprocess.run(list(args),capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if p.returncode:raise RuntimeError(f'{args[:3]}: {p.stdout[-1200:]} {p.stderr[-600:]}')
    return p.stdout


def docker(*args,**kwargs):return command('docker','exec',CONTAINER,*args,**kwargs)


def http(path,post=False):
    global port
    if port is None:port=int(command('docker','port',CONTAINER,'38383/tcp').splitlines()[0].rsplit(':',1)[1])
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=b'{}' if post else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)


def stop_check():
    if (OUT/'stop-requested').exists():raise KeyboardInterrupt('RL stop requested')


def export(model,version):
    (ROOT/'src/models/rl_policy.py').write_text('"""Portable team-level RL checkpoint."""\nMODEL = '+repr(model)+'\n')
    path=ROOT/'agent.toml'
    path.write_text('\n'.join(f'version = "{version}"' if s.startswith('version = ') else s
                              for s in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')


def build(model,version,deploy=False):
    export(model,version)
    job=uuid.uuid4().hex
    (IPC/'request.json').write_text(json.dumps(dict(id=job,deploy=deploy)))
    deadline=time.monotonic()+600
    while not (IPC/f'{job}.json').exists():
        stop_check()
        if builder.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Package build failed or timed out')
        time.sleep(.5)
    result=json.loads((IPC/f'{job}.json').read_text())
    if not result['success']:raise RuntimeError(result['error'])
    source=ROOT/f'build/{AGENT}-{version}.agent'
    dest=OUT/f'{model["episode"]}.agent'
    shutil.copy2(source,dest)
    (OUT/f'{model["episode"]}-model.json').write_text(json.dumps(model,indent=2))
    return dest


def set_speed(speed):
    output=docker('/usr/local/booster_robot/booster_robocup_sim/.venv/bin/python',
                  '/tmp/rl-simulator-speed.py','--speed',str(speed),'--seconds','2')
    record=json.loads(next(line[len('SPEED_RESULT '):] for line in output.splitlines() if line.startswith('SPEED_RESULT ')))
    if record['reported_target']!=speed:raise RuntimeError(f'Simulator did not accept speed target {speed}: {record}')
    return record


def install(archive):
    http('/match/end',True)
    docker('python3',RUNNER+'/launcher.py','--stop')
    command('docker','cp',str(archive),CONTAINER+':/tmp/rl-candidate.agent')
    docker('python3',RUNNER+'/install_agent_package.py','/tmp/rl-candidate.agent',INSTALL)


def launch(archive,speed):
    install(archive)
    output=docker('/usr/local/booster_robot/booster_robocup_sim/.venv/bin/python','/tmp/rl-reset.py')
    reset=json.loads(next(s[len('RESET_RESULT '):] for s in output.splitlines() if s.startswith('RESET_RESULT ')))
    docker('python3',RUNNER+'/launcher.py','--publish-logs','--teams',AGENT,'com.booster.default3v3ai')
    measured=set_speed(speed)
    # Runner readiness does not guarantee the ROS observations have arrived.
    # Refuse to start a match with a blind bot.
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        stop_check()
        ticks=[line for line in read_rollout().splitlines() if 'tick #' in line]
        if ticks and all(token in ticks[-1] for token in ['teammates_seen=3/3','opponents_seen=3/3']) and 'game=None' not in ticks[-1] and 'ball=None' not in ticks[-1]:
            break
        time.sleep(2)
    else:
        raise RuntimeError('ROS observations unavailable before match; refusing to start blind training')
    setup=http('/status')
    if not setup['gameControl']['ready'] or setup['game']['regularDurationSeconds']!=600:
        raise RuntimeError('Unexpected match setup')
    return reset,measured


def read_rollout():
    script="""import pathlib,glob,json
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='ignore')
  if not any('/com_example_lqinternalmatch/pyagent_'+arch in cmd for arch in ['x86_64','aarch64']):continue
  if b'SOCCER_TEAM_ID=1' not in (p/'environ').read_bytes().split(b'\\0'):continue
  paths=glob.glob(f'{p}/root/var/log/booster_agent/com.example.lqinternalmatch/agent_*.log')
  if paths:
   print(pathlib.Path(sorted(paths)[-1]).read_text(errors='replace'));break
 except (OSError,PermissionError):pass
"""
    return docker('python3','-c',script)


def collect_online(model,episode,text):
    seen={r['update_index'] for r in episode.get('online_updates',[])}
    for line in text.splitlines():
        if 'RL_ONLINE_UPDATE ' not in line:continue
        record=json.loads(line.split('RL_ONLINE_UPDATE ',1)[1])
        stats=record['stats']
        if stats['episode']!=episode['name']:raise RuntimeError('Mixed online checkpoint')
        if stats['update_index'] in seen:continue
        if stats['update_index']!=model.get('updates',0)+1:raise RuntimeError('Missing online update')
        if not record['model'].get('training'):raise RuntimeError('Unexpected evaluation update')
        model.clear();model.update(record['model'])
        episode.setdefault('online_updates',[]).append(stats)
        seen.add(stats['update_index'])
        snapshot=OUT/f'online-checkpoint-{stats["update_index"]:05d}.json'
        snapshot.write_text(json.dumps(model,indent=2))
        (OUT/'latest-model.json').write_text(json.dumps(model,indent=2))
        metrics.scalars({'online/'+k:v for k,v in stats.items() if isinstance(v,(int,float))},
                        stats['update_index'],stats['wall_time'])
        print('ONLINE_UPDATE '+json.dumps(stats),flush=True)


def play(model,archive,version,speed):
    reset,measured=launch(archive,speed)
    episode=dict(name=model['episode'],training=model['training'],version=version,
        package=str(archive.relative_to(ROOT)),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        requested_speed=speed,speed_probe=measured,reset=reset,started=time.time(),samples=[])
    state['episodes'].append(episode)
    save()
    http('/match/start',True)
    deadline=time.monotonic()+1500
    while time.monotonic()<deadline:
        stop_check()
        status=http('/status')
        g=status['game']
        sample=dict(at=time.time(),state=g['state'],remaining=g['timeRemaining'],
                    score=[g['teams']['home']['score'],g['teams']['away']['score']])
        episode['samples'].append(sample)
        if model.get('training') and model.get('update_interval_seconds'):
            runtime_text=read_rollout()
            ticks=[line for line in runtime_text.splitlines() if 'tick #' in line]
            if len(ticks)>=3 and all('game=None' in line or 'ball=None' in line or 'teammates_seen=0/3' in line for line in ticks[-3:]):
                raise RuntimeError('Observations lost during match; stopping invalid training')
            collect_online(model,episode,runtime_text)
        metrics.scalars({'live/goals_for':sample['score'][0],'live/goals_against':sample['score'][1],
                         'live/game_seconds_remaining':sample['remaining'],'live/requested_speed':speed},
                        sum(len(e.get('samples',[])) for e in state['episodes']))
        save()
        if len(episode['samples'])%3==1 or g['state']=='finished':
            print(f'{model["episode"]} speed_target={speed} | {sample["score"]} | {g["state"]} | {g["timeRemaining"]:.0f}s',flush=True)
        if status['gameControl'].get('lastError'):raise RuntimeError(status['gameControl']['lastError'])
        if g['state']=='finished':
            if g.get('finishReason')!='time_expired' or g['timeRemaining']>1:raise RuntimeError('Abnormal match ending')
            break
        time.sleep(5)
    else:raise RuntimeError('Match wall-time limit exceeded')
    episode.update(score=sample['score'],finished=time.time(),finish_reason=g['finishReason'],
                   match_seconds_per_wall_second=600/(time.time()-episode['started']))
    time.sleep(2)
    text=read_rollout()
    if model.get('training') and model.get('update_interval_seconds'):
        collect_online(model,episode,text)
    (OUT/f'{model["episode"]}-runtime.log').write_text(text,encoding='utf8')
    if 'control loop tick failed' in text or 'Traceback' in text:raise RuntimeError('RL runtime contains Python errors')
    rows=[json.loads(line.split('RL_TRANSITION ',1)[1]) for line in text.splitlines() if 'RL_TRANSITION ' in line]
    ends=[json.loads(line.split('RL_EPISODE_END ',1)[1]) for line in text.splitlines() if 'RL_EPISODE_END ' in line]
    if len(rows)<20:raise RuntimeError('Too few RL transitions; refusing an empty or partial rollout')
    if not ends or ends[-1]['episode']!=model['episode'] or ends[-1]['transitions']!=len(rows):
        raise RuntimeError('Incomplete RL trajectory')
    if list(ends[-1]['score'])!=episode['score']:raise RuntimeError('Runtime/controller scores differ')
    if any(r['episode']!=model['episode'] or r['sequence']!=i for i,r in enumerate(rows)):
        raise RuntimeError('Mixed or missing trajectory sequence')
    (OUT/f'{model["episode"]}-rollout.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    episode['transitions']=len(rows)
    episode['action_counts']=dict(collections.Counter(r['action'] for r in rows))
    episode['python_errors']=0
    command('docker','cp',CONTAINER+':/usr/local/booster_robot/booster_robocup_sim/logs/game-control/events.jsonl',
            str(OUT/f'{model["episode"]}-events.jsonl'))
    ledger=ends[-1]
    emitted=sum(row['environment_reward'] for row in rows)
    if abs(emitted-ledger['emitted_reward'])>1e-6 or abs(ledger['observed_reward']-emitted-ledger['unassigned_reward'])>1e-6:
        raise RuntimeError('Observed rewards were lost from the rollout')
    events=[json.loads(s) for s in (OUT/f'{model["episode"]}-events.jsonl').read_text().splitlines()]
    events=[e for e in events if episode['started']-.5<=e['wallTime']<=episode['finished']]
    penalties=[e for e in events if e.get('type')=='penalty' and e.get('actor',{}).get('side')=='home']
    expected_reward=10*(episode['score'][0]-episode['score'][1])-.5*len(penalties)
    episode['reward_audit']=dict(expected_environment_reward=expected_reward,
        observed_environment_reward=ledger['observed_reward'],emitted_environment_reward=emitted,
        unassigned_reward=ledger['unassigned_reward'],referee_penalties=len(penalties),observed_penalties=ledger['observed_penalties'])
    if abs(ledger['observed_reward']-expected_reward)>1e-6:
        raise RuntimeError('Referee events and observed rewards differ; refusing to accept this episode')
    save()
    return episode,rows


def main():
    global builder,restore_archive
    if state.get('complete'):
        print('RL pilot already completed. Results: '+str(REPORT),flush=True)
        return
    IPC.mkdir(parents=True,exist_ok=True)
    for name in ['ready.json','stop','request.json']:(IPC/name).unlink(missing_ok=True)
    state.pop('error',None)
    state.pop('restore_error',None)
    with (OUT/'builder.log').open('a') as log:
        builder=subprocess.Popen(['node',str(ROOT/'training/native_builder.cjs'),'--ipc',str(IPC)],
            cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        deadline=time.monotonic()+300
        while not (IPC/'ready.json').exists():
            stop_check()
            if builder.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Builder initialization failed')
            time.sleep(.5)
        for local,remote in [(ROOT/'training/simulator_speed.py','/tmp/rl-simulator-speed.py'),
                             (ROOT/'build/benchmark/reset_field.py','/tmp/rl-reset.py')]:
            command('docker','cp',str(local),CONTAINER+':'+remote)
        checkpoint=OUT/'latest-model.json'
        if checkpoint.exists():
            model=json.loads(checkpoint.read_text())
        elif args.initial_model:
            model=json.loads(args.initial_model.read_text())
            model['value_weights']=[0.]*12
            state['initial_checkpoint']=str(args.initial_model)
        else:
            weights=[[0.]*12 for _ in range(5)]
            weights[0][0]=.25
            model=dict(format_version=1,enabled=True,training=True,weights=weights,
                       value_weights=[0.]*12,seed=314,updates=0)
        # Build a frozen recovery package on this machine; GitHub snapshots do
        # not contain the old Windows experiment's .agent archives.
        baseline=dict(model,episode='baseline',training=False)
        print('BUILDING baseline',flush=True)
        restore_archive=build(baseline,f'{args.version_prefix}.0',deploy=True)
        completed={e['name'] for e in state['episodes'] if e.get('update')}
        for index in range(TRAINING_EPISODES):
            name=f'train-{index+1:02d}'
            if name in completed:continue
            model.update(episode=name,training=True,seed=314+index,update_interval_seconds=args.update_seconds)
            version=f'{args.version_prefix}.{index+1}'
            print('BUILDING '+name,flush=True)
            archive=build(model,version)
            episode,rows=play(model,archive,version,args.training_speed)
            if model.get('update_interval_seconds'):
                updates=episode.get('online_updates',[])
                if not updates:raise RuntimeError('No online learning updates occurred')
                stats=dict(transitions=sum(u['transitions'] for u in updates),
                           reward_sum=sum(r['reward'] for r in rows),online_update_count=len(updates),
                           weight_change_l2=sum(u['weight_change_l2'] for u in updates))
            else:
                model,stats=update(model,rows)
            episode['update']=stats
            episode['update_index']=model['updates']
            checkpoint.write_text(json.dumps(model,indent=2))
            save()
            metrics.episode(episode)
            print('RL_UPDATE '+json.dumps(stats),flush=True)
        evaluation_name='evaluation-1x' if TRAINING_EPISODES==4 else f'evaluation-{TRAINING_EPISODES:02d}-1x'
        evaluation_version=f'{args.version_prefix}.{TRAINING_EPISODES+1}'
        model.update(episode=evaluation_name,training=False)
        archive=build(model,evaluation_version)
        episode,_=play(model,archive,evaluation_version,1)
        episode['update_index']=model['updates']
        metrics.episode(episode)
        state.update(complete=True,evaluation=episode['score'],trained_updates=model['updates'],
                     active_version=evaluation_version,training_exploration_disabled=True)
        save()
        print('RL PILOT COMPLETE '+json.dumps(dict(score=episode['score'],updates=model['updates'])),flush=True)


try:
    main()
except BaseException as error:
    state['error']=repr(error)
    save()
    print('RL ERROR '+repr(error),flush=True)
    raise
finally:
    if builder is not None:
        try:
            state['restored_speed']=set_speed(1)
            if not state.get('complete') and restore_archive is not None:
                install(restore_archive)
                docker('python3',RUNNER+'/launcher.py','--publish-logs','--teams',AGENT,'com.booster.default3v3ai')
                state['active_version']=f'{args.version_prefix}.0'
        except Exception as error:state['restore_error']=repr(error)
        (IPC/'stop').touch()
    save()
    metrics.close()
