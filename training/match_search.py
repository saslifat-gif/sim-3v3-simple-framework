"""Resumeable evolution-strategy training against the unmodified default AI.

Each reward comes from a complete standard 600-second simulator match. A target
candidate is replayed unchanged to report how well the result repeats.
No simulator rules, robot capabilities, referee or opponent code are modified.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'build/match-training'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT/'search.json'
IPC = OUT/'native'
CONTAINER = '7gofn-k1'
RUNNER = '/tmp/booster_agent_deploy/football3v3_runner'
INSTALL = '/opt/booster/booster_agent_data/data/agents/extract/com.example.lqinternalmatch'
AGENT = 'com.example.lqinternalmatch'
BOUNDS = dict(near_power=(3.0, 10.0), far_power=(4.0, 10.0), keeper_depth=(.35, 1.3),
              support_distance=(.7, 3.5), support_width=(0.0, 1.6), pressure_radius=(0.0, 2.0),
              center_bias=(0.0, .35), imitation_bonus=(0.0, .35))
INITIAL = dict(near_power=7.0, far_power=8.0, keeper_depth=.55, support_distance=1.6,
               support_width=.65, pressure_radius=1.0, center_bias=.04, imitation_bonus=.08)
state = json.loads(REPORT.read_text()) if REPORT.exists() else dict(
    method='bounded (1+1) evolution strategy with periodic elite re-evaluation',
    target='win a complete match against default AI by at least five goals; then replay unchanged to assess repeatability',
    opponent='com.booster.default3v3ai', duration_seconds=600, episodes=[], best=None, achieved=False)
rng = random.Random(128 + len(state['episodes']))
builder = None
port = None


def save():
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(REPORT)


def command(*args, timeout=180):
    run = subprocess.run(list(args), capture_output=True, text=True, encoding='utf8', errors='replace', timeout=timeout)
    if run.returncode:
        raise RuntimeError(f'{args[0:3]}: {run.stdout[-1200:]} {run.stderr[-1200:]}')
    return run.stdout


def docker(*args, **kwargs):
    return command('docker', 'exec', CONTAINER, *args, **kwargs)


def http(endpoint, post=False):
    global port
    if port is None:
        port = int(command('docker','port',CONTAINER,'38383/tcp').splitlines()[0].rsplit(':',1)[1])
    req = urllib.request.Request(f'http://127.0.0.1:{port}{endpoint}', data=b'{}' if post else None,
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def stop_requested():
    if (OUT/'stop-requested').exists():
        raise KeyboardInterrupt('Training stop requested')


def build(candidate, parameters):
    model = dict(enabled=True, candidate=candidate, parameters=parameters,
                 method='evolution strategy; full-match goal difference reward')
    (ROOT/'src/models/match_policy.py').write_text('"""Match-reward policy checkpoint."""\nMODEL = '+repr(model)+'\n')
    version = f'1.3.{len(state["episodes"])+1}'
    manifest = ROOT/'agent.toml'
    lines = manifest.read_text(encoding='utf8').splitlines()
    manifest.write_text('\n'.join(f'version = "{version}"' if line.startswith('version = ') else line for line in lines)+'\n', encoding='utf8')
    job = uuid.uuid4().hex
    (IPC/'request.json').write_text(json.dumps(dict(id=job)))
    deadline = time.monotonic()+600
    while not (IPC/f'{job}.json').exists():
        stop_requested()
        if builder.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError('Official package builder stopped or timed out')
        time.sleep(1)
    result = json.loads((IPC/f'{job}.json').read_text())
    if not result['success']:
        raise RuntimeError(result['error'])
    source = ROOT/f'build/{AGENT}-{version}.agent'
    if not source.is_file():
        raise RuntimeError(f'Builder did not produce {source}')
    archive = OUT/f'{candidate}.agent'
    shutil.copy2(source, archive)
    return archive, version


def launch(archive):
    http('/match/end', True)
    docker('python3', f'{RUNNER}/launcher.py', '--stop')
    command('docker','cp',str(archive),f'{CONTAINER}:/tmp/match-search.agent')
    docker('python3',f'{RUNNER}/install_agent_package.py','/tmp/match-search.agent',INSTALL)
    command('docker','cp',str(ROOT/'build/benchmark/reset_field.py'),f'{CONTAINER}:/tmp/match-search-reset.py')
    output = docker('/usr/local/booster_robot/booster_robocup_sim/.venv/bin/python','/tmp/match-search-reset.py')
    reset = json.loads(next(s[len('RESET_RESULT '):] for s in output.splitlines() if s.startswith('RESET_RESULT ')))
    docker('python3',f'{RUNNER}/launcher.py','--publish-logs','--teams',AGENT,'com.booster.default3v3ai')
    status = http('/status')
    if not status['gameControl']['ready'] or status['game']['regularDurationSeconds'] != 600:
        raise RuntimeError('Unexpected match setup')
    return reset


def play(candidate, parameters, archive, version, validation=False):
    reset = launch(archive)
    episode = dict(number=len(state['episodes'])+1, candidate=candidate, parameters=parameters,
                   package=str(archive.relative_to(ROOT)), version=version,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), validation=validation,
                   reset=reset, started=time.time(), samples=[])
    state['episodes'].append(episode)
    save()
    http('/match/start', True)
    deadline = time.monotonic()+1500
    while time.monotonic()<deadline:
        stop_requested()
        status = http('/status')
        game = status['game']
        sample = dict(at=time.time(), state=game['state'], remaining=game['timeRemaining'],
                      score=[game['teams']['home']['score'],game['teams']['away']['score']])
        episode['samples'].append(sample)
        save()
        print(f'EPISODE {episode["number"]} {candidate}'+(' VALIDATION' if validation else '')+
              f' | {sample["score"][0]}-{sample["score"][1]} | {sample["state"]} | {sample["remaining"]:.0f}s', flush=True)
        if status['gameControl'].get('lastError'):
            raise RuntimeError(status['gameControl']['lastError'])
        if game['state']=='finished':
            if game.get('finishReason') != 'time_expired' or game['timeRemaining']>1:
                raise RuntimeError('Abnormal match ending')
            break
        time.sleep(20)
    else:
        raise RuntimeError('Match exceeded 25-minute wall-time limit')
    episode.update(finished=time.time(), score=sample['score'], finish_reason=game['finishReason'])
    event_file = OUT/f'events-{episode["number"]:04d}.jsonl'
    command('docker','cp',f'{CONTAINER}:/usr/local/booster_robot/booster_robocup_sim/logs/game-control/events.jsonl',str(event_file))
    events = [json.loads(line) for line in event_file.read_text().splitlines()]
    penalties = [e for e in events if e.get('type')=='penalty' and e.get('actor',{}).get('side')=='home'
                 and episode['started']-.5 <= e['wallTime'] <= episode['finished']]
    episode['penalties'] = dict(collections.Counter(e['outcome'].get('penalty') for e in penalties))
    episode['margin'] = episode['score'][0]-episode['score'][1]
    episode['reward'] = episode['margin'] + .03*episode['score'][0] - .1*len(penalties)
    # Read actual isolated runtime logs, not the install directory's namespace.
    inspected = command(sys.executable, str(ROOT/'build/benchmark/run_comparison.py'), '--inspect')
    episode['runtime'] = json.loads(inspected)['home']
    if episode['runtime'].get('unavailable') or episode['runtime'].get('control_loop_exceptions') or episode['runtime'].get('tracebacks'):
        raise RuntimeError('Candidate runtime unavailable or contains Python errors')
    save()
    print('FINISHED '+json.dumps({k:episode[k] for k in ['number','candidate','score','margin','reward','penalties','validation']}),flush=True)
    return episode


def choose_parameters():
    completed = [e for e in state['episodes'] if 'reward' in e and not e.get('validation')]
    if not completed:
        return INITIAL.copy()
    best = max(completed,key=lambda e:e['reward'])
    state['best'] = {k:best[k] for k in ['candidate','package','parameters','version','score','margin','reward','sha256']}
    if len(completed)%4 == 0:
        return best['parameters'].copy()
    sigma = .28 if len(completed)<8 else .17
    return {name:round(max(lo,min(hi,best['parameters'][name]+rng.gauss(0,sigma*(hi-lo)))),5)
            for name,(lo,hi) in BOUNDS.items()}


def main():
    global builder
    if state.get('achieved'):
        print('Target already achieved; refusing unnecessary retraining.',flush=True)
        return
    IPC.mkdir(parents=True,exist_ok=True)
    ready = IPC/'ready.json'
    ready.unlink(missing_ok=True)
    (IPC/'stop').unlink(missing_ok=True)
    (IPC/'request.json').unlink(missing_ok=True)
    with (OUT/'builder.log').open('a') as log:
        builder = subprocess.Popen(['node',str(ROOT/'training/native_builder.cjs')],cwd=ROOT,stdout=log,stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+300
        while not ready.exists():
            stop_requested()
            if builder.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Could not initialize simulator/builder; inspect builder.log')
            time.sleep(1)
        while not state.get('achieved'):
            stop_requested()
            parameters=choose_parameters()
            candidate=f'candidate-{len(state["episodes"])+1:04d}'
            print(f'BUILDING {candidate} '+json.dumps(parameters),flush=True)
            archive,version=build(candidate,parameters)
            result=play(candidate,parameters,archive,version)
            if state['best'] is None or result['reward']>state['best']['reward']:
                state['best']={k:result[k] for k in ['candidate','package','parameters','version','score','margin','reward','sha256']}
            save()
            if result['margin']>=5:
                confirmation=play(candidate,parameters,archive,version,validation=True)
                state['achieved']=True
                state['target']='win a complete match against default AI by at least five goals; then replay unchanged to assess repeatability'
                state['winning_episodes']=[result['number']]
                state['confirmation_episode']=confirmation['number']
                state['confirmation_reached_five_goal_margin']=confirmation['margin']>=5
                state['qualified']={k:result[k] for k in ['candidate','package','parameters','version','score','margin','reward','sha256']}
                save()
                print('TARGET ACHIEVED '+json.dumps(state['qualified']),flush=True)


try:
    main()
except BaseException as error:
    state['last_error']=str(error)
    save()
    print('TRAINING ERROR '+repr(error),flush=True)
    raise
finally:
    if builder is not None:
        (IPC/'stop').touch()
    save()
