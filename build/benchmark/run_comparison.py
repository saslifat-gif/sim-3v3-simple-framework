"""Two original signed packages, side-swapped, using the simulator's team launcher."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTAINER = '7gofn-k1'
RUNNER = '/tmp/booster_agent_deploy/football3v3_runner'
INSTALL_ROOT = '/opt/booster/booster_agent_data/data/agents/extract'
AGENT_ID = 'com.example.lqinternalmatch'
AWAY_KEY = 'football_imports/football3v3/away'
PACKAGES = {'ml': ROOT/'build/com.example.lqinternalmatch-1.2.0.agent',
            'rule': ROOT/'build/com.example.lqinternalmatch-1.1.2.agent'}
VERSIONS = {'ml': '1.2.0', 'rule': '1.1.2'}
result = {'design': 'two 600-second head-to-head games; swap home/away; reset physics before each game',
          'versions': VERSIONS, 'packages': {}, 'games': [],
          'limitations': ['Only two matches; not a statistically reliable win-rate estimate.',
                          'No deterministic seed interface was established.'], 'restored': False}
if '--resume' in sys.argv and (HERE/'comparison.json').exists():
    previous = json.loads((HERE/'comparison.json').read_text())
    if previous.get('versions') != VERSIONS:
        raise RuntimeError('Saved benchmark versions differ; refusing to mix results')
    result = previous
    result['games'] = [g for g in previous['games']
                       if g.get('finish_reason') == 'time_expired' and 'final_score' in g
                       and 'runtime' in g and 'penalties' in g]
    for key in ['error', 'restore_error', 'totals']:
        result.pop(key, None)
    result['restored'] = False
result['complete'] = False


def command(*args, timeout=150):
    p = subprocess.run(list(args), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{args}: {p.stdout[-1500:]} {p.stderr[-1500:]}')
    return p.stdout


def docker(*args, timeout=150):
    return command('docker', 'exec', CONTAINER, *args, timeout=timeout)


def save():
    (HERE/'comparison.json').write_text(json.dumps(result, indent=2))


port = int(command('docker', 'port', CONTAINER, '38383/tcp').splitlines()[0].rsplit(':', 1)[1])


def http(path, post=False):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                                 data=b'{}' if post else None,
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def launch(home, away=None):
    args = ['python3', f'{RUNNER}/launcher.py', '--publish-logs', '--teams', AGENT_ID,
            AGENT_ID if away else 'com.booster.default3v3ai']
    if away:
        args += ['--away-agent-dir-key', AWAY_KEY]
    output = docker(*args)
    if 'error:' in output.lower():
        raise RuntimeError(output)
    print(output.strip(), flush=True)


def install(kind, away=False):
    destination = f'{INSTALL_ROOT}/{AWAY_KEY if away else AGENT_ID}'
    docker('python3', f'{RUNNER}/install_agent_package.py', f'/tmp/policy-benchmark/{kind}.agent', destination)
    metadata = json.loads(docker('cat', f'{destination}/agent.json'))
    if metadata['version'] != VERSIONS[kind]:
        raise RuntimeError(f'Wrong package loaded: {metadata}')
    return {'id': metadata['id'], 'version': metadata['version'], 'source_directory': destination}


def runtime_errors(team):
    # Logs are read from each team's actual process mount namespace.
    script = """import glob,json,pathlib
team=TEAM
pids=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  data=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='ignore')
  if '/com_example_lqinternalmatch/pyagent_x86_64' in data:
   env=(p/'environ').read_bytes().split(b'\\0')
   if f'SOCCER_TEAM_ID={team}'.encode() in env:pids.append(p.name)
 except (OSError,PermissionError):pass
for pid in pids:
 paths=glob.glob(f'/proc/{pid}/root/var/log/booster_agent/com.example.lqinternalmatch/agent_*.log')
 if paths:
  text=pathlib.Path(sorted(paths)[-1]).read_text(errors='replace')
  print(json.dumps(dict(control_loop_exceptions=text.count('control loop tick failed'),tracebacks=text.count('Traceback'),movement_failures=sum('set_velocity' in s and 'failed' in s for s in text.splitlines()),last_model_checkpoint=next((s for s in reversed(text.splitlines()) if 'Imitation policy decisions=' in s),None))))
  break
else: print(json.dumps(dict(unavailable=True)))
""".replace('team=TEAM', f'team={team}')
    return json.loads(docker('python3', '-c', script).splitlines()[-1])


if '--inspect' in sys.argv:
    print(json.dumps({'home':runtime_errors(1), 'away':runtime_errors(2)}, indent=2))
    raise SystemExit(0)

try:
    for kind, package in PACKAGES.items():
        result['packages'][kind] = {'file': package.name, 'sha256': hashlib.sha256(package.read_bytes()).hexdigest()}
    save()
    docker('mkdir', '-p', '/tmp/policy-benchmark')
    for kind, package in PACKAGES.items():
        command('docker', 'cp', str(package), f'{CONTAINER}:/tmp/policy-benchmark/{kind}.agent')
    command('docker', 'cp', str(HERE/'reset_field.py'), f'{CONTAINER}:/tmp/policy-benchmark/reset_field.py')
    for number, (home, away) in enumerate([('ml','rule'), ('rule','ml')], 1):
        if any(g['number'] == number for g in result['games']):
            print(f'RESUME: preserving completed game {number}', flush=True)
            continue
        print(f'PREPARING game {number}: home={home}, away={away}', flush=True)
        http('/match/end', True)
        docker('python3', f'{RUNNER}/launcher.py', '--stop')
        home_metadata, away_metadata = install(home), install(away, True)
        reset_output = docker('/usr/local/booster_robot/booster_robocup_sim/.venv/bin/python', '/tmp/policy-benchmark/reset_field.py')
        reset_record = json.loads(next(s.removeprefix('RESET_RESULT ') for s in reset_output.splitlines() if s.startswith('RESET_RESULT ')))
        launch(home, away)
        status = http('/status')
        if not status['gameControl']['ready']:
            raise RuntimeError('Controller is not ready after both team launches')
        if abs(status['game']['regularDurationSeconds']-600) > .01:
            raise RuntimeError('Unexpected match duration')
        game = dict(number=number, home=home, away=away, home_metadata=home_metadata,
                    away_metadata=away_metadata, verified_reset=reset_record, samples=[])
        result['games'].append(game)
        game['start_wall_time'] = time.time()
        http('/match/start', True)
        deadline = time.monotonic()+1200
        while time.monotonic() < deadline:
            if (HERE/'stop-requested').exists():
                raise KeyboardInterrupt('Benchmark stop requested')
            status = http('/status')
            g = status['game']
            sample = dict(wall_time=time.time(), state=g['state'], remaining=g['timeRemaining'],
                          home_score=g['teams']['home']['score'], away_score=g['teams']['away']['score'],
                          controller_error=status['gameControl'].get('lastError'))
            game['samples'].append(sample)
            save()
            print(f'GAME {number} {home} {sample["home_score"]}-{sample["away_score"]} {away} | {sample["state"]} remaining={sample["remaining"]:.0f}s', flush=True)
            if sample['controller_error']:
                raise RuntimeError(sample['controller_error'])
            if g['state'] == 'finished':
                break
            time.sleep(30)
        else:
            raise RuntimeError('Match did not finish within 20 minutes')
        game['finish_reason'] = g.get('finishReason')
        if game['finish_reason'] != 'time_expired' or g['timeRemaining'] > 1:
            raise RuntimeError(f'Match ended abnormally: {game["finish_reason"]}')
        game['final_score'] = [sample['home_score'], sample['away_score']]
        game['runtime'] = {'home':runtime_errors(1), 'away':runtime_errors(2)}
        event_file = HERE/f'game-{number}-events.jsonl'
        command('docker','cp',f'{CONTAINER}:/usr/local/booster_robot/booster_robocup_sim/logs/game-control/events.jsonl',str(event_file))
        events = [json.loads(s) for s in event_file.read_text().splitlines()]
        events = [e for e in events if game['start_wall_time']-.5 <= e['wallTime'] <= sample['wall_time']]
        game['penalties'] = {side:dict(collections.Counter(e.get('outcome',{}).get('penalty') for e in events if e.get('type')=='penalty' and e.get('actor',{}).get('side')==side)) for side in ['home','away']}
        save()
    totals = {kind:dict(wins=0,draws=0,losses=0,goals_for=0,goals_against=0,penalties=0) for kind in PACKAGES}
    for game in result['games']:
        for index, side in enumerate(['home','away']):
            kind = game[side]
            gf, ga = game['final_score'][index], game['final_score'][1-index]
            totals[kind]['goals_for'] += gf
            totals[kind]['goals_against'] += ga
            totals[kind]['wins' if gf>ga else 'losses' if gf<ga else 'draws'] += 1
            totals[kind]['penalties'] += sum(game['penalties'][side].values())
    result['totals'] = totals
    result['complete'] = True
except BaseException as error:
    result['error'] = str(error)
    print(f'BENCHMARK ERROR: {error}', flush=True)
    raise
finally:
    try:
        print('RESTORING production ML agent and default opponent', flush=True)
        http('/match/end', True)
        docker('python3', f'{RUNNER}/launcher.py', '--stop')
        install('ml')
        launch('ml')
        result['restored'] = True
    except Exception as error:
        result['restore_error'] = str(error)
    save()
    print('RESULT '+json.dumps({k:v for k,v in result.items() if k not in ['games','packages']}), flush=True)
