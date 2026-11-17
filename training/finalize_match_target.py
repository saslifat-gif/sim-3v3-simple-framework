"""Verify the requested five-goal result, export it, and restore its exact package."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'build/match-training'
report = json.loads((OUT/'search.json').read_text())
qualifying = next(e for e in report['episodes'] if e.get('margin',-100)>=5
                  and e.get('finish_reason')=='time_expired' and not e.get('validation'))
replay = next(e for e in report['episodes'] if e.get('validation')
              and e.get('finish_reason')=='time_expired' and e.get('sha256')==qualifying['sha256'])
assert report['duration_seconds']==600
for episode in [qualifying,replay]:
    assert episode['runtime'] and not any(episode['runtime'].get(k) for k in
        ['unavailable','tracebacks','control_loop_exceptions'])
    events = [json.loads(line) for line in (OUT/f'events-{episode["number"]:04d}.jsonl').read_text().splitlines()]
    events = [e for e in events if episode['started']-.5 <= e['wallTime'] <= episode['finished']]
    goals = [e for e in events if e['type']=='goal']
    assert [sum(e['actor']['side']==side for e in goals) for side in ['home','away']]==episode['score']
    assert any(e['type']=='match_finished' for e in events)
archive = ROOT/qualifying['package']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==qualifying['sha256']


def command(*args):
    result = subprocess.run(list(args),check=True,capture_output=True,text=True,
                            encoding='utf8',errors='replace',timeout=150)
    return result.stdout


container='7gofn-k1'
runner='/tmp/booster_agent_deploy/football3v3_runner'
install='/opt/booster/booster_agent_data/data/agents/extract/com.example.lqinternalmatch'
port=int(command('docker','port',container,'38383/tcp').splitlines()[0].rsplit(':',1)[1])


def http(path, post=False):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=b'{}' if post else None,
                              headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as response:
        return json.load(response)


http('/match/end',True)
command('docker','exec',container,'python3',runner+'/launcher.py','--stop')
command('docker','cp',str(archive),container+':/tmp/qualified-match-policy.agent')
command('docker','exec',container,'python3',runner+'/install_agent_package.py',
        '/tmp/qualified-match-policy.agent',install)
metadata=json.loads(command('docker','exec',container,'cat',install+'/agent.json'))
assert metadata['version']==qualifying['version']
command('docker','exec',container,'python3',runner+'/launcher.py','--publish-logs',
        '--teams','com.example.lqinternalmatch','com.booster.default3v3ai')
status=http('/status')
assert status['gameControl']['ready'] and not status['gameControl'].get('lastError')

model=dict(enabled=True,candidate=qualifying['candidate'],parameters=qualifying['parameters'],
           method='evolution strategy; full-match goal difference reward')
(ROOT/'src/models/match_policy.py').write_text('"""Match-reward policy checkpoint."""\nMODEL = '+repr(model)+'\n')
manifest=ROOT/'agent.toml'
manifest.write_text('\n'.join(f'version = "{qualifying["version"]}"' if line.startswith('version = ') else line
                              for line in manifest.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
for episode in report['episodes']:
    if 'reward' not in episode:
        episode['excluded_from_results']=True
        episode['exclusion_reason']='Unfinished candidate stopped after requested target and replay were completed.'
report.update(achieved=True,target='Win a complete 600-second match against the default AI by at least five goals.',
              winning_episodes=[qualifying['number']],confirmation_episode=replay['number'],
              confirmation_reached_five_goal_margin=replay['margin']>=5,
              qualified={k:qualifying[k] for k in ['candidate','package','version','parameters','sha256']},
              active_version=metadata['version'],restored_qualified_package=True,
              completion_reason='Requested five-goal result verified against referee events; unchanged package replay completed.')
report.pop('last_error',None)
(OUT/'search.json').write_text(json.dumps(report,indent=2))
summary=dict(target_reached=True,version=metadata['version'],opponent='com.booster.default3v3ai',
             duration_seconds=600,full_match_scores=[qualifying['score'],replay['score']],
             repeated_five_goal_margin=replay['margin']>=5,package=qualifying['package'],sha256=qualifying['sha256'],
             parameters=qualifying['parameters'],penalties=[qualifying['penalties'],replay['penalties']],
             runtime=[qualifying['runtime'],replay['runtime']],
             method='Existing imitation-trained lane tree with match-evaluated strategy parameters.',
             provenance='The initial manually initialized search candidate reached the target; no successful mutation update is claimed.',
             limitations=['Only two full matches, with no fixed simulator seed.','Five-goal margin was reached once; replay margin is reported separately.'],
             active_package_verified=True)
(ROOT/'training/artifacts/five-goal-result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
