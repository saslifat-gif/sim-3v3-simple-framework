"""Flush local TensorBoard scalar events; no TensorFlow dependency."""
import json
import time
from pathlib import Path
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class Metrics:
    def __init__(self, directory, event_directory=None):
        self.directory=Path(directory)
        self.event_directory=Path(event_directory) if event_directory else self.directory.parent
        self.writer=EventFileWriter(str(self.directory))
        self.marker=self.directory/'imported.json'
        self.imported=set(json.loads(self.marker.read_text())) if self.marker.exists() else set()

    def scalars(self, values, step, wall_time=None):
        self.writer.add_event(Event(wall_time=wall_time or time.time(),step=int(step),
            summary=Summary(value=[Summary.Value(tag=k,simple_value=float(v)) for k,v in values.items()])))
        self.writer.flush()

    def episode(self,e):
        key=e['name']+':'+e.get('sha256','')
        if key in self.imported or e.get('excluded_from_learning'):return
        if e.get('training') and not e.get('update'):return
        if not e.get('transitions'):return
        step=int(e['update_index']) if 'update_index' in e else (int(e['name'].split('-')[-1]) if e['training'] else 4)
        prefix='train' if e['training'] else 'evaluation'
        values={f'{prefix}/goals_for':e['score'][0],f'{prefix}/goals_against':e['score'][1],
            f'{prefix}/goal_difference':e['score'][0]-e['score'][1],f'{prefix}/transitions':e['transitions'],
            f'{prefix}/actual_match_speed':e['match_seconds_per_wall_second']}
        for k,v in e.get('update',{}).items():
            if isinstance(v,(int,float)):values['learning/'+k]=v
        event_file=self.event_directory/f'{e["name"]}-events.jsonl'
        if event_file.exists():
            events=[json.loads(s) for s in event_file.read_text().splitlines()]
            penalties=[v for v in events if v.get('type')=='penalty' and v.get('actor',{}).get('side')=='home'
                       and e['started']-.5<=v['wallTime']<=e['finished']]
            values[f'{prefix}/penalties']=len(penalties)
        for action,count in e.get('action_counts',{}).items():values[f'{prefix}/action_{action}_count']=count
        self.scalars(values,step,e['finished'])
        self.imported.add(key)
        self.marker.write_text(json.dumps(sorted(self.imported)))

    def close(self):self.writer.close()


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'build/rl-training'
    metrics=Metrics(root/'tensorboard')
    for episode in json.loads((root/'report.json').read_text())['episodes']:metrics.episode(episode)
    metrics.close()
