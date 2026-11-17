"""Evaluate an exported policy on a new, unused live rollout. Does not fit a model."""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml_policy import encode_state, predict_tree

parser = argparse.ArgumentParser()
parser.add_argument('data', type=Path)
parser.add_argument('--model', type=Path, default=Path('src/models/shot_policy.json'))
parser.add_argument('--output', type=Path, default=Path('training/artifacts/independent-rollout-report.json'))
args = parser.parse_args()
model = json.loads(args.model.read_text())
records = [json.loads(line) for line in args.data.read_text().splitlines()]
if not records:
    raise SystemExit('No examples to evaluate')
correct = accepted = accepted_correct = 0
classes = collections.Counter()
for record in records:
    label, confidence = predict_tree(model, encode_state(record['state']))
    correct += label == record['label']
    classes[record['label']] += 1
    if confidence >= model['min_confidence']:
        accepted += 1
        accepted_correct += label == record['label']
report = dict(examples=len(records), episodes=sorted({r['episode'] for r in records}),
              exact_teacher_agreement=correct/len(records), accepted_coverage=accepted/len(records),
              accepted_agreement=accepted_correct/accepted if accepted else None,
              majority_baseline=max(classes.values())/len(records), class_counts=dict(classes),
              model_id=model['model_id'], training_data_sha256=model['data_sha256'],
              trained_on_this_rollout=False)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
