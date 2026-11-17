"""Train on live teacher records, export a dependency-free tree, and evaluate held-out time blocks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.framework.types import ADULT_FIELD_DIMENSIONS, BallState, Context, Pose2D, RobotState
from src.ml_policy import FEATURE_NAMES, encode_state, predict_tree
from src.param import SHOT_POST_MARGIN_M, SHOT_TARGET_COUNT
from src.utils.tactics import choose_shot_target


def label_state(state):
    ctx = Context(0, 0.2, 1, ADULT_FIELD_DIMENSIONS, ball=BallState(*state['ball']),
                  opponents={i: RobotState(i, Pose2D(*p)) for i, p in enumerate(state['opponents'], 10)},
                  teammates={i: RobotState(i, Pose2D(*p)) for i, p in enumerate(state['teammates'], 1)})
    tx, ty = choose_shot_target(ctx, 0)
    bx, by = state['ball']
    crossing = by + (ty-by)*(state['length']/2-bx)/(tx-bx)
    half = state['goal_width']/2-SHOT_POST_MARGIN_M
    return max(0, min(SHOT_TARGET_COUNT-1, round((crossing/half+1)*(SHOT_TARGET_COUNT-1)/2)))


def augmented(state, rng):
    state = copy.deepcopy(state)
    flip = rng.choice([-1, 1])
    for kind in ['ball', 'opponents', 'teammates']:
        points = [state[kind]] if kind == 'ball' else state[kind]
        for point in points:
            point[0] = float(np.clip(point[0] + rng.normal(0, 0.35), -6.7, 6.7))
            point[1] = float(np.clip(flip*point[1] + rng.normal(0, 0.35), -4.2, 4.2))
    return state


def export_tree(clf):
    tree = clf.tree_
    nodes = []
    for i in range(tree.node_count):
        value = tree.value[i][0]
        label = int(clf.classes_[value.argmax()])
        confidence = float(value.max()/value.sum())
        nodes.append([int(tree.feature[i]), float(tree.threshold[i]),
                      int(tree.children_left[i]), int(tree.children_right[i]), label, confidence])
    return dict(format_version=1, model_id='imitation-shooting-v1',
                feature_names=FEATURE_NAMES, min_confidence=0.85, nodes=nodes,
                teacher_version='1.1.2', field=[14.0, 9.0, 2.6])


def evaluate(clf, records, confidence):
    x = np.asarray([encode_state(r['state']) for r in records], dtype=np.float32)
    y = np.asarray([r['label'] for r in records])
    pred, probs = clf.predict(x), clf.predict_proba(x).max(axis=1)
    accepted = probs >= confidence
    return dict(samples=len(y), agreement=float(accuracy_score(y, pred)),
                accepted_coverage=float(accepted.mean()),
                accepted_agreement=float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else None,
                majority_baseline=float(max(np.bincount(y, minlength=9))/len(y)),
                class_counts=np.bincount(y, minlength=9).tolist(),
                confusion_matrix=confusion_matrix(y, pred, labels=list(range(9))).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path('src/models/shot_policy.json'))
    parser.add_argument('--report', type=Path, default=Path('training/artifacts/training-report.json'))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(records) < 500:
        raise SystemExit('Need at least 500 real match examples before training.')
    if any(label_state(r['state']) != r['label'] for r in records):
        raise SystemExit('Recorded labels do not match the frozen teacher; check dataset provenance.')
    # Split before augmentation; no shuffled neighboring-frame validation.
    n = len(records)
    train, valid, test = records[:int(n*.6)-10], records[int(n*.6)+10:int(n*.8)-10], records[int(n*.8)+10:]
    rng = np.random.default_rng(42)
    x = [encode_state(r['state']) for r in train]
    y = [r['label'] for r in train]
    for record in train:
        for _ in range(12):
            state = augmented(record['state'], rng)
            x.append(encode_state(state))
            y.append(label_state(state))
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y)
    candidates = []
    for depth in [6, 10, 14, 18]:
        for leaf in [5, 20]:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=42)
            clf.fit(x, y)
            metrics = evaluate(clf, valid, .85)
            candidates.append((metrics['agreement'], -clf.tree_.node_count, clf, metrics))
    _, _, clf, validation = max(candidates, key=lambda entry: entry[:2])
    model = export_tree(clf)
    # Confidence threshold chosen on validation only. The final test is untouched.
    for threshold in [.6, .75, .85, .95]:
        metrics = evaluate(clf, valid, threshold)
        if (metrics['accepted_agreement'] or 0) >= .95 and metrics['accepted_coverage'] >= .2:
            model['min_confidence'] = threshold
            break
    model['data_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    all_x = np.asarray([encode_state(r['state']) for r in valid+test], dtype=np.float32)
    native_predictions = clf.predict(all_x)
    portable_predictions = np.asarray([predict_tree(model, row.tolist())[0] for row in all_x])
    parity = bool(np.array_equal(native_predictions, portable_predictions))
    if not parity:
        raise SystemExit('Export parity failed; refusing to write model.')
    test_metrics = evaluate(clf, test, model['min_confidence'])
    report = dict(method='DecisionTreeClassifier behavior cloning of the 1.1.2 shot planner',
                  scope='shooting lane only; role assignment and restart legality remain coded',
                  real_examples=n, training_real_examples=len(train), augmented_training_examples=len(x)-len(train),
                  real_class_counts=np.bincount([r['label'] for r in records], minlength=9).tolist(),
                  split='chronological 60/20/20, 10-record purge on each boundary, augment training only',
                  validation=evaluate(clf, valid, model['min_confidence']), test=test_metrics,
                  nodes=len(model['nodes']), max_depth=clf.get_depth(), min_confidence=model['min_confidence'],
                  export_parity=parity, teacher_version='1.1.2', data_sha256=model['data_sha256'],
                  limitations=['Single teacher match; temporal test, not an independent match.',
                               'Augmented samples are synthetic teacher-labelled perturbations of training states.',
                               'Imitation agreement is not evidence of improved match strength.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, separators=(',', ':')))
    args.output.with_suffix('.py').write_text('# Generated model data; trained by training/train_imitation.py.\nMODEL = ' + repr(model) + '\n')
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
