"""Independent blinded direction judgments, after both prediction files are frozen."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from .llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
from .schemas import require
from .storage import read_json, write_json

JUDGE_TASK = 'Compare the predicted strategy with the human Before/After edit. Judge both methods by the same standard.\nCONSISTENT: the concrete operation and target broadly match a salient human change. Exact replacement words and\nprediction of every minor extra edit are unnecessary. A vague "make it neutral" goal alone is insufficient.\nINCONSISTENT: a concrete prediction materially differs in operation or target. UNCERTAIN: insufficient information.\nDo not reward contradictory alternative lists or infer missing facts. Method identity is absent; use only supplied text.\nReturn {"judgment":"CONSISTENT"|"INCONSISTENT"|"UNCERTAIN","reason":str}; explain briefly.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION


def judge_decode(raw):
    require(set(raw) == {'judgment', 'reason'} and raw['judgment'] in ('CONSISTENT', 'INCONSISTENT', 'UNCERTAIN')
            and isinstance(raw['reason'], str) and bool(raw['reason'].strip()), 'Invalid independent judgment')
    return raw


def judge_payload(pair, prediction):
    # An explicit allowlist prevents method identity, rule IDs, categories or scores entering the judge.
    return {'before': pair['before'], 'after': pair['after'], 'predicted_strategy': prediction['strategy']}


def evaluate(pair_path, before_path, predictions_directory, output, backend, expected_count=200):
    predictions_directory, output = Path(predictions_directory), Path(output)
    completion = read_json(predictions_directory / 'prediction_complete.json')
    require(completion['status'] == 'complete' and completion['count_per_method'] == expected_count,
            'Both complete prediction files are required before loading held-out After')
    require(hashlib.sha256(Path(before_path).read_bytes()).hexdigest() == completion['input_sha256'], 'Prediction inputs changed before evaluation')
    predictions = {}
    expected_ids = {f'test-{i + 1:04d}' for i in range(expected_count)}
    for method in ('theory', 'npov'):
        path = predictions_directory / (method + '_predictions.json')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == completion['prediction_files_sha256'][method], 'Prediction file modified after completion')
        rows = read_json(path)
        require(len(rows) == expected_count and {r['case_id'] for r in rows} == expected_ids, 'Missing/duplicate predictions')
        predictions[method] = {r['case_id']: r for r in rows}
    # After is opened only below, after all prediction integrity/completion checks.
    with Path(pair_path).open(encoding='utf-8') as handle:
        pairs = [json.loads(line) for line in handle]
    with Path(before_path).open(encoding='utf-8') as handle:
        befores = [json.loads(line) for line in handle]
    require(len(pairs) == len(befores) == expected_count, 'Wrong evaluation cohort size')
    require(all(set(p) == {'before', 'after'} and set(b) == {'before'} and p['before'] == b['before']
                for p, b in zip(pairs, befores)), 'Evaluation pair/input mismatch')
    tasks = [(method, index) for method in predictions for index in range(expected_count)]
    if hasattr(backend, 'reserve_work_calls'):
        backend.reserve_work_calls(len(tasks) * (backend.config.retries + 1) ** 2, 'heldout_direction_evaluation')
    random.Random(42).shuffle(tasks)
    judgments = {m: [] for m in predictions}
    for progress, (method, index) in enumerate(tasks):
        import logging
        logging.getLogger(__name__).info('Independent blinded direction judgment %d/%d', progress + 1, len(tasks))
        cid = f'test-{index + 1:04d}'
        prediction = predictions[method][cid]
        if prediction['abstained']:
            judgment = {'judgment': 'UNCERTAIN', 'reason': 'The predictor abstained; no concrete strategy was predicted.'}
        else:
            judgment = checked_infer(backend, JUDGE_TASK, judge_payload(pairs[index], prediction), judge_decode)
        judgments[method].append({'case_id': cid, **judgment})
        write_json(output / (method + '_judgments.json'), judgments[method])
    summary = {}
    for method, rows in judgments.items():
        counts = Counter(r['judgment'] for r in rows)
        summary[method] = {'cases': expected_count, 'consistent': counts['CONSISTENT'],
            'inconsistent': counts['INCONSISTENT'], 'uncertain': counts['UNCERTAIN'],
            'abstentions': sum(p['abstained'] for p in predictions[method].values()),
            'direction_consistency_rate_all_cases': counts['CONSISTENT'] / expected_count,
            'rate_on_decidable_cases': counts['CONSISTENT'] / (counts['CONSISTENT'] + counts['INCONSISTENT'])
                if counts['CONSISTENT'] + counts['INCONSISTENT'] else None}
    paired = []
    by_method = {m: {r['case_id']: r for r in rows} for m, rows in judgments.items()}
    for index, pair in enumerate(pairs):
        cid = f'test-{index + 1:04d}'
        paired.append({'case_id': cid, **pair, **{m: {'prediction': predictions[m][cid],
            'evaluation': by_method[m][cid]} for m in predictions}})
    write_json(output / 'paired_comparison.json', paired)
    write_json(output / 'comparison_summary.json', {'status': 'complete', 'evaluator': backend.config.model,
        'criterion': 'general editing direction; exact wording not required', 'blinded_to_method': True,
        'method_display_names': {'theory': 'Theory-guided', 'npov': 'NPOV baseline'},
        'methods': summary, 'human_validated': False})
    report = ['# Held-out direction comparison', '',
              'The frozen predictive theory and NPOV baseline predicted the same 200 Before-only cases. Gemma 4 judged the frozen predictions against human edits without method labels.', '',
              '| Method | Consistent | Inconsistent | Uncertain | Abstained | Consistent / all 200 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, stats in summary.items():
        report.append(f"| { {'theory': 'Theory-guided', 'npov': 'NPOV baseline'}[method]} | {stats['consistent']} | {stats['inconsistent']} | {stats['uncertain']} | {stats['abstentions']} | {stats['direction_consistency_rate_all_cases']:.1%} |")
    report += ['', 'Matching the broad concrete direction is sufficient; exact replacement wording is not required.',
               'These are automatic model judgments, not human-verified ground truth. Uncertain cases and abstentions remain in the main denominator.']
    (output / 'comparison_report.md').write_text('\n'.join(report) + '\n')
    return summary
