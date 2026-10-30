"""Label the completed W8 metric columns and build a three-experiment layer table.

Does not run a model or change logits/scales. Historical experiment files are
read only. This postprocessing source and hash are saved with the new result.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.absolute()
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
    if 'w8a8_n16_accuracy' not in result or (run / 'postprocessing.json').exists():
        raise RuntimeError('Expected a completed, not yet finalized W8A8/N16 run')
    w16 = Path(metadata['w16_reference'])
    if run == w16 or run == Path(metadata['fp32_reference']):
        raise RuntimeError('Output must be the new experiment')
    for basename in ('layer_comparison', 'nonlinear_comparison'):
        path = run / (basename + '.json')
        rows = json.loads(path.read_text(encoding='utf-8'))
        for row in rows:
            if 'w16a16_max_abs' in row:
                row['w8a8_n16_max_abs'] = row.pop('w16a16_max_abs')
        write_json(path, rows)
        write_csv(run / (basename + '.csv'), rows)
    newer = json.loads((run / 'layer_comparison.json').read_text(encoding='utf-8'))
    older = {row['stage']: row for row in json.loads((w16 / 'layer_comparison.json').read_text(encoding='utf-8'))}
    metrics = ('mae', 'rmse', 'max_abs_error', 'relative_l2', 'cosine')
    combined = []
    for row in newer:
        combined.append(dict(stage=row['stage'], model='FP32', **{k: 1.0 if k == 'cosine' else 0.0 for k in metrics}))
        for name, values in [('W16A16_FP32_nonlinear', older[row['stage']]), ('W8A8_N16_AV_F6', row)]:
            combined.append(dict(stage=row['stage'], model=name, **{k: values[k] for k in metrics}))
    write_csv(run / 'layer_comparison_three_precisions.csv', combined)
    write_json(run / 'layer_comparison_three_precisions.json', combined)
    tools_dir = run / 'analysis_tools'
    tools_dir.mkdir(exist_ok=True)
    script = Path(__file__).absolute()
    shutil.copyfile(script, tools_dir / script.name)
    write_json(run / 'postprocessing.json', dict(
        inference_logits_unchanged=True, quantization_scales_unchanged=True,
        renamed_column={'w16a16_max_abs': 'w8a8_n16_max_abs'},
        reason='Shared metric accumulator had a legacy column name; numerical values are unchanged',
        combined_layer_table='layer_comparison_three_precisions.csv',
        source_file='analysis_tools/' + script.name,
        source_sha256=hashlib.sha256(script.read_bytes()).hexdigest()))
    with (run / 'README_量化结果.md').open('a', encoding='utf-8') as stream:
        stream.write('\n补充对照：`layer_comparison_three_precisions.csv` 汇总 FP32、上次 W16A16、本次 W8A8/N16 的每层指标，误差均相对于 FP32。'
                     '`nonlinear_comparison.csv` 对比整条量化路径的非线性节点输出，包含上游矩阵乘和 AV Q6 引入的累积误差，并非单独的非线性近似误差。'
                     '共享统计器中的旧列名已更正为 `w8a8_n16_max_abs`，未改动任何推理数值；见 `postprocessing.json`。\n')
    print(json.dumps(dict(status='completed', run=str(run), combined_layer_rows=len(combined)), ensure_ascii=False))


if __name__ == '__main__':
    main()
