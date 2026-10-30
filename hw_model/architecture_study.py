"""Independent architecture experiment; the main compare command is unchanged.

Run all eight mappings on both fixed subgraphs, freeze sources, and require
exact compatibility with every result in the original fixed-300 experiment.
Replay uses the copied baseline reference, so the frozen source tree is enough.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import sys

from .compute import Shape
from .config import load_config, source_manifest
from .experiments import ROOT, _freeze_sources, config_hash, dump
from .fixed_schedule import mapping_traits
from .fixed_simulation import simulate
from .reporting import _csv, _flat


BASELINE_MAPPINGS = ('A', 'B0', 'B1_G9', 'B1_G99')
NEW_MAPPINGS = ('A18x8', 'A16x8', 'A32x4', 'B1_G11')
STUDY_MAPPINGS = BASELINE_MAPPINGS + NEW_MAPPINGS
OPERATORS = ('projection', 'ffn2')
BASELINE_REPORT = ROOT/'hw_model'/'reports'/'fixed300_20261029_v2'
BOUNDARY = 'first_weight_request_to_final_internal_int8_output_ack'


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _key(result):
    return result['operator'], result['mapping']


def _index(results):
    rows = {}
    for result in results:
        key = _key(result)
        if key in rows:
            raise ValueError('duplicate result: %s/%s' % key)
        rows[key] = result
    return rows


def _differences(expected, actual, path='', limit=25):
    """A bounded diagnostic; equality itself always compares complete dicts."""
    found = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            where = path+'.'+str(key) if path else str(key)
            if key not in expected or key not in actual:
                found.append(where+': missing or extra field')
            elif expected[key] != actual[key]:
                found.extend(_differences(expected[key], actual[key], where, limit-len(found)))
            if len(found) >= limit:
                break
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            found.append(path+': list lengths %s/%s' % (len(expected), len(actual)))
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                found.extend(_differences(left, right, path+'[%d]' % index, limit-len(found)))
            if len(found) >= limit:
                break
    else:
        found.append('%s: expected=%r actual=%r' % (path, expected, actual))
    return found[:limit]


def baseline_compatibility(results, baseline):
    """Compare all fields of the eight original results, without source hashes."""
    expected = _index(baseline['results'])
    required = {(operator, mapping) for operator in OPERATORS for mapping in BASELINE_MAPPINGS}
    if set(expected) != required:
        raise ValueError('reference baseline must contain exactly the eight original cases')
    actual = _index(results)
    cases = []
    for key in sorted(required):
        reference, current = expected[key], actual.get(key)
        equal = current == reference
        cases.append({'operator': key[0], 'mapping': key[1], 'exact_dict_equal': equal,
                      'reference_result_sha256': _hash(reference),
                      'actual_result_sha256': None if current is None else _hash(current),
                      'differences': [] if equal else _differences(reference, current)})
    return {'passed': all(case['exact_dict_equal'] for case in cases),
            'comparison': 'complete result dictionaries, not source_manifest',
            'reference_result_count': len(expected), 'compared_result_count': len(cases),
            'reference_summary_sha256': _hash(baseline), 'cases': cases}


def run_cases(config, mappings=STUDY_MAPPINGS):
    """Run fixed-resource metadata simulations; a subset supports smoke checks."""
    results, traits = [], []
    model = config['model']
    for operator in OPERATORS:
        shape = Shape(model['tokens'], model['hidden'],
                      model['hidden'] if operator == 'projection' else model['ffn'])
        for mapping in mappings:
            metadata = mapping_traits(shape, mapping)
            result, _ = simulate(config, mapping, operator, capture_trace=False)
            if result.get('scope') != BOUNDARY:
                raise AssertionError('unexpected comparison boundary for '+mapping)
            if result['shared_activation_read_bytes'] or result['shared_activation_write_bytes']:
                raise AssertionError('resident activation contract violated by '+mapping)
            if sum(result['cycle_breakdown'].values()) != result['total_cycles']:
                raise AssertionError('cycle breakdown does not partition the comparison window')
            results.append(result)
            traits.append({'operator': operator, 'mapping': mapping, 'traits': metadata})
            print('%s %s: %d cycles, first matrix row %d, final matrix row %d' %
                  (operator, mapping, result['total_cycles'], result['first_matrix_row'],
                   max(result['row_matrix_cycles'])), flush=True)
    return results, traits


def _derived(results, traits):
    indexed = _index(results)
    metadata = {_key(item): item['traits'] for item in traits}
    rows = []
    for result in results:
        key = _key(result)
        data = metadata[key]
        resources = result['resources']
        registers = resources['register_payload_bytes']
        port, local = result['weight_port'], result['local_memory']
        final_matrix = max(result['row_matrix_cycles'])
        row = {field: result[field] for field in ('operator', 'mapping', 'architecture', 'total_cycles',
               'first_matrix_fragment', 'first_matrix_row', 'first_ln_row_done', 'last_internal_output_confirm',
               'physical_multipliers', 'issued_updates', 'issue_lane_utilization', 'issue_duty_cycle', 'compute_utilization')}
        row.update(group_rows=data['group_rows'], column_lanes=data['column_lanes'],
                   reduction_lanes=data['reduction_lanes'], final_matrix_row=final_matrix,
                   after_matrix_tail_cycles=result['total_cycles']-final_matrix,
                   useful_weight_bytes=port['useful_weight_bytes'], physical_weight_bytes=port['physical_weight_bytes'],
                   local_physical_read_bytes=local['physical_read_bytes'], local_physical_write_bytes=local['physical_write_bytes'],
                   local_denied_cycles=local['denied_cycles'], local_sram_bytes=resources['local_sram_bytes'],
                   register_payload_bytes=resources['total_register_payload_bytes'],
                   weight_register_bytes=registers['weight_register_bytes'],
                   activation_register_bytes=registers['activation_register_bytes'],
                   accumulator_register_bytes=registers['accumulator_register_bytes'],
                   partial_sram_bytes=resources['b1_partial_sram_bytes'],
                   area_status=resources['area']['status'])
        for reference in ('A', 'B1_G9', 'B1_G99', 'B1_G11'):
            other = indexed.get((result['operator'], reference))
            if other is not None:
                row['speedup_vs_'+reference] = other['total_cycles']/result['total_cycles']
        rows.append(row)
    return rows


def _tables(destination, results, traits, derived):
    _csv(destination/'operators.csv', derived)
    _csv(destination/'traits.csv', [_flat(item) for item in traits])
    collections = {name: [] for name in ('memory', 'registers', 'buffers', 'initialization', 'cycle_breakdown')}
    for result in results:
        identity = {key: result[key] for key in ('mapping', 'architecture', 'operator')}
        memory = dict(identity)
        for key in ('weight_port', 'local_memory', 'physical_local_layout'):
            memory.update(_flat(result[key], key))
        collections['memory'].append(memory)
        collections['registers'].append(dict(identity, **_flat(result['resources'])))
        for key in ('buffers', 'initialization', 'cycle_breakdown'):
            collections[key].append(dict(identity, **_flat(result[key])))
    for name, rows in collections.items():
        _csv(destination/(name+'.csv'), rows)


def _number(value):
    return 'unknown' if value is None else '%.6g' % value if isinstance(value, float) else str(value)


def _table(columns, rows):
    return '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (
        ''.join('<th>%s</th>' % escape(label) for _, label in columns),
        ''.join('<tr>%s</tr>' % ''.join('<td>%s</td>' % escape(_number(row.get(key))) for key, _ in columns) for row in rows))


def _analysis(results, derived):
    lines = ['# 新阵列形状与 G=11 调度的行为时序研究', '',
             '固定 150 MHz、300 MB/s 权重端口、同一 64 KiB 局部 SRAM、LN 核心 128/128/1。',
             '测量从第一次权重请求到最后内部 INT8 输出确认；初始化单列，不含共享输出写回。', '',
             '原始 A、B0、B1_G9、B1_G99 的八个完整结果字典与 fixed300_20261029_v2 完全一致；详见 baseline_compatibility.json。', '']
    for operator in OPERATORS:
        candidates = [row for row in derived if row['operator'] == operator]
        ranked = sorted(candidates, key=lambda row: row['total_cycles'])
        winner = ranked[0]
        lines += ['## '+operator, '',
                  '本模型下完成最快为 %s：%s cycles；相对原始 A 的周期比为 %.4g 倍（A 周期 / 当前周期）。' %
                  (winner['mapping'], winner['total_cycles'], winner['speedup_vs_A']), '',
                  '| 映射 | 完成周期 | 首矩阵行 | 全部矩阵行 | 矩阵后尾部 | 相对 A 加速比 |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in ranked:
            lines.append('| %s | %d | %d | %d | %d | %.6g |' %
                         (row['mapping'], row['total_cycles'], row['first_matrix_row'], row['final_matrix_row'],
                          row['after_matrix_tail_cycles'], row['speedup_vs_A']))
        lines += ['', '新阵列与 B 的三个复用组配置：', '']
        for row in candidates:
            if row['mapping'] not in ('A18x8', 'A16x8', 'A32x4'):
                continue
            lines.append('- %s：%d 乘法器，权重/激活/累加寄存器 %d/%d/%d B；权重物理读取 %d B，局部物理读写 %d/%d B。相对 G9/G99/G11 的周期比分别 %.4g/%.4g/%.4g。' %
                         (row['mapping'], row['physical_multipliers'], row['weight_register_bytes'],
                          row['activation_register_bytes'], row['accumulator_register_bytes'], row['physical_weight_bytes'],
                          row['local_physical_read_bytes'], row['local_physical_write_bytes'],
                          row['speedup_vs_B1_G9'], row['speedup_vs_B1_G99'], row['speedup_vs_B1_G11']))
        lines.append('')
    lines += ['## 如何理解资源与周期', '',
              '- 新阵列改变行/列并行度和权重包宽度。每包消费者增多可减少权重遍历，但更宽 token 组会改变 bank 冲突、活跃行和 LN 排队。尾组无效物理 lane 不计为有效 MAC。',
              '- A18x8、A16x8、A32x4 的乘法器、权重/激活寄存器和累加器均按真实 traits 与模拟器容量列账；没有假设它们与原始 A 资源完全相同。',
              '- B1_G11 只改同一 B 硬件的跨 token 复用调度。部分和留在同一局部 SRAM；重读激活和部分和访问继续占用公共 bank。',
              '- “矩阵后尾部”=最终内部输出确认−最后矩阵行完成，只描述该时间区间；不能把它当成全部 LN 忙碌周期。LN 与矩阵重叠，cycle_breakdown.csv 才是每周期互斥分类。',
              '- 发射有效比例分母为物理乘法器×发射次数；发射占空比分母为总周期；端到端利用率分母为物理乘法器×总周期。', '',
              '## 验证边界', '',
              '这是已实现并运行的元数据行为时序模型，不是 RTL 仿真或综合。独立数值参考仍采用浮点 LN；没有声称位精确定点 LN、真实 checkpoint 准确率、可达频率、综合面积或功耗。',
              '不同阵列形状是分别建模的独立硬件候选，不代表已经实现一套可重构阵列；形状切换、控制、互连与布线成本尚未综合。',
              '局部 SRAM 固定八个宏；面积缺失项保持 unknown，未由周期改善推断满足 2.25 mm²。输入初装、排列和参数初始化仍单列。',
              '所有结果、traits、配置、基线参考、源码哈希与快照随实验保存。精确重放需在 source/ 目录使用同一 Python 版本运行 architecture_study --replay <报告绝对路径>。', '']
    return '\n'.join(lines)


def _chart(derived):
    maximum = max(row['total_cycles'] for row in derived)
    bars = []
    for index, row in enumerate(derived):
        y, length = 18+index*29, 610*row['total_cycles']/maximum
        color = '#177a74' if row['mapping'].startswith('A') else '#9b6b2e' if row['mapping'] == 'B0' else '#5974b0'
        bars.append('<text x="175" y="%d" text-anchor="end">%s · %s</text><rect x="190" y="%d" width="%.2f" height="19" rx="3" fill="%s"/><text x="%.2f" y="%d">%d</text>' %
                    (y+14, escape(row['operator']), escape(row['mapping']), y, length, color, 198+length, y+14, row['total_cycles']))
    return '<svg role="img" aria-label="completion cycles for eight mappings and two operators" viewBox="0 0 940 %d">%s</svg>' % (len(derived)*29+35, ''.join(bars))


def _html(config, results, traits, derived, manifest, analysis):
    selected = [row for row in derived if row['mapping'] != 'B0']
    milestones = [('operator','子图'),('mapping','映射'),('total_cycles','完成周期'),('first_matrix_fragment','首矩阵片段'),
                  ('first_matrix_row','首矩阵行'),('final_matrix_row','最后矩阵行'),('first_ln_row_done','首 LN 行'),('after_matrix_tail_cycles','矩阵后尾部')]
    comparisons = [('operator','子图'),('mapping','映射'),('speedup_vs_A','A / 当前'),('speedup_vs_B1_G9','G9 / 当前'),
                   ('speedup_vs_B1_G99','G99 / 当前'),('speedup_vs_B1_G11','G11 / 当前')]
    resources = [('operator','子图'),('mapping','映射'),('physical_multipliers','乘法器'),('group_rows','token 组'),
                 ('column_lanes','输出 lanes'),('weight_register_bytes','权重寄存器 B'),('activation_register_bytes','激活寄存器 B'),
                 ('accumulator_register_bytes','累加器 B'),('register_payload_bytes','总寄存器 B'),('partial_sram_bytes','部分和 SRAM B')]
    traffic = [('operator','子图'),('mapping','映射'),('physical_weight_bytes','权重物理 B'),('local_physical_read_bytes','局部物理读 B'),
               ('local_physical_write_bytes','局部物理写 B'),('local_denied_cycles','局部冲突周期')]
    utilization = [('operator','子图'),('mapping','映射'),('issue_lane_utilization','发射有效比例'),('issue_duty_cycle','发射占空比'),('compute_utilization','端到端利用率')]
    detail = []
    for result in results:
        data = {key: result[key] for key in ('architecture','cycle_breakdown','initialization','buffers','resources','physical_local_layout')}
        detail.append('<details><summary>%s · %s：周期拆分与物理资源</summary><pre>%s</pre></details>' %
                      (escape(result['operator']), escape(result['mapping']), escape(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))))
    style = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#edf3f5;color:#243943;font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif}
    main{max-width:1370px;margin:30px auto;padding:0 26px 45px}header{background:#173e4a;color:white;padding:28px 34px;border-radius:12px}
    h1{font-size:30px;margin:6px 0 12px;line-height:1.4}.eyebrow{font-size:12px;letter-spacing:2px;color:#a8d6cf}header p{margin:7px 0;color:#d1e5e8}
    section{padding:24px 28px;margin:20px 0;background:white;border:1px solid #d9e3e8;border-radius:10px}h2{font-size:21px;margin:0 0 13px}
    .notice{background:#fff7e7;border-left:4px solid #c68a35;padding:15px 20px;margin:18px 0}.muted{font-size:13px;color:#607884}
    .scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;white-space:nowrap;text-align:left;border-bottom:1px solid #dce5e9}th{background:#f2f7f8}
    svg{width:100%;height:auto;display:block}svg text{font:12px system-ui,'Microsoft YaHei',sans-serif;fill:#647a84}details{border-top:1px solid #dce5e9;padding:12px 0}summary{cursor:pointer;color:#177a74;font-weight:600}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:16px;font:12px/1.65 Consolas,monospace}code{font:12px Consolas,monospace;overflow-wrap:anywhere}footer{color:#607884;font-size:12px}
    @media(max-width:700px){main{padding:0 12px;margin:12px auto}header,section{padding:18px}h1{font-size:24px}}
    '''
    return ''.join([
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KWT 新阵列形状行为时序研究</title><style>',style,'</style></head><body><main>',
        '<header><div class="eyebrow">KWT · ARCHITECTURE STUDY · FIXED 300 MB/s</div><h1>18×8、16×8、32×4 阵列与 G=11 复用调度</h1><p>原始四种映射 + 四个新候选，共 16 个融合子图实验</p><p>主 compare 入口保持不变；原始八个完整结果字典兼容性检查通过。</p></header>',
        '<div class="notice"><strong>已实现的行为时序研究：</strong>固定同一 150 MHz、300 MB/s 权重端口、64 KiB 单端口激活 SRAM 和 LN128/II128/单行。窗口从第一次权重请求到最后内部 INT8 输出确认。初始化单列，不含共享 SRAM 输出写回。不同阵列形状是独立候选，未实现一套可重构硬件；形状切换控制与互连成本未综合。结果来自元数据仿真，不是 RTL、综合、功耗或位精确定点 LN 验证。</div>',
        '<section><h2>完成周期与阶段边界</h2>',_chart(derived),_table(milestones,derived),'<p class="muted">矩阵后尾部只是最后矩阵行至最终内部输出的区间；LN 与矩阵重叠，不能据此替代全部 LN 忙碌周期。</p></section>',
        '<section><h2>相对 A / G9 / G99 / G11</h2>',_table(comparisons,selected),'<p class="muted">比值＝参照映射完成周期 / 当前映射完成周期；大于 1 表示当前候选较快。三种新阵列的实际资源变化见下一表。</p></section>',
        '<section><h2>并行形状与显式寄存器</h2>',_table(resources,derived),'<p class="muted">架构名称及物理资源取自核心 traits 和资源清单，不按名称推测。每个候选保持八个 4096×16-bit 激活宏；没有整矩阵权重 SRAM。部分和 SRAM 已在同一物理宏中，未重复计算寄存器。宏面积、标准单元面积和控制/重排逻辑的实际开销仍需 RTL 与库数据。</p></section>',
        '<section><h2>权重与局部流量</h2>',_table(traffic,derived),'<p class="muted">权重端口只有 4 B/beat、II2、响应3、最多2在途；所有激活、残差、部分和、LN提取和输出写入争用同一局部 bank。输入保持驻留，主窗口共享激活访问为零。</p></section>',
        '<section><h2>三类利用率</h2>',_table(utilization,derived),'<p class="muted">发射有效比例＝MAC/(乘法器×发射次数)；发射占空比＝发射次数/总周期；端到端利用率＝MAC/(乘法器×总周期)。均为 0 至 1 比例。</p></section>',
        '<section><h2>分析与验证边界</h2><pre>',escape(analysis),'</pre></section>',
        '<section><h2>逐候选周期拆分、初始化、缓冲与资源原始字段</h2>',''.join(detail),'</section>',
        '<footer><p>配置 SHA-256：<code>',escape(config_hash(config)),'</code><br>源码集合 SHA-256：<code>',escape(manifest['aggregate_sha256']),'</code></p><p>summary.json 保留 16 个原始结果；operators/memory/registers/buffers/initialization/cycle_breakdown/traits.csv 为分项；baseline_compatibility.json 和 verification.json 记录兼容与重放结果。source/ 保存精确源码，baseline_reference_summary.json 保存原始基线参考。</p></footer></main></body></html>'])


def build_study(destination):
    config = load_config()
    reference = _load(BASELINE_REPORT/'summary.json')
    reference_config = _load(BASELINE_REPORT/'run_config.json')
    if config != reference_config:
        raise ValueError('current fixed config differs from the original baseline config')
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest(ROOT)
    _freeze_sources(destination, manifest)
    dump(destination/'run_config.json', config)
    dump(destination/'source_manifest.json', manifest)
    dump(destination/'baseline_reference_summary.json', reference)
    dump(destination/'invocation.json', {'argv':sys.argv, 'python':sys.version, 'executable':sys.executable,
                                       'client_timezone':'Asia/Shanghai',
                                       'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
                                       'entrypoint':'hw_model.architecture_study', 'trace_enabled':False})
    results, traits = run_cases(config)
    if len(results) != 16:
        raise AssertionError('study requires all sixteen detailed cases')
    compatibility = baseline_compatibility(results, reference)
    dump(destination/'baseline_compatibility.json', compatibility)
    if not compatibility['passed']:
        dump(destination/'verification.json', {'status':'failed', 'baseline_exact_dicts':False,
                                             'exact_replay':False, 'details':'see baseline_compatibility.json'})
        raise AssertionError('original baseline result dictionaries differ; study is not complete')
    if source_manifest(ROOT) != manifest:
        raise ValueError('source files changed during the study; refusing an inconsistent source bundle')
    derived = _derived(results, traits)
    summary = {'schema_version':1, 'study':'architecture_shapes_fixed300', 'config_sha256':config_hash(config),
               'measurement_boundary':BOUNDARY, 'source_manifest':manifest,
               'baseline_reference_sha256':_hash(reference), 'mappings':list(STUDY_MAPPINGS),
               'results':results, 'mapping_traits':traits, 'derived_metrics':derived,
               'claims':{'metadata_behavior_simulation':True, 'baseline_complete_dicts_identical':True,
                         'rtl_verified':False, 'area_synthesized':False, 'power_measured':False,
                         'fixed_point_ln_bit_accuracy_verified':False, 'checkpoint_accuracy_measured':False}}
    dump(destination/'summary.json', summary)
    _tables(destination, results, traits, derived)
    analysis = _analysis(results, derived)
    (destination/'analysis.md').write_text(analysis, encoding='utf-8')
    (destination/'report.html').write_text(_html(config, results, traits, derived, manifest, analysis), encoding='utf-8')
    dump(destination/'verification.json', {'status':'simulation_and_baseline_pass_replay_pending',
                                         'result_count':16, 'baseline_exact_dicts':True,
                                         'baseline_case_count':8, 'source_bundle_consistent':True,
                                         'exact_replay':False})
    print('Architecture study: '+str(destination/'report.html'), flush=True)
    return 0


def replay_study(report):
    report = Path(report).absolute()
    saved = _load(report/'summary.json')
    if saved.get('study') != 'architecture_shapes_fixed300':
        raise ValueError('this is not an architecture study report')
    config = load_config(report/'run_config.json')
    if config_hash(config) != saved['config_sha256']:
        raise ValueError('saved config hash differs')
    manifest = source_manifest(ROOT)
    if manifest != saved['source_manifest']:
        raise ValueError('source/runtime differs; run replay from the frozen source/ directory with the recorded Python')
    reference = _load(report/'baseline_reference_summary.json')
    if _hash(reference) != saved['baseline_reference_sha256']:
        raise ValueError('copied baseline reference hash differs')
    results, traits = run_cases(config)
    compatibility = baseline_compatibility(results, reference)
    passed = (len(results) == 16 and results == saved['results'] and traits == saved['mapping_traits'] and
              _derived(results, traits) == saved['derived_metrics'] and compatibility['passed'])
    verification = {'status':'pass' if passed else 'failed', 'result_count':len(results),
                    'exact_replay':results == saved['results'], 'traits_exact':traits == saved['mapping_traits'],
                    'derived_metrics_exact':_derived(results, traits) == saved['derived_metrics'],
                    'baseline_exact_dicts':compatibility['passed'], 'baseline_case_count':8,
                    'source_bundle_consistent':True, 'python_version':manifest['python_version']}
    dump(report/'verification.json', verification)
    if not passed:
        raise AssertionError('architecture study exact replay differs from saved results')
    print('Exact architecture replay PASS: 16 complete result dictionaries; baseline 8 complete dictionaries identical', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent fixed300 architecture-shape experiment (16 cases)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', help='new immutable report directory')
    group.add_argument('--replay', help='existing report directory to reproduce exactly')
    args = parser.parse_args(argv)
    try:
        return replay_study(args.replay) if args.replay else build_study(args.out)
    except (ValueError, AssertionError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, 'error: %s\n' % error)


if __name__ == '__main__':
    raise SystemExit(main())
