"""The user's 32 KiB / LN64 architecture study, separate from frozen reports.

Every case is either a complete feasible result or a structured local-capacity
diagnosis. Capacity failures do not abort the other candidates. Historical
64 KiB / LN128 cases run separately and are never mixed into the new ranking.
"""
import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import sys

from .architecture_study import _hash, _load, _differences, _table
from .compute import Shape
from .config import REVISED_CONFIG, REVISED_PROFILE_ID, load_config, source_manifest
from .experiments import ROOT, _freeze_sources, config_hash, dump
from .fixed_memory import LocalCapacityError
from .fixed_schedule import MAPPINGS, EXPERIMENTAL_MAPPINGS, REVISION_MAPPINGS, mapping_traits
from .fixed_simulation import simulate
from .reporting import _csv, _flat


OPERATORS = ('projection', 'ffn2')
# Keep this independent 32 KiB experiment at its recorded twenty mappings;
# later registry extensions belong to the separate 64 KiB / LN64 study.
STUDY_MAPPINGS = MAPPINGS + EXPERIMENTAL_MAPPINGS + REVISION_MAPPINGS
REFERENCE_REPORT = ROOT/'hw_model'/'reports'/'architecture_study_20261029_v1'
REVISION_SPEC = 'hw_model/spec/user_revision_20261029_ln64_local32k.md'
BOUNDARY = 'first_weight_request_to_final_internal_int8_output_ack'
STUDY_ID = 'architecture_shapes_local32k_ln64'


def _shape(config, operator):
    model = config['model']
    return Shape(model['tokens'], model['hidden'], model['hidden'] if operator == 'projection' else model['ffn'])


def _capacity_reason(error):
    required, capacity = error.required_bytes, error.capacity_bytes
    return {'code':'local_capacity_exceeded', 'mapping':error.mapping,
            'required_bytes':required, 'capacity_bytes':capacity,
            'excess_bytes':required-capacity, 'tensors':error.tensors}


def _check_result(result, config):
    if result['scope'] != BOUNDARY:
        raise AssertionError('unexpected comparison boundary')
    if result['shared_activation_read_bytes'] or result['shared_activation_write_bytes']:
        raise AssertionError('local-residency contract violated')
    if result['local_memory']['banks'] != config['local_memory']['banks']:
        raise AssertionError('result uses the wrong local bank count')
    if sum(result['cycle_breakdown'].values()) != result['total_cycles']:
        raise AssertionError('cycle breakdown does not partition the measurement window')


def run_cases(config, mappings=None):
    """Capture only LocalCapacityError; all other implementation failures surface."""
    mappings = tuple(STUDY_MAPPINGS if mappings is None else mappings)
    cases = []
    for operator in OPERATORS:
        shape = _shape(config, operator)
        for mapping in mappings:
            traits = mapping_traits(shape, mapping)
            case = {'operator':operator, 'mapping':mapping, 'architecture':traits['architecture'],
                    'shape':{'m':shape.m, 'n':shape.n, 'r':shape.r}, 'traits':traits}
            try:
                result, _ = simulate(config, mapping, operator, capture_trace=False)
            except LocalCapacityError as error:
                case.update(status='infeasible', result=None, capacity_reason=_capacity_reason(error))
                print('%s %s: INFEASIBLE, requires %d B / capacity %d B' %
                      (operator, mapping, error.required_bytes, error.capacity_bytes), flush=True)
            else:
                _check_result(result, config)
                case.update(status='feasible', result=result, capacity_reason=None)
                print('%s %s: %d cycles' % (operator, mapping, result['total_cycles']), flush=True)
            cases.append(case)
    return cases


def _traits(cases):
    return [{key:case[key] for key in ('operator','mapping','shape','traits')} for case in cases]


def _derived(cases):
    rows = []
    index = {(case['operator'],case['mapping']):case for case in cases}
    for case in cases:
        shape, traits = case['shape'], case['traits']
        row = {key:case[key] for key in ('operator','mapping','architecture','status')}
        row.update(input_shape='%d x %d' % (shape['m'],shape['r']),
                   weight_shape='%d x %d' % (shape['r'],shape['n']),
                   output_shape='%d x %d' % (shape['m'],shape['n']),
                   physical_multipliers=traits['physical_multipliers'],
                   group_rows=traits['group_rows'], column_lanes=traits['column_lanes'],
                   reduction_lanes=traits['reduction_lanes'])
        if case['status'] == 'infeasible':
            reason = case['capacity_reason']
            row.update(total_cycles=None, required_bytes=reason['required_bytes'],
                       capacity_bytes=reason['capacity_bytes'], excess_bytes=reason['excess_bytes'],
                       infeasibility_code=reason['code'])
        else:
            result = case['result']
            resources = result['resources']
            registers = resources['register_payload_bytes']
            layout = result['physical_local_layout']
            for key in ('total_cycles','first_matrix_fragment','first_matrix_row','first_ln_row_done',
                        'last_internal_output_confirm','issued_updates','issue_lane_utilization',
                        'issue_duty_cycle','compute_utilization'):
                row[key] = result[key]
            final_matrix = max(result['row_matrix_cycles'])
            row.update(final_matrix_row=final_matrix, after_matrix_tail_cycles=result['total_cycles']-final_matrix,
                       required_bytes=layout['used_bytes'], capacity_bytes=layout['capacity_bytes'],
                       excess_bytes=0, free_bytes=layout['free_bytes'],
                       useful_weight_bytes=result['weight_port']['useful_weight_bytes'],
                       physical_weight_bytes=result['weight_port']['physical_weight_bytes'],
                       local_physical_read_bytes=result['local_memory']['physical_read_bytes'],
                       local_physical_write_bytes=result['local_memory']['physical_write_bytes'],
                       local_denied_cycles=result['local_memory']['denied_cycles'],
                       register_payload_bytes=resources['total_register_payload_bytes'],
                       weight_register_bytes=registers['weight_register_bytes'],
                       activation_register_bytes=registers['activation_register_bytes'],
                       accumulator_register_bytes=registers['accumulator_register_bytes'],
                       partial_sram_bytes=resources['b1_partial_sram_bytes'], area_status=resources['area']['status'])
            reference = index.get((case['operator'],'A'))
            if reference is not None and reference['status'] == 'feasible':
                row['same_condition_speedup_vs_A'] = reference['result']['total_cycles']/result['total_cycles']
        rows.append(row)
    return rows


def historical_regression(reference, reference_config):
    """Run the original sixteen complete dictionaries under their old profile."""
    mappings = tuple(reference['mappings'])
    if len(mappings) != 8 or len(reference['results']) != 16:
        raise ValueError('historical reference must contain the original eight mappings and sixteen results')
    expected = {(row['operator'],row['mapping']):row for row in reference['results']}
    if len(expected) != 16:
        raise ValueError('historical reference contains duplicate cases')
    actual, comparisons = [], []
    for operator in OPERATORS:
        for mapping in mappings:
            result, _ = simulate(reference_config, mapping, operator, capture_trace=False)
            original = expected[(operator,mapping)]
            same = result == original
            actual.append(result)
            comparisons.append({'operator':operator, 'mapping':mapping, 'exact_dict_equal':same,
                                'expected_sha256':_hash(original), 'actual_sha256':_hash(result),
                                'differences':[] if same else _differences(original,result)})
            print('historical64k/LN128 %s %s: %s' % (operator,mapping,'PASS' if same else 'FAIL'), flush=True)
    return {'passed':all(row['exact_dict_equal'] for row in comparisons),
            'profile':'historical_local64k_ln128', 'mixed_into_new_ranking':False,
            'reference_summary_sha256':_hash(reference), 'reference_config_sha256':config_hash(reference_config),
            'comparisons':comparisons, 'results':actual}


def _counts(cases):
    return {'total':len(cases), 'feasible':sum(case['status']=='feasible' for case in cases),
            'infeasible':sum(case['status']=='infeasible' for case in cases)}


def _tables(destination, cases, derived):
    _csv(destination/'cases.csv', derived)
    _csv(destination/'feasibility.csv', [{key:row.get(key) for key in
         ('operator','mapping','architecture','status','input_shape','weight_shape','output_shape',
          'physical_multipliers','required_bytes','capacity_bytes','excess_bytes','free_bytes','infeasibility_code')}
         for row in derived])
    _csv(destination/'operators.csv', [row for row in derived if row['status']=='feasible'])
    _csv(destination/'traits.csv', [_flat(row) for row in _traits(cases)])
    grouped = {key:[] for key in ('memory','resources','buffers','initialization','cycle_breakdown','capacity')}
    for case in cases:
        identity = {key:case[key] for key in ('operator','mapping','architecture','status')}
        if case['status']=='infeasible':
            grouped['capacity'].append(dict(identity,**_flat(case['capacity_reason'])))
            continue
        result = case['result']
        memory = dict(identity)
        for key in ('weight_port','local_memory','physical_local_layout'):
            memory.update(_flat(result[key],key))
        grouped['memory'].append(memory)
        for key in ('resources','buffers','initialization','cycle_breakdown'):
            grouped[key].append(dict(identity,**_flat(result[key])))
    for name, rows in grouped.items():
        _csv(destination/(name+'.csv'),rows)


def _analysis(cases, derived):
    counts = _counts(cases)
    lines = ['# 用户修订条件：32 KiB 局部存储与 LN64', '',
             '条件来源为用户后续修订及 hw_model/spec/user_revision_20261029_ln64_local32k.md；原始 300 MB/s 修改条款的 64 KiB / LN128 不代表本轮条件。',
             '本轮使用同一 150 MHz、300 MB/s 权重通路，4 个 4096×16-bit 单端口宏共 32 KiB，局部无冲突读写上限 8 B/cycle。LN 核心 L=64、II=64、最多一行，输入提取与输出确认另计。',
             '主窗口从第一次权重请求到最后内部 INT8 输出确认。激活、残差和参数预先就绪，初始化单列；不包括共享输出写回。',
             '候选不限制乘法器数，报告实际并行形状和寄存器。不同阵列为独立候选，未实现可重构硬件，形状切换与互连成本未综合。', '',
             '共 %d 个案例：%d 个可行，%d 个因当前驻留布局超容量而不可行。不可行项没有仿真完成周期，不以 0 填充或参与性能排名。' %
             (counts['total'],counts['feasible'],counts['infeasible']), '']
    for operator in OPERATORS:
        feasible = sorted((row for row in derived if row['operator']==operator and row['status']=='feasible'),key=lambda row:row['total_cycles'])
        infeasible = [row for row in derived if row['operator']==operator and row['status']=='infeasible']
        lines += ['## '+operator, '']
        if feasible:
            best = feasible[0]
            lines += ['本条件下最短完成周期为 %s：%d cycles。该候选有 %d 个乘法器；面积与功耗仍未知。' %
                      (best['mapping'],best['total_cycles'],best['physical_multipliers']), '',
                      '| 映射 | 输入/权重形状 | 乘法器 | 完成周期 | 首矩阵行 | 全矩阵行 | 矩阵后尾部 |',
                      '|---|---|---:|---:|---:|---:|---:|']
            for row in feasible:
                lines.append('| %s | %s / %s | %d | %d | %d | %d | %d |' %
                             (row['mapping'],row['input_shape'],row['weight_shape'],row['physical_multipliers'],
                              row['total_cycles'],row['first_matrix_row'],row['final_matrix_row'],row['after_matrix_tail_cycles']))
            lines.append('')
        else:
            lines += ['当前完整驻留布局下没有可行候选，不能给出此子图的性能排名。', '']
        if infeasible:
            lines += ['| 不可行映射 | 所需 B | 容量 B | 超额 B |', '|---|---:|---:|---:|']
            for row in infeasible:
                lines.append('| %s | %d | %d | %d |' % (row['mapping'],row['required_bytes'],row['capacity_bytes'],row['excess_bytes']))
            lines += ['', '逐张量预留、填充及部分和详见 capacity.csv 和 report.html。不可行结论针对当前完整驻留布局，未实现流式复用、缩小驻留集或共享 SRAM 外溢替代。', '']
    lines += ['## 历史回归与结论边界', '',
              '旧 64 KiB / LN128 的原 16 个完整结果字典在其原配置下独立重跑，见 historical_regression.json。它们仅验证历史兼容性，不与新 32 KiB / LN64 结果混排，不作同条件加速比。',
              '矩阵后尾部仅为最后矩阵行至最后内部输出的时间差；LN 与矩阵可重叠，不能将尾部当成全部 LN 忙碌时间。',
              '利用率分别报告有效 MAC/(物理乘法器×发射次数)、发射次数/总周期、有效 MAC/(物理乘法器×总周期)。',
              '这是已实现的元数据行为时序模型，不是 RTL、综合、功耗或真实 checkpoint 准确率验证。浮点 LN 参考独立，未声称位精确定点 LN。面积缺失保持 unknown。',
              '冻结重放逐项比较全部可行结果字典、不可行原因、traits、派生指标、配置与源码哈希，并独立重跑旧 16 个案例。', '']
    return '\n'.join(lines)


def _chart(feasible):
    if not feasible:
        return '<p class="muted">没有可用于周期图的可行案例。</p>'
    maximum = max(row['total_cycles'] for row in feasible)
    bars = []
    for index,row in enumerate(feasible):
        y, width = 18+index*28, 610*row['total_cycles']/maximum
        color = '#177a74' if row['architecture'].startswith('array') else '#647bb2'
        bars.append('<text x="180" y="%d" text-anchor="end">%s</text><rect x="195" y="%d" width="%.2f" height="18" rx="3" fill="%s"/><text x="%.2f" y="%d">%d</text>' %
                    (y+13,escape(row['mapping']),y,width,color,202+width,y+13,row['total_cycles']))
    return '<svg role="img" aria-label="feasible completion cycles under 32 KiB LN64" viewBox="0 0 940 %d">%s</svg>' % (len(feasible)*28+35,''.join(bars))


def _html(config,cases,derived,manifest,analysis):
    counts = _counts(cases)
    sections = []
    timing = [('mapping','映射'),('input_shape','输入 M×R'),('weight_shape','权重 R×N'),('physical_multipliers','乘法器'),
              ('total_cycles','完成周期'),('first_matrix_row','首矩阵行'),('final_matrix_row','最后矩阵行'),
              ('first_ln_row_done','首 LN 行'),('after_matrix_tail_cycles','矩阵后尾部')]
    capacity = [('mapping','映射'),('input_shape','输入'),('weight_shape','权重'),('required_bytes','所需 B'),
                ('capacity_bytes','容量 B'),('excess_bytes','超额 B'),('infeasibility_code','原因')]
    resources = [('mapping','映射'),('group_rows','token 组'),('column_lanes','输出 lanes'),('physical_multipliers','乘法器'),
                 ('register_payload_bytes','总寄存器 B'),('weight_register_bytes','权重寄存器 B'),
                 ('activation_register_bytes','激活寄存器 B'),('accumulator_register_bytes','累加器 B'),('partial_sram_bytes','部分和 SRAM B')]
    traffic = [('mapping','映射'),('physical_weight_bytes','权重物理 B'),('local_physical_read_bytes','局部物理读 B'),
               ('local_physical_write_bytes','局部物理写 B'),('local_denied_cycles','局部冲突周期'),
               ('issue_lane_utilization','发射有效比例'),('issue_duty_cycle','发射占空比'),('compute_utilization','端到端利用率')]
    for operator in OPERATORS:
        feasible = sorted((row for row in derived if row['operator']==operator and row['status']=='feasible'),key=lambda row:row['total_cycles'])
        failed = [row for row in derived if row['operator']==operator and row['status']=='infeasible']
        sections.append('<section><h2>%s：可行 %d / 不可行 %d</h2>' % (escape(operator),len(feasible),len(failed)))
        if feasible:
            sections += [_chart(feasible),_table(timing,feasible),'<h3>实际资源与并行形状</h3>',_table(resources,feasible),
                         '<h3>公共端口流量与利用率</h3>',_table(traffic,feasible)]
        else:
            sections.append('<p class="notice">当前完整驻留布局下没有可行候选；不提供虚假的完成周期或性能排名。</p>')
        if failed:
            sections += ['<h3>容量不可行诊断</h3>',_table(capacity,failed)]
        sections.append('</section>')
    details = []
    for case in cases:
        if case['status']=='feasible':
            result = case['result']
            data = {key:result[key] for key in ('cycle_breakdown','physical_local_layout','resources','buffers','initialization')}
        else:
            data = case['capacity_reason']
        details.append('<details><summary>%s · %s · %s</summary><pre>%s</pre></details>' %
                       (escape(case['operator']),escape(case['mapping']),escape(case['status']),escape(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True))))
    style = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#233944;font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif}
    main{max-width:1390px;margin:28px auto;padding:0 25px 45px}header{background:#173f4b;color:white;padding:28px 32px;border-radius:12px}h1{font-size:29px;line-height:1.4;margin:7px 0 12px}
    .eyebrow{font-size:12px;letter-spacing:2px;color:#a8d8d0}header p{margin:7px 0;color:#d2e7ea}section{background:white;border:1px solid #dce5e9;border-radius:10px;padding:24px 28px;margin:20px 0}
    h2{font-size:21px;margin:0 0 13px}h3{font-size:16px;margin:21px 0 9px}.notice{background:#fff7e7;border-left:4px solid #c99038;padding:14px 18px;margin:18px 0}.muted{font-size:13px;color:#627c88}
    .scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{white-space:nowrap;text-align:left;padding:10px 12px;border-bottom:1px solid #dce5e9}th{background:#f2f7f8}
    svg{width:100%;height:auto;display:block}svg text{font:12px system-ui,'Microsoft YaHei',sans-serif;fill:#607a86}details{border-top:1px solid #dce5e9;padding:12px 0}summary{cursor:pointer;color:#177a74;font-weight:600}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:16px;font:12px/1.65 Consolas,monospace}code{font:12px Consolas,monospace;overflow-wrap:anywhere}footer{font-size:12px;color:#627c88}
    @media(max-width:700px){main{padding:0 12px;margin:12px auto}header,section{padding:18px}h1{font-size:24px}}
    '''
    return ''.join(['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KWT 32 KiB / LN64 架构比较与容量诊断</title><style>',style,'</style></head><body><main>',
                    '<header><div class="eyebrow">USER REVISION · LOCAL 32 KiB · LN64 · FIXED 300 MB/s</div><h1>全部候选的可行性能与驻留容量诊断</h1><p>%d 个案例：%d 个可行 / %d 个容量不可行；不限制乘法器数</p><p>用户 profile：%s</p></header>' % (counts['total'],counts['feasible'],counts['infeasible'],escape(REVISED_PROFILE_ID)),
                    '<div class="notice"><strong>本轮条件来自用户后续修订：</strong>4×4096×16-bit 单端口宏＝32 KiB，局部读写合计峰值 8 B/cycle；LN 核心 L=64、II=64、最多一行，I/O 另计。权重通路仍为 300 MB/s。旧 64 KiB / LN128 报告保持冻结，仅在原条件下独立回归，不与新结果混排。</div>',
                    '<section><h2>测量与布局边界</h2><p>第一次权重请求至最后内部 INT8 输出确认。输入、残差和参数预先就绪，初始化单列；没有共享输出写回。保留完整输入、残差、LN 前 INT16 区和输出的当前驻留布局。容量失败记录结构化原因，继续其他候选，不默认外溢共享 SRAM 或悄悄改成流式布局。</p><p class="muted">阵列形状取自核心 traits，候选按实际乘法器和寄存器列账。不同形状是独立硬件候选，未实现可重构互连，也未完成 RTL、综合、功耗或位精确定点 LN 验证。</p></section>',
                    ''.join(sections),'<section><h2>分析与解释</h2><pre>',escape(analysis),'</pre></section>',
                    '<section><h2>逐案例 tensor 分配、容量原因和阶段账本</h2>',''.join(details),'</section>',
                    '<footer><p>配置 SHA-256：<code>',escape(config_hash(config)),'</code><br>源码集合 SHA-256：<code>',escape(manifest['aggregate_sha256']),'</code><br>用户修订说明：<code>',escape(REVISION_SPEC),'</code></p><p>summary.json 保存所有案例，包括不可行原因。cases/feasibility/capacity.csv 列全部状态；operators 等 CSV 只列可行结果。historical_regression.json 单列旧 16 项回归。verification.json 记录可行、不可行及历史结果的冻结重放检查。</p></footer></main></body></html>'])


def build_study(destination):
    config = load_config(REVISED_CONFIG)
    if config['provenance']['scenario_id'] != REVISED_PROFILE_ID:
        raise ValueError('wrong user-revision profile')
    if len(STUDY_MAPPINGS)!=20:
        raise ValueError('the complete revised candidate registry must contain twenty mappings')
    reference = _load(REFERENCE_REPORT/'summary.json')
    old_config = load_config(REFERENCE_REPORT/'run_config.json')
    destination = Path(destination).absolute()
    destination.mkdir(parents=True,exist_ok=False)
    manifest = source_manifest(ROOT)
    _freeze_sources(destination,manifest)
    reference_dir = destination/'references'
    reference_dir.mkdir()
    dump(reference_dir/'historical_summary.json',reference)
    dump(reference_dir/'historical_run_config.json',old_config)
    dump(destination/'run_config.json',config)
    dump(destination/'source_manifest.json',manifest)
    dump(destination/'invocation.json',{'argv':sys.argv,'python':sys.version,'executable':sys.executable,
                                      'entrypoint':'hw_model.architecture_study_32k_ln64','profile_id':REVISED_PROFILE_ID,
                                      'client_timezone':'Asia/Shanghai','trace_enabled':False,
                                      'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat()})
    cases = run_cases(config)
    derived = _derived(cases)
    historical = historical_regression(reference,old_config)
    dump(destination/'historical_regression.json',historical)
    if not historical['passed']:
        dump(destination/'verification.json',{'status':'failed','historical_complete_dicts_identical':False})
        raise AssertionError('historical sixteen complete result dictionaries differ')
    counts = _counts(cases)
    if counts!={'total':40,'feasible':20,'infeasible':20}:
        raise AssertionError('unexpected full-layout feasibility counts: %r' % counts)
    if source_manifest(ROOT)!=manifest:
        raise ValueError('sources changed during study; refusing an inconsistent source bundle')
    summary = {'schema_version':1,'study':STUDY_ID,'profile_id':REVISED_PROFILE_ID,
               'measurement_boundary':BOUNDARY,'config_sha256':config_hash(config),'source_manifest':manifest,
               'authority':{'revision_spec':REVISION_SPEC,'revision_spec_sha256':manifest['files'].get(REVISION_SPEC)},
               'historical_reference_summary_sha256':_hash(reference),'historical_reference_config_sha256':config_hash(old_config),
               'mappings':list(STUDY_MAPPINGS),'case_counts':counts,'cases':cases,'mapping_traits':_traits(cases),'derived_metrics':derived,
               'claims':{'metadata_behavior_simulation':True,'capacity_infeasibility_is_layout_specific':True,
                         'historical_results_ranked_with_new_conditions':False,'multiplier_count_capped':False,
                         'rtl_verified':False,'area_synthesized':False,'power_measured':False,
                         'fixed_point_ln_bit_accuracy_verified':False,'checkpoint_accuracy_measured':False}}
    dump(destination/'summary.json',summary)
    _tables(destination,cases,derived)
    analysis = _analysis(cases,derived)
    (destination/'analysis.md').write_text(analysis,encoding='utf-8')
    (destination/'report.html').write_text(_html(config,cases,derived,manifest,analysis),encoding='utf-8')
    dump(destination/'verification.json',{'status':'simulation_and_historical_pass_replay_pending','case_counts':counts,
                                         'historical_complete_dicts_identical':True,'historical_case_count':16,
                                         'source_bundle_consistent':True,'exact_replay':False})
    print('32 KiB / LN64 study: '+str(destination/'report.html'),flush=True)
    return 0


def replay_study(report):
    report = Path(report).absolute()
    saved = _load(report/'summary.json')
    if saved.get('study')!=STUDY_ID:
        raise ValueError('not a 32 KiB / LN64 study report')
    config = load_config(report/'run_config.json')
    if config_hash(config)!=saved['config_sha256']:
        raise ValueError('saved revision config hash differs')
    manifest = source_manifest(ROOT)
    if manifest!=saved['source_manifest']:
        raise ValueError('source/runtime differs; replay from frozen source/ with the same Python')
    reference = _load(report/'references'/'historical_summary.json')
    old_config = load_config(report/'references'/'historical_run_config.json')
    if _hash(reference)!=saved['historical_reference_summary_sha256'] or config_hash(old_config)!=saved['historical_reference_config_sha256']:
        raise ValueError('historical reference hashes differ')
    cases = run_cases(config,saved['mappings'])
    historical = historical_regression(reference,old_config)
    original_historical = _load(report/'historical_regression.json')
    checks = {'case_envelopes_exact':cases==saved['cases'],'traits_exact':_traits(cases)==saved['mapping_traits'],
              'derived_metrics_exact':_derived(cases)==saved['derived_metrics'],
              'case_counts_exact':_counts(cases)==saved['case_counts'],
              'historical_complete_dicts_identical':historical['passed'],
              'historical_replay_exact':historical==original_historical,'source_bundle_consistent':True}
    passed = all(checks.values())
    dump(report/'verification.json',dict(checks,status='pass' if passed else 'failed',exact_replay=passed,
                                          case_counts=_counts(cases),historical_case_count=16,python_version=manifest['python_version']))
    if not passed:
        raise AssertionError('revision exact replay differs in feasible results, capacity reasons, traits or historical regression')
    print('Exact 32 KiB / LN64 replay PASS: 40 case envelopes and 16 historical complete result dictionaries',flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='User-revised 32 KiB / LN64 study with per-case capacity diagnoses')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out',help='new immutable report directory')
    group.add_argument('--replay',help='existing revision report directory')
    args = parser.parse_args(argv)
    try:
        return replay_study(args.replay) if args.replay else build_study(args.out)
    except (ValueError,AssertionError,FileExistsError,FileNotFoundError) as error:
        parser.exit(2,'error: %s\n' % error)


if __name__=='__main__':
    raise SystemExit(main())
