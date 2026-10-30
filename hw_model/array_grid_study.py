"""Explicit P=20..33, Q=4/6/8 grid under LN64 / 64 KiB / 300 MB/s.

The previous default mapping registry is unchanged. Each spawned worker runs
one independent operator/mapping. Results are always returned in task order,
so worker count cannot change the saved experiment dictionaries.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
import json
import multiprocessing
from pathlib import Path
import sys

from .architecture_study import _hash, _load, _table, _differences
from .architecture_study_32k_ln64 import _traits, _tables
from .architecture_study_ln64 import _validate_profile, _augment_derived
from .compute import Shape
from .config import LN64_LOCAL64K_CONFIG, LN64_LOCAL64K_PROFILE_ID, load_config, source_manifest
from .experiments import ROOT, _freeze_sources, config_hash, dump
from .fixed_schedule import GRID_MAPPINGS, ALL_MAPPINGS, mapping_traits
from .fixed_simulation import simulate
from .reporting import _csv
from .reference_data import reference_report_path


STUDY_ID = 'array_grid_p20_33_q4_6_8_ln64_local64k'
ROWS = tuple(range(20,34))
COLUMNS = (4,6,8)
OPERATORS = ('projection','ffn2')
TARGET = 'A26x8'
REFERENCE_REPORT = reference_report_path('architecture_study_ln64_20261029_v1')
REVISION_SPEC = 'hw_model/spec/user_revision_20261029_array_grid.md'
PROFILE_PATH = 'hw_model/configs/ln64_local64k.json'
OVERLAP_MAPPINGS = ('A32x4','A32x8','A33x4','A33x8')


def expected_weight_ledger(shape, traits):
    """Independent absolute-byte beat count, including every unaligned packet."""
    p,q = traits['group_rows'],traits['column_lanes']
    groups = (shape.m+p-1)//p
    useful,physical,packets = 0,0,0
    for output_group,start in enumerate(range(0,shape.n,q)):
        width = min(q,shape.n-start)
        for k in range(shape.r):
            address = (output_group*shape.r+k)*q
            useful += width
            physical += 4*((address%4+width+3)//4)
            packets += 1
    return {'policy':'compact_slot_Q_no_cross_packet_coalescing','slot_bytes':q,
            'token_groups':groups,'last_token_group_rows':shape.m-(groups-1)*p,
            'output_groups':(shape.n+q-1)//q,'last_output_group_columns':shape.n%q or q,
            'per_group_useful_bytes':useful,'per_group_physical_bytes':physical,
            'useful_weight_bytes':useful*groups,'physical_weight_bytes':physical*groups,
            'packet_count':packets*groups}


def _run_one(task):
    """Top-level spawn-safe worker; no output, clocks, or source mutation."""
    index,config,operator,mapping = task
    model = config['model']
    shape = Shape(model['tokens'],model['hidden'],model['hidden'] if operator=='projection' else model['ffn'])
    traits = mapping_traits(shape,mapping)
    result,_ = simulate(config,mapping,operator,capture_trace=False)
    ledger = expected_weight_ledger(shape,traits)
    if result['weight_port']['useful_weight_bytes']!=ledger['useful_weight_bytes'] or result['weight_port']['physical_weight_bytes']!=ledger['physical_weight_bytes']:
        raise AssertionError('absolute-address weight beat ledger differs for '+mapping)
    if result['shared_activation_read_bytes'] or result['shared_activation_write_bytes']:
        raise AssertionError('local activation-residency contract violated')
    if result['local_memory']['capacity_bytes']!=65536 or result['local_memory']['banks']!=8:
        raise AssertionError('this grid retains eight banks / 64 KiB')
    if sum(result['cycle_breakdown'].values())!=result['total_cycles']:
        raise AssertionError('cycle partition mismatch')
    case = {'operator':operator,'mapping':mapping,'architecture':traits['architecture'],
            'shape':{'m':shape.m,'n':shape.n,'r':shape.r},'traits':traits,
            'status':'feasible','capacity_reason':None,'weight_ledger':ledger,'result':result}
    return index,case


def run_cases(config, mappings=None, workers=4):
    """Deterministic task order for both sequential and Windows-spawn workers."""
    if type(workers) is not int or workers<1:
        raise ValueError('workers must be a positive integer')
    mappings = tuple(GRID_MAPPINGS if mappings is None else mappings)
    tasks = [(index,config,operator,mapping) for index,(operator,mapping) in
             enumerate((operator,mapping) for operator in OPERATORS for mapping in mappings)]
    results = [None]*len(tasks)
    if workers==1:
        for task in tasks:
            index,case = _run_one(task)
            results[index] = case
            print('%d/%d %s %s: %d cycles' % (index+1,len(tasks),case['operator'],case['mapping'],case['result']['total_cycles']),flush=True)
    else:
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
            futures = [pool.submit(_run_one,task) for task in tasks]
            for completed,future in enumerate(as_completed(futures),1):
                index,case = future.result()
                results[index] = case
                print('%d/%d %s %s: %d cycles' % (completed,len(tasks),case['operator'],case['mapping'],case['result']['total_cycles']),flush=True)
    return results


def derive_grid(cases):
    rows = _augment_derived(cases)
    for row,case in zip(rows,cases):
        traits,ledger,shape = case['traits'],case['weight_ledger'],case['shape']
        row.update(P=traits['group_rows'],Q=traits['column_lanes'],
                   first_internal_output_confirm=case['result']['first_row_cycles'],
                   token_groups=ledger['token_groups'],last_token_group_rows=ledger['last_token_group_rows'],
                   output_groups=ledger['output_groups'],last_output_group_columns=ledger['last_output_group_columns'],
                   weight_slot_bytes=ledger['slot_bytes'],weight_amplification=ledger['physical_weight_bytes']/ledger['useful_weight_bytes'],
                   per_group_useful_weight_bytes=ledger['per_group_useful_bytes'],
                   per_group_physical_weight_bytes=ledger['per_group_physical_bytes'],
                   channel_tail_lanes=traits['column_lanes']-ledger['last_output_group_columns'],
                   valid_macs=shape['m']*shape['n']*shape['r'])
    return rows


def aggregate_shapes(derived):
    indexed = {(row['operator'],row['mapping']):row for row in derived}
    aggregates = []
    for mapping in GRID_MAPPINGS:
        if ('projection',mapping) not in indexed or ('ffn2',mapping) not in indexed:
            continue
        projection,ffn = indexed[('projection',mapping)],indexed[('ffn2',mapping)]
        aggregates.append({'operator':'projection_plus_ffn2_once','mapping':mapping,'architecture':projection['architecture'],
                           'P':projection['P'],'Q':projection['Q'],'physical_multipliers':projection['physical_multipliers'],
                           'projection_cycles':projection['total_cycles'],'ffn2_cycles':ffn['total_cycles'],
                           'total_cycles':projection['total_cycles']+ffn['total_cycles'],
                           'physical_weight_bytes':projection['physical_weight_bytes']+ffn['physical_weight_bytes'],
                           'local_physical_total_bytes':projection['local_physical_total_bytes']+ffn['local_physical_total_bytes'],
                           'max_register_payload_bytes':max(projection['register_payload_bytes'],ffn['register_payload_bytes']),
                           'not_full_network_latency':True})
    return aggregates


def pareto(rows):
    records = []
    for operator in sorted(set(row['operator'] for row in rows)):
        group = [row for row in rows if row['operator']==operator]
        for row in group:
            dominating = [other['mapping'] for other in group
                          if other['total_cycles']<=row['total_cycles'] and other['physical_multipliers']<=row['physical_multipliers']
                          and (other['total_cycles']<row['total_cycles'] or other['physical_multipliers']<row['physical_multipliers'])]
            records.append({'operator':operator,'mapping':row['mapping'],'architecture':row['architecture'],
                            'total_cycles':row['total_cycles'],'physical_multipliers':row['physical_multipliers'],
                            'nondominated':not dominating,'dominated_by':sorted(dominating)})
    return records


def target_comparisons(derived, aggregates):
    comparisons = []
    for operator in OPERATORS+('projection_plus_ffn2_once',):
        rows = [row for row in derived+aggregates if row['operator']==operator]
        target = next((row for row in rows if row['mapping']==TARGET),None)
        if target is None:
            continue
        best = min(rows,key=lambda row:(row['total_cycles'],row['physical_multipliers'],row['mapping']))
        comparisons.append({'operator':operator,'target':TARGET,'target_cycles':target['total_cycles'],
                            'best_mapping':best['mapping'],'best_cycles':best['total_cycles'],
                            'cycle_gap':target['total_cycles']-best['total_cycles'],
                            'target_over_best':target['total_cycles']/best['total_cycles'],
                            'cycle_rank':1+sum(row['total_cycles']<target['total_cycles'] for row in rows),
                            'target_multipliers':target['physical_multipliers']})
    return comparisons


def compare_overlap(cases, reference):
    expected = {(row['operator'],row['mapping']):row for row in reference['results']}
    actual = {(case['operator'],case['mapping']):case['result'] for case in cases}
    overlap = set(reference['mappings']) & set(GRID_MAPPINGS)
    if overlap!=set(OVERLAP_MAPPINGS):
        raise ValueError('unexpected previous/grid mapping intersection')
    comparisons = []
    for operator in OPERATORS:
        for mapping in OVERLAP_MAPPINGS:
            left,right = expected[(operator,mapping)],actual[(operator,mapping)]
            same = left==right
            comparisons.append({'operator':operator,'mapping':mapping,'exact_dict_equal':same,
                                'reference_sha256':_hash(left),'actual_sha256':_hash(right),
                                'differences':[] if same else _differences(left,right)})
    return {'passed':all(row['exact_dict_equal'] for row in comparisons),
            'case_count':8,'included_in_grid_case_count':True,'additional_simulation_count':0,
            'comparison':'complete result dictionaries under the same LN64 / 64 KiB condition',
            'reference_summary_sha256':_hash(reference),'comparisons':comparisons}


def _heatmap(derived, operator):
    values = {(row['P'],row['Q']):row['total_cycles'] for row in derived if row['operator']==operator}
    minimum,maximum = min(values.values()),max(values.values())
    items = ['<text x="24" y="30">%s · cycles</text>' % escape(operator)]
    for column,q in enumerate(COLUMNS):
        items.append('<text x="%d" y="58" text-anchor="middle">Q=%d</text>' % (185+column*150,q))
    for index,p in enumerate(ROWS):
        y = 72+index*35
        items.append('<text x="90" y="%d" text-anchor="end">P=%d</text>' % (y+21,p))
        for column,q in enumerate(COLUMNS):
            value = values[(p,q)]
            ratio = (value-minimum)/max(1,maximum-minimum)
            light = 48+ratio*46
            selected = p==26 and q==8
            items.append('<rect x="%d" y="%d" width="140" height="29" rx="3" fill="hsl(166,45%%,%.2f%%)" stroke="%s" stroke-width="%s"><title>A%dx%d: %d cycles; %d multipliers</title></rect>' %
                         (115+column*150,y,light,'#cd812b' if selected else '#e0e9e8','3' if selected else '1',p,q,value,p*q))
            items.append('<text x="%d" y="%d" text-anchor="middle" fill="%s">%d%s</text>' %
                         (185+column*150,y+20,'white' if light<65 else '#213e43',value,' *' if value==minimum else ''))
    return '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="%s 14 by 3 cycle heatmap" viewBox="0 0 590 580" style="font:13px system-ui,sans-serif">%s</svg>' % (escape(operator,quote=True),''.join(items))


def _analysis(derived, aggregates, comparisons, pareto_rows):
    lines = ['# P=20..33、Q=4/6/8 阵列网格研究','',
             '只显式选择 GRID_MAPPINGS 的 42 个阵列形状，两个子图共 84 个案例；旧 ALL_MAPPINGS 的 28 个默认候选集合保持不变。','',
             '共同条件：150 MHz、300 MB/s 权重接口、八个 4096×16-bit 单端口宏共 64 KiB、LN 核心 L64/II64/单在途。LN 输入提取和输出确认另计；局部读写共用 bank，峰值合计 16 B/cycle。',
             '主窗口为第一次权重请求至最后内部 INT8 输出确认；输入、残差和参数初装单列，不包含共享输出写回。','',
             '## 26×8 与当前网格最佳','',
             '| 范围 | 26×8 周期 | 最佳形状 | 最佳周期 | 差值 | 26×8/最佳 | 周期排名 |',
             '|---|---:|---|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append('| %s | %d | %s | %d | %d | %.6g | %d |' %
                     (row['operator'],row['target_cycles'],row['best_mapping'],row['best_cycles'],row['cycle_gap'],row['target_over_best'],row['cycle_rank']))
    lines += ['', 'projection_plus_ffn2_once 只是一次投影加一次 FFN2 的周期之和，不是完整 KWT、不是 12 层推理，也不包含两个独立窗口间的系统调度成本。','',
              '## 非整除通道与真实总线拍', '',
              '权重布局使用 compact slot=Q。每个包按绝对字节地址独立经过 4 B 总线拍；不把 slot 向 4 B 对齐，不默认相邻包拼拍、缓存已读拍或免费 coalescing。',
              'Q4 与 Q8 整除 64；Q6 有 10 个满六通道组和一个四通道尾组。六字节满包占两个四字节拍；尾四字节包因 stride=6 交替落在 mod4=0/2，分别占一拍/两拍。R 为偶数时，每 token 组真实物理流量为 86R B，而有效权重为 64R B。','',
              '| 子图 | Q | 一组有效权重 B | 一组物理权重 B | 物理/有效 | 输出组数 | 尾组通道 |']
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for operator in OPERATORS:
        for q in COLUMNS:
            row = next(item for item in derived if item['operator']==operator and item['P']==20 and item['Q']==q)
            lines.append('| %s | %d | %d | %d | %.6g | %d | %d |' %
                         (operator,q,row['per_group_useful_weight_bytes'],row['per_group_physical_weight_bytes'],row['weight_amplification'],row['output_groups'],row['last_output_group_columns']))
    lines += ['', 'token 组数、token 尾组、通道尾组和物理拍浪费分别计账。更多乘法器不保证更短完成周期；寄存器、输入提取、排空、后处理与 LN 排队均可能改变收益。','',
              '## 周期 / 乘法器 Pareto', '']
    for operator in OPERATORS+('projection_plus_ffn2_once',):
        names = [row['mapping'] for row in pareto_rows if row['operator']==operator and row['nondominated']]
        lines.append('- %s 非支配候选：%s。' % (operator,', '.join(names)))
    lines += ['', 'Pareto 同时最小化已模拟周期与实际乘法器数；同坐标候选均保留。这不是面积、功耗或能量估计，也不证明连续设计空间最优。','',
              '## 验证边界', '',
              '前次 LN64/64 KiB 报告中的 A32x4、A32x8、A33x4、A33x8 共八个完整结果字典与本次网格结果逐字段比较。它们已包含在新跑的 84 个案例中，不额外重复仿真，也不把前次其他候选混入网格排名。',
              '前次 LN128 的原 16 项历史通过证据只复制并校验哈希，本轮未重新运行那 16 项。',
              '完成周期来自元数据行为时序模型；数值 LN 参考独立。未声称 RTL、位精确定点 LN、真实 checkpoint 准确率、可达频率、综合面积或功耗通过。不同阵列是独立候选，未实现可重构切换及互连。',
              '发射有效比例分母为乘法器×发射次数；发射占空比分母为总周期；端到端 MAC 利用率分母为乘法器×总周期。矩阵后尾部仅为最后矩阵行至最终输出确认的间隔，不等于全部 LN 忙碌时间。','']
    return '\n'.join(lines)


def _html(derived, aggregates, comparisons, pareto_rows, manifest, analysis):
    timing = [('mapping','形状'),('physical_multipliers','乘法器'),('total_cycles','完成周期'),('first_matrix_row','首矩阵行'),
              ('final_matrix_row','最后矩阵行'),('first_ln_row_done','首 LN 核心完成'),
              ('first_internal_output_confirm','首行内部输出确认'),('after_matrix_tail_cycles','后处理/LN尾部')]
    metrics = [('mapping','形状'),('register_payload_bytes','总寄存器 B'),('token_groups','token 组数'),('last_token_group_rows','token 尾组'),
               ('last_output_group_columns','通道尾组'),('physical_weight_bytes','权重物理 B'),('weight_amplification','权重放大'),
               ('local_physical_total_bytes','局部读写 B'),('issue_lane_utilization','有效 lane'),('issue_duty_cycle','发射占空比'),('compute_utilization','端到端利用率')]
    sections = []
    for operator in OPERATORS:
        rows = sorted((row for row in derived if row['operator']==operator),key=lambda row:(row['total_cycles'],row['physical_multipliers'],row['mapping']))
        same_p = []
        indexed = {(row['P'],row['Q']):row for row in rows}
        for p in ROWS:
            choices = [indexed[(p,q)] for q in COLUMNS]
            best = min(choices,key=lambda row:row['total_cycles'])
            same_p.append({'P':p,'Q4':indexed[(p,4)]['total_cycles'],'Q6':indexed[(p,6)]['total_cycles'],
                           'Q8':indexed[(p,8)]['total_cycles'],'best_Q':best['Q']})
        sections += ['<section><h2>%s：14×3 周期热图</h2>' % escape(operator),_heatmap(derived,operator),
                     '<p class="muted">每个单元为实际完成周期，* 为子图最短周期；橙色边框突出 26×8。颜色只在同一子图内比较。</p>',
                     '<h3>同 P 下的 Q4 / Q6 / Q8</h3>',_table([('P','P'),('Q4','Q4 周期'),('Q6','Q6 周期'),('Q8','Q8 周期'),('best_Q','最短周期 Q')],same_p),
                     '<h3>全部 42 形状：按本子图周期排序</h3>',_table(timing,rows),'<h3>资源、尾组、流量与利用率</h3>',_table(metrics,rows),'</section>']
    aggregate_sorted = sorted(aggregates,key=lambda row:(row['total_cycles'],row['physical_multipliers'],row['mapping']))
    fronts = [{'operator':operator,'mappings':', '.join(row['mapping'] for row in pareto_rows if row['operator']==operator and row['nondominated'])} for operator in OPERATORS+('projection_plus_ffn2_once',)]
    style = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#233944;font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif}main{max-width:1420px;margin:28px auto;padding:0 25px 45px}
    header{background:#173f4b;color:white;padding:28px 32px;border-radius:12px}h1{font-size:29px;line-height:1.4;margin:7px 0 12px}.eyebrow{font-size:12px;letter-spacing:2px;color:#a8d8d0}header p{margin:7px 0;color:#d2e7ea}
    section{background:white;border:1px solid #dce5e9;border-radius:10px;padding:24px 28px;margin:20px 0}h2{font-size:21px;margin:0 0 13px}h3{font-size:16px;margin:21px 0 9px}.notice{background:#fff7e7;border-left:4px solid #c99038;padding:14px 18px;margin:18px 0}.muted{font-size:13px;color:#627c88}
    .scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{white-space:nowrap;text-align:left;padding:10px 12px;border-bottom:1px solid #dce5e9}th{background:#f2f7f8}svg{width:100%;max-width:740px;height:auto;display:block;margin:auto}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:16px;font:12px/1.65 Consolas,monospace}code{font:12px Consolas,monospace;overflow-wrap:anywhere}footer{font-size:12px;color:#627c88}@media(max-width:700px){main{padding:0 12px;margin:12px auto}header,section{padding:18px}h1{font-size:24px}}
    '''
    return ''.join(['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KWT P20–33 × Q4/6/8 阵列网格</title><style>',style,'</style></head><body><main>',
                    '<header><div class="eyebrow">42 SHAPES · 84 CASES · LN64 · LOCAL 64 KiB · 300 MB/s</div><h1>P=20–33，Q=4 / 6 / 8 的完整阵列网格</h1><p>显式 GRID_MAPPINGS；原 28 个默认候选不变。26×8 仅作突出比较，不预设最佳。</p></header>',
                    '<div class="notice">共同条件：150 MHz、300 MB/s 权重接口、八个单端口宏共 64 KiB、LN 核心 L64/II64/单在途。输入提取与输出确认另计。主窗口截至最后内部 INT8 输出确认，初始化单列，不包含共享输出写回。</div>',
                    '<section><h2>26×8 与各子图 / 合计最佳</h2>',_table([('operator','范围'),('target_cycles','26×8 周期'),('best_mapping','最佳形状'),('best_cycles','最佳周期'),('cycle_gap','差值'),('target_over_best','26×8/最佳'),('cycle_rank','周期排名')],comparisons),
                    '<p class="muted">projection_plus_ffn2_once 只是一次投影加一次 FFN2 的合计，不是整网、不是 12 层推理，也未计两个独立窗口之间的系统衔接。</p></section>',
                    ''.join(sections),'<section><h2>42 个形状汇总：一次投影 + 一次 FFN2</h2>',_table([('mapping','形状'),('physical_multipliers','乘法器'),('projection_cycles','投影周期'),('ffn2_cycles','FFN2 周期'),('total_cycles','二子图各一次合计'),('max_register_payload_bytes','最大寄存器 B')],aggregate_sorted),
                    '<h3>周期 / 乘法器非支配候选</h3>',_table([('operator','范围'),('mappings','非支配形状')],fronts),'<p class="muted">Pareto 只比较周期与乘法器计数，不是面积或功耗优化结论。</p></section>',
                    '<section><h2>总线跨拍、尾通道与验证边界</h2><pre>',escape(analysis),'</pre></section>',
                    '<footer><p>源码集合 SHA-256：<code>',escape(manifest['aggregate_sha256']),'</code></p><p>summary.json 保留84完整案例及派生指标；all_cases.csv / shape_summary.csv / heatmap_*.csv 与 SVG 提供全部数表；memory/resources/buffers/cycle_breakdown.csv保留详细账本。intersection_regression.json 校验8个重叠结果。source/ 与 references/ 保存精确重放所需来源，verification.json 记录校验结果。</p></footer></main></body></html>'])


def _validate_grid():
    expected = tuple('A%dx%d' % (p,q) for p in ROWS for q in COLUMNS)
    if tuple(GRID_MAPPINGS)!=expected or len(GRID_MAPPINGS)!=42:
        raise ValueError('GRID_MAPPINGS must be exactly P20..33 x Q4/6/8 in row-major order')


def build_study(destination, workers=4):
    _validate_grid()
    config = load_config(LN64_LOCAL64K_CONFIG)
    _validate_profile(config)
    reference = _load(REFERENCE_REPORT/'summary.json')
    previous_config = load_config(REFERENCE_REPORT/'run_config.json')
    if config!=previous_config or reference['mappings']!=list(ALL_MAPPINGS) or len(reference['results'])!=56:
        raise ValueError('previous reference must be the same-condition 28-mapping / 56-result LN64 study')
    evidence = _load(REFERENCE_REPORT/'historical_regression.json')
    if not evidence.get('passed') or len(evidence['results'])!=16:
        raise ValueError('previous LN128 historical evidence did not pass')
    destination = Path(destination).absolute()
    destination.mkdir(parents=True,exist_ok=False)
    manifest = source_manifest(ROOT)
    if REVISION_SPEC not in manifest['files']:
        raise ValueError('grid user-revision specification is missing')
    _freeze_sources(destination,manifest)
    references = destination/'references'
    references.mkdir()
    dump(references/'previous_summary.json',reference)
    dump(references/'previous_run_config.json',previous_config)
    dump(references/'previous_ln128_evidence.json',evidence)
    dump(references/'profile_definition.json',_load(LN64_LOCAL64K_CONFIG))
    (references/'user_revision.md').write_bytes((ROOT/REVISION_SPEC).read_bytes())
    dump(destination/'run_config.json',config)
    dump(destination/'source_manifest.json',manifest)
    dump(destination/'invocation.json',{'argv':sys.argv,'python':sys.version,'executable':sys.executable,'workers':workers,
                                      'entrypoint':'hw_model.array_grid_study','client_timezone':'Asia/Shanghai','trace_enabled':False,
                                      'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat()})
    cases = run_cases(config,workers=workers)
    if len(cases)!=84:
        raise AssertionError('grid requires exactly 84 complete cases')
    regression = compare_overlap(cases,reference)
    dump(destination/'intersection_regression.json',regression)
    if not regression['passed']:
        dump(destination/'verification.json',{'status':'failed','intersection_exact_dicts':False})
        raise AssertionError('eight previous/grid complete result dictionaries differ')
    if source_manifest(ROOT)!=manifest:
        raise ValueError('sources changed during grid run; refusing inconsistent source freeze')
    derived = derive_grid(cases)
    aggregates = aggregate_shapes(derived)
    pareto_rows = pareto(derived+aggregates)
    comparisons = target_comparisons(derived,aggregates)
    summary = {'schema_version':1,'study':STUDY_ID,'config_sha256':config_hash(config),'source_manifest':manifest,
               'profile_id':LN64_LOCAL64K_PROFILE_ID,'grid_rows':list(ROWS),'grid_columns':list(COLUMNS),
               'mappings':list(GRID_MAPPINGS),'case_count':84,'cases':cases,'results':[case['result'] for case in cases],
               'mapping_traits':_traits(cases),'derived_metrics':derived,'shape_summary':aggregates,
               'pareto':pareto_rows,'target_comparisons':comparisons,
               'references':{'previous_summary_sha256':_hash(reference),'previous_config_sha256':config_hash(previous_config),
                             'previous_ln128_evidence_sha256':_hash(evidence),
                             'profile_definition_sha256':_hash(_load(LN64_LOCAL64K_CONFIG)),
                             'revision_spec':REVISION_SPEC,'revision_spec_sha256':manifest['files'][REVISION_SPEC]},
               'claims':{'weight_slot_policy':'compact_slot_Q_no_cross_packet_coalescing','metadata_behavior_simulation':True,
                         'previous_default_registry_unchanged':True,'aggregate_is_full_network':False,
                         'ln128_historical_cases_rerun':False,'rtl_verified':False,'area_synthesized':False,
                         'power_measured':False,'fixed_point_ln_bit_accuracy_verified':False}}
    dump(destination/'summary.json',summary)
    _tables(destination,cases,derived)
    _csv(destination/'all_cases.csv',derived)
    _csv(destination/'shape_summary.csv',aggregates)
    _csv(destination/'pareto.csv',pareto_rows)
    _csv(destination/'target_26x8.csv',comparisons)
    for operator in OPERATORS:
        values = {(row['P'],row['Q']):row['total_cycles'] for row in derived if row['operator']==operator}
        _csv(destination/('heatmap_'+operator+'.csv'),[{'P':p,**{'Q%d'%q:values[(p,q)] for q in COLUMNS}} for p in ROWS])
        (destination/('heatmap_'+operator+'.svg')).write_text(_heatmap(derived,operator),encoding='utf-8')
    analysis = _analysis(derived,aggregates,comparisons,pareto_rows)
    (destination/'analysis.md').write_text(analysis,encoding='utf-8')
    (destination/'report.html').write_text(_html(derived,aggregates,comparisons,pareto_rows,manifest,analysis),encoding='utf-8')
    dump(destination/'verification.json',{'status':'simulation_and_intersection_pass_replay_pending','case_count':84,
                                         'intersection_exact_dicts':True,'intersection_case_count':8,
                                         'ln128_historical_evidence_hash_verified':True,'exact_replay':False})
    print('Array grid study: '+str(destination/'report.html'),flush=True)
    return 0


def replay_study(report, workers=4):
    _validate_grid()
    report = Path(report).absolute()
    saved = _load(report/'summary.json')
    if saved.get('study')!=STUDY_ID or saved['mappings']!=list(GRID_MAPPINGS) or saved['case_count']!=84:
        raise ValueError('not the saved 42-shape / 84-case compact-slot grid')
    config = load_config(report/'run_config.json')
    _validate_profile(config)
    if config_hash(config)!=saved['config_sha256']:
        raise ValueError('saved config hash differs')
    manifest = source_manifest(ROOT)
    if manifest!=saved['source_manifest'] or _load(report/'source_manifest.json')!=manifest:
        raise ValueError('source/runtime differs; replay in frozen source/ using the recorded Python')
    references = saved['references']
    reference = _load(report/'references'/'previous_summary.json')
    previous_config = load_config(report/'references'/'previous_run_config.json')
    evidence = _load(report/'references'/'previous_ln128_evidence.json')
    if (_hash(reference)!=references['previous_summary_sha256'] or config_hash(previous_config)!=references['previous_config_sha256'] or
            _hash(evidence)!=references['previous_ln128_evidence_sha256'] or _hash(_load(report/'references'/'profile_definition.json'))!=references['profile_definition_sha256']):
        raise ValueError('previous reference, profile, or carried historical evidence hash differs')
    revision_hash = hashlib.sha256((report/'references'/'user_revision.md').read_bytes()).hexdigest()
    if revision_hash!=references['revision_spec_sha256'] or revision_hash!=manifest['files'][REVISION_SPEC]:
        raise ValueError('grid user-revision specification hash differs')
    cases = run_cases(config,workers=workers)
    derived = derive_grid(cases)
    aggregates = aggregate_shapes(derived)
    regression = compare_overlap(cases,reference)
    checks = {'case_envelopes_exact':cases==saved['cases'],'complete_results_exact':[case['result'] for case in cases]==saved['results'],
              'traits_exact':_traits(cases)==saved['mapping_traits'],'derived_metrics_exact':derived==saved['derived_metrics'],
              'shape_summary_exact':aggregates==saved['shape_summary'],'pareto_exact':pareto(derived+aggregates)==saved['pareto'],
              'target_comparisons_exact':target_comparisons(derived,aggregates)==saved['target_comparisons'],
              'intersection_exact_dicts':regression['passed'],'intersection_replay_exact':regression==_load(report/'intersection_regression.json'),
              'source_and_reference_hashes_verified':True,'ln128_historical_evidence_hash_verified':True}
    passed = all(checks.values())
    dump(report/'verification.json',dict(checks,status='pass' if passed else 'failed',exact_replay=passed,
                                         case_count=len(cases),intersection_case_count=8,workers=workers))
    if not passed:
        raise AssertionError('grid replay differs in complete results, ledgers, metrics or overlap regression')
    print('Exact grid replay PASS: 84 complete cases with eight same-condition intersection checks',flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Explicit P20..33 x Q4/6/8 array grid; LN64 / 64 KiB / 300 MB/s')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out',help='new immutable report directory')
    group.add_argument('--replay',help='existing grid report directory')
    parser.add_argument('--workers',type=int,default=4,help='independent spawn workers; default 4, use 1 for sequential replay')
    args = parser.parse_args(argv)
    if args.workers<1:
        parser.error('--workers must be positive')
    try:
        return replay_study(args.replay,args.workers) if args.replay else build_study(args.out,args.workers)
    except (ValueError,AssertionError,FileExistsError,FileNotFoundError) as error:
        parser.exit(2,'error: %s\n' % error)


if __name__=='__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
