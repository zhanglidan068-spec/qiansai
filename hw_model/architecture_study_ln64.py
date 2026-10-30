"""LN64 with the existing 64 KiB SRAM: an independent reproducible study.

The 32 KiB experiment is a separate profile and is not modified here. Candidate
counts come from ALL_MAPPINGS; historical LN128 results are regression evidence,
not competitors under the new conditions. Pareto objectives are explicit counts
and traffic, never inferred synthesized area, power, or energy.
"""
import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import sys

from .architecture_study import _hash, _load, _table
from .architecture_study_32k_ln64 import run_cases, _traits, _derived, _tables, historical_regression
from .config import LN64_LOCAL64K_CONFIG, LN64_LOCAL64K_PROFILE_ID, load_config, source_manifest
from .experiments import ROOT, _freeze_sources, config_hash, dump
from .fixed_schedule import ALL_MAPPINGS
from .reporting import _csv
from .reference_data import reference_report_path


STUDY_ID = 'architecture_shapes_local64k_ln64'
OPERATORS = ('projection', 'ffn2')
REFERENCE_REPORT = reference_report_path('architecture_study_20261029_v1')
REVISION_SPEC = 'hw_model/spec/user_revision_20261029_ln64_local64k.md'
PROFILE_PATH = 'hw_model/configs/ln64_local64k.json'
PARETO_AXES = {'cycles_multipliers':'physical_multipliers',
               'cycles_weight_traffic':'physical_weight_bytes',
               'cycles_local_traffic':'local_physical_total_bytes'}


def _validate_profile(config, old_config=None):
    if config['provenance']['scenario_id'] != LN64_LOCAL64K_PROFILE_ID:
        raise ValueError('wrong LN64 / local64k profile')
    local = config['local_memory']
    if (local['banks'],local['depth_words'],local['word_bytes'],local['port_mode'],local['read_latency']) != (8,4096,2,'single',1):
        raise ValueError('this study retains eight 4096x16-bit single-port SRAM macros')
    ln = config['ln']
    if (ln['latency_cycles'],ln['initiation_interval_cycles'],ln['max_inflight_rows']) != (64,64,1):
        raise ValueError('this study requires LN core L64 / II64 / one inflight row')
    if old_config is not None:
        for section in ('model','numeric','platform','local_memory','architecture'):
            if config[section] != old_config[section]:
                raise ValueError('only LN changes; %s differs from the historical condition' % section)
        old_ln = dict(old_config['ln'],latency_cycles=64,initiation_interval_cycles=64)
        if ln != old_ln:
            raise ValueError('other LN settings differ from the historical condition')


def _augment_derived(cases):
    rows = _derived(cases)
    for row in rows:
        if row['status']=='feasible':
            row['local_physical_total_bytes'] = row['local_physical_read_bytes']+row['local_physical_write_bytes']
    return rows


def pareto_fronts(derived):
    """Non-dominated sampled points; both objectives are minimized.

    Equal-coordinate candidates both remain on the front. A front only compares
    feasible results for the same operator under this saved platform profile.
    """
    fronts, records = {}, []
    for operator in OPERATORS:
        rows = [row for row in derived if row['operator']==operator and row['status']=='feasible']
        fronts[operator] = {}
        for name,axis in PARETO_AXES.items():
            nondominated = []
            for row in rows:
                dominating = [other['mapping'] for other in rows
                              if other['total_cycles']<=row['total_cycles'] and other[axis]<=row[axis]
                              and (other['total_cycles']<row['total_cycles'] or other[axis]<row[axis])]
                front = not dominating
                if front:
                    nondominated.append(row['mapping'])
                records.append({'operator':operator,'mapping':row['mapping'],'architecture':row['architecture'],
                                'front':name,'x_objective':axis,'x_value':row[axis],
                                'total_cycles':row['total_cycles'],'nondominated':front,
                                'dominated_by':sorted(dominating)})
            fronts[operator][name] = sorted(nondominated)
    return {'objectives':'minimize total_cycles and the named count/traffic objective',
            'not_area_power_or_energy_estimates':True,'fronts':fronts,'records':records}


def _analysis(cases, derived, pareto):
    feasible = [row for row in derived if row['status']=='feasible']
    lines = ['# LN64：保留 64 KiB 局部 SRAM 的全候选研究','',
             '本次用户只修改 LN 核心为 L=64、II=64、最多一行，并要求不限乘法器数比较 33×4 等新方案及此前全部方案。SRAM 保持八个 4096×16-bit 单端口宏，共 64 KiB；工作区另有 32 KiB 实验，它是独立 profile，不是本轮条件。',
             '权重仍为 150 MHz 下 300 MB/s：4 B/beat、II2、响应3、最多2在途。局部读写共用 bank，峰值合计 16 B/cycle。LN 输入提取与输出写入确认在 64 周期核心之外计时。',
             '主窗口从第一次权重请求到最后内部 INT8 输出确认；激活、残差和参数预先就绪，初始化单列，不含共享输出写回。','',
             '共 %d 种映射、%d 个案例，其中 %d 个可行。候选数与并行形状来自保存的 registry/traits，未限制为 128 或 144 个乘法器。' %
             (len({case['mapping'] for case in cases}),len(cases),len(feasible)),'']
    for operator in OPERATORS:
        rows = sorted((row for row in feasible if row['operator']==operator),key=lambda row:(row['total_cycles'],row['physical_multipliers'],row['mapping']))
        lines += ['## '+operator,'']
        if rows:
            best = rows[0]
            lines += ['本模型下完成最快为 %s：%d cycles，%d 个乘法器、%d B 寄存器载荷。该结论只比较时序，不代表面积或功耗最优。' %
                      (best['mapping'],best['total_cycles'],best['physical_multipliers'],best['register_payload_bytes']),'',
                      '| 映射 | 乘法器 | 完成周期 | 首矩阵行 | 最后矩阵行 | 矩阵后尾部 | 权重物理 B | 局部读写 B |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for row in rows:
                lines.append('| %s | %d | %d | %d | %d | %d | %d | %d |' %
                             (row['mapping'],row['physical_multipliers'],row['total_cycles'],row['first_matrix_row'],
                              row['final_matrix_row'],row['after_matrix_tail_cycles'],row['physical_weight_bytes'],row['local_physical_total_bytes']))
            lines += ['','样本中的非支配候选（同时最小化周期与第二项指标）：','']
            for name,label in (('cycles_multipliers','周期 / 乘法器数'),('cycles_weight_traffic','周期 / 权重物理流量'),('cycles_local_traffic','周期 / 局部物理流量')):
                lines.append('- %s：%s。' % (label,', '.join(pareto['fronts'][operator][name])))
            lines += ['','非支配只表示本次离散候选在两项计数上的取舍，不是连续设计空间最优，也不是综合面积、功耗或能量结论。','']
            indexed = {row['mapping']:row for row in rows}
            if all(mapping in indexed for mapping in ('A32x4','A33x4','A33x8','A99x16')):
                c32,c33,c338,c9916 = (indexed[mapping] for mapping in ('A32x4','A33x4','A33x8','A99x16'))
                lines += ['### 33×4、32×4 与更宽列数的供数取舍','',
                          '本次实测：A32x4=%d、A33x4=%d、A33x8=%d、A99x16=%d cycles。A33x4 / A33x8 的局部物理读取分别为 %d / %d B，权重物理读取分别为 %d / %d B。' %
                          (c32['total_cycles'],c33['total_cycles'],c338['total_cycles'],c9916['total_cycles'],
                           c33['local_physical_read_bytes'],c338['local_physical_read_bytes'],
                           c33['physical_weight_bytes'],c338['physical_weight_bytes']),'',
                          '99 个 token 可正好分成三组 33 行；32 行需四组，尾组只有 3 行。然而每个 k 的 33 B 激活至少涉及 17 个 16-bit 字，八 bank 至少三个授权周期；32 行的 16 个字至少两个周期。消除尾组不等于每次发射的局部取数成本更低。该下界不是详细周期的替代，实际读写、冲突和流水依赖仍由模拟器决定。',
                          '四列权重包为 4 B，稳态服务占用两周期；八列包为 8 B，稳态服务占用四周期，更接近 33 行激活提取的节奏，并使相同输出通道数需要的激活重读减少。代价是更多乘法器和累加寄存器：A33x4=%d 个乘法器、%d B 累加器；A33x8=%d 个乘法器、%d B 累加器。' %
                          (c33['physical_multipliers'],c33['accumulator_register_bytes'],c338['physical_multipliers'],c338['accumulator_register_bytes']),
                          'A99x16 的物理乘法器为 %d 个，总寄存器载荷 %d B，其中激活 %d B、累加器 %d B。它们均已计入报告，不能把更大并行度和寄存器视为免费。最短周期及 Pareto 选择仍以本子图动态结果为准。' %
                          (c9916['physical_multipliers'],c9916['register_payload_bytes'],c9916['activation_register_bytes'],c9916['accumulator_register_bytes']),'']
    lines += ['## 如何解释资源、时序与验证','',
              '- 形状改变会改变物理乘法器、权重包、激活寄存器、累加器、尾组和 bank 冲突。更大乘法器数量没有被当成免费资源；寄存器、SRAM 和实际访问均单列。',
              '- 不同阵列是分别建模的候选，不代表已实现可重构硬件；形状切换、控制、互连与布线成本未综合。',
              '- “矩阵后尾部”=最终内部输出确认−最后矩阵行完成；LN 与矩阵重叠，不能把尾部直接当成全部 LN 忙碌周期。',
              '- 发射有效比例的分母为乘法器×发射次数；发射占空比分母为总周期；端到端 MAC 利用率分母为乘法器×总周期。',
              '- 局部 SRAM 仍为八个宏，峰值 16 B/cycle。无争用时，一行 128 B INT16 LN 输入需至少 8 个局部服务周期，64 B INT8 输出需至少 4 个服务周期；读取延迟、确认和冲突另计。实际 LN 启动间隔可能大于核心 II64。',
              '- 旧 64 KiB / LN128 的原 16 个完整结果字典在其旧配置下独立重跑，用于历史兼容性。它们没有混入新排名或新条件加速比。',
              '- 这是元数据行为时序仿真，不是 RTL 或综合。独立浮点 LN 参考不等于位精确定点 LN；真实 checkpoint 准确率、可达频率、面积、功耗和能量尚未验证。',
              '- 冻结重放核对所有案例完整字典、traits、派生指标、Pareto 集、引用摘要/config/spec 哈希和旧 16 项回归。','']
    return '\n'.join(lines)


def _pareto_chart(rows, front, axis, title):
    if not rows:
        return ''
    maximum_x, maximum_y = max(row[axis] for row in rows) or 1,max(row['total_cycles'] for row in rows) or 1
    items = []
    for step in range(5):
        x,y = 95+step*700/4,30+step*260/4
        items.append('<line x1="%.2f" y1="30" x2="%.2f" y2="290" stroke="#e3e9ec"/><text x="%.2f" y="312" text-anchor="middle">%.5g</text>' % (x,x,x,maximum_x*step/4))
        items.append('<line x1="95" y1="%.2f" x2="795" y2="%.2f" stroke="#e3e9ec"/><text x="85" y="%.2f" text-anchor="end">%.5g</text>' % (y,y,y+4,maximum_y*(1-step/4)))
    for row in rows:
        x,y = 95+700*row[axis]/maximum_x,290-260*row['total_cycles']/maximum_y
        nondominated = row['mapping'] in front
        title_text = '%s: cycles=%d, %s=%s' % (row['mapping'],row['total_cycles'],axis,row[axis])
        items.append('<circle cx="%.2f" cy="%.2f" r="%s" fill="%s" opacity="%s"><title>%s</title></circle>' %
                     (x,y,'5' if nondominated else '3','#177a74' if nondominated else '#879bb1','1' if nondominated else '.45',escape(title_text)))
        if nondominated:
            items.append('<text x="%.2f" y="%.2f">%s</text>' % (x+7,y-7,escape(row['mapping'])))
    return ('<h3>%s</h3><p class="muted">横轴：%s；纵轴：完成周期。绿色为当前候选中的非支配点；其他点可悬停查看。</p><svg role="img" aria-label="%s" viewBox="0 0 920 335">%s</svg>' %
            (escape(title),escape(axis),escape(title,quote=True),''.join(items)))


def _cycle_chart(rows):
    if not rows:
        return ''
    maximum = max(row['total_cycles'] for row in rows)
    items = []
    for index,row in enumerate(rows):
        y,width = 15+index*27,610*row['total_cycles']/maximum
        color = '#177a74' if row['architecture'].startswith('array') else '#657fb6'
        items.append('<text x="175" y="%d" text-anchor="end">%s</text><rect x="190" y="%d" width="%.2f" height="18" rx="3" fill="%s"/><text x="%.2f" y="%d">%d</text>' %
                     (y+13,escape(row['mapping']),y,width,color,198+width,y+13,row['total_cycles']))
    return '<svg role="img" aria-label="completion cycles under LN64 and 64 KiB" viewBox="0 0 940 %d">%s</svg>' % (len(rows)*27+35,''.join(items))


def _html(config,cases,derived,pareto,manifest,analysis):
    sections = []
    timing = [('mapping','映射'),('input_shape','输入 M×R'),('weight_shape','权重 R×N'),('physical_multipliers','乘法器'),
              ('total_cycles','完成周期'),('first_matrix_fragment','首矩阵片段'),('first_matrix_row','首矩阵行'),
              ('final_matrix_row','最后矩阵行'),('first_ln_row_done','首 LN 行'),('after_matrix_tail_cycles','矩阵后尾部')]
    resources = [('mapping','映射'),('group_rows','token 组'),('column_lanes','输出 lanes'),('physical_multipliers','乘法器'),
                 ('register_payload_bytes','总寄存器 B'),('weight_register_bytes','权重寄存器 B'),
                 ('activation_register_bytes','激活寄存器 B'),('accumulator_register_bytes','累加器 B'),('partial_sram_bytes','部分和 SRAM B')]
    traffic = [('mapping','映射'),('physical_weight_bytes','权重物理 B'),('local_physical_read_bytes','局部物理读 B'),
               ('local_physical_write_bytes','局部物理写 B'),('local_denied_cycles','局部冲突周期'),
               ('issue_lane_utilization','发射有效比例'),('issue_duty_cycle','发射占空比'),('compute_utilization','端到端 MAC 利用率')]
    for operator in OPERATORS:
        rows = sorted((row for row in derived if row['operator']==operator and row['status']=='feasible'),key=lambda row:(row['total_cycles'],row['physical_multipliers'],row['mapping']))
        sections += ['<section><h2>%s：同条件完成周期</h2>' % escape(operator),_cycle_chart(rows),_table(timing,rows),
                     '<h3>并行形状与实际寄存器</h3>',_table(resources,rows),'<h3>流量、冲突与三个利用率</h3>',_table(traffic,rows),'</section>']
        sections.append('<section><h2>%s：周期与资源/流量的取舍</h2>' % escape(operator))
        for front,axis,title in (('cycles_multipliers','physical_multipliers','完成周期 / 物理乘法器'),
                                 ('cycles_weight_traffic','physical_weight_bytes','完成周期 / 权重物理流量')):
            sections.append(_pareto_chart(rows,pareto['fronts'][operator][front],axis,title))
        fronts = [{'front':name,'mappings':', '.join(names)} for name,names in pareto['fronts'][operator].items()]
        sections += [_table([('front','最小化目标'),('mappings','非支配候选')],fronts),
                     '<p class="muted">相同坐标的候选都保留。乘法器数和字节流量只是明确计数，不是综合面积或能耗替代量；本轮没有以未知功耗计算效率。</p></section>']
    details = []
    for case in cases:
        result = case['result']
        data = {key:result[key] for key in ('cycle_breakdown','physical_local_layout','resources','buffers','initialization')}
        details.append('<details><summary>%s · %s</summary><pre>%s</pre></details>' %
                       (escape(case['operator']),escape(case['mapping']),escape(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True))))
    style = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#233944;font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif}
    main{max-width:1420px;margin:28px auto;padding:0 25px 45px}header{background:#173f4b;color:white;padding:28px 32px;border-radius:12px}h1{font-size:29px;line-height:1.4;margin:7px 0 12px}
    .eyebrow{font-size:12px;letter-spacing:2px;color:#a8d8d0}header p{margin:7px 0;color:#d2e7ea}section{background:white;border:1px solid #dce5e9;border-radius:10px;padding:24px 28px;margin:20px 0}
    h2{font-size:21px;margin:0 0 13px}h3{font-size:16px;margin:21px 0 9px}.notice{background:#fff7e7;border-left:4px solid #c99038;padding:14px 18px;margin:18px 0}.muted{font-size:13px;color:#627c88}
    .scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{white-space:nowrap;text-align:left;padding:10px 12px;border-bottom:1px solid #dce5e9}th{background:#f2f7f8}
    svg{width:100%;height:auto;display:block}svg text{font:12px system-ui,'Microsoft YaHei',sans-serif;fill:#607a86}details{border-top:1px solid #dce5e9;padding:12px 0}summary{cursor:pointer;color:#177a74;font-weight:600}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:16px;font:12px/1.65 Consolas,monospace}code{font:12px Consolas,monospace;overflow-wrap:anywhere}footer{font-size:12px;color:#627c88}
    @media(max-width:700px){main{padding:0 12px;margin:12px auto}header,section{padding:18px}h1{font-size:24px}}
    '''
    return ''.join(['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KWT LN64：保留64 KiB的全候选架构研究</title><style>',style,'</style></head><body><main>',
                    '<header><div class="eyebrow">USER REVISION · LN64 · LOCAL 64 KiB · FIXED 300 MB/s</div><h1>只改 LN 时序，保留 64 KiB 局部 SRAM</h1><p>%d 种映射 × 两个子图，共 %d 个详细案例；不限制乘法器数</p><p>用户 profile：%s</p></header>' % (len({case['mapping'] for case in cases}),len(cases),escape(LN64_LOCAL64K_PROFILE_ID)),
                    '<div class="notice"><strong>本轮没有把 SRAM 改为 32 KiB。</strong>使用八个 4096×16-bit 单端口宏，共 64 KiB，局部读写峰值合计 16 B/cycle。LN 核心 L=64、II=64、单行，输入提取和输出确认另计。工作区 32 KiB 实验是独立 profile，仍保留，不参与本轮排名。</div>',
                    '<section><h2>共同条件与验证范围</h2><p>同一 150 MHz、300 MB/s 权重接口（4 B/beat、II2、响应3、最多2在途）、相同数值格式与后处理。主窗口从第一次权重请求到最后内部 INT8 输出确认。初始化单列；不含共享输出写回。</p><p>候选形状、寄存器、尾组和访问来自已实现的行为模型。不同形状是独立候选，未实现可重构切换，控制与互连成本未综合。本报告不是 RTL、位精确定点 LN、真实 checkpoint、面积或功耗验证。</p></section>',
                    ''.join(sections),'<section><h2>分析与边界</h2><pre>',escape(analysis),'</pre></section>',
                    '<section><h2>逐候选阶段、缓冲、布局与初始化明细</h2>',''.join(details),'</section>',
                    '<footer><p>配置 SHA-256：<code>',escape(config_hash(config)),'</code><br>源码集合：<code>',escape(manifest['aggregate_sha256']),'</code><br>用户修订说明：<code>',escape(REVISION_SPEC),'</code></p><p>summary.json 保留全部新条件结果、traits、Pareto与派生指标；historical_regression.json 单列旧 16 项完整字典回归；source/ 与 references/ 保存源码及配置/说明/参考摘要。verification.json 记录冻结精确重放。</p></footer></main></body></html>'])


def build_study(destination):
    config = load_config(LN64_LOCAL64K_CONFIG)
    reference = _load(REFERENCE_REPORT/'summary.json')
    old_config = load_config(REFERENCE_REPORT/'run_config.json')
    _validate_profile(config,old_config)
    mappings = tuple(ALL_MAPPINGS)
    destination = Path(destination).absolute()
    destination.mkdir(parents=True,exist_ok=False)
    manifest = source_manifest(ROOT)
    if REVISION_SPEC not in manifest['files']:
        raise ValueError('the current user-revision specification is missing from the source manifest')
    _freeze_sources(destination,manifest)
    reference_dir = destination/'references'
    reference_dir.mkdir()
    dump(reference_dir/'historical_summary.json',reference)
    dump(reference_dir/'historical_run_config.json',old_config)
    definition = _load(LN64_LOCAL64K_CONFIG)
    dump(reference_dir/'profile_definition.json',definition)
    revision_bytes = (ROOT/REVISION_SPEC).read_bytes()
    (reference_dir/'user_revision.md').write_bytes(revision_bytes)
    dump(destination/'run_config.json',config)
    dump(destination/'source_manifest.json',manifest)
    dump(destination/'invocation.json',{'argv':sys.argv,'python':sys.version,'executable':sys.executable,
                                      'entrypoint':'hw_model.architecture_study_ln64','profile_id':LN64_LOCAL64K_PROFILE_ID,
                                      'client_timezone':'Asia/Shanghai','trace_enabled':False,
                                      'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat()})
    cases = run_cases(config,mappings=mappings)
    expected = len(mappings)*len(OPERATORS)
    if len(cases)!=expected or any(case['status']!='feasible' for case in cases):
        dump(destination/'incomplete_cases.json',cases)
        raise AssertionError('the full 64 KiB study must have all %d detailed cases feasible' % expected)
    derived = _augment_derived(cases)
    pareto = pareto_fronts(derived)
    historical = historical_regression(reference,old_config)
    dump(destination/'historical_regression.json',historical)
    if not historical['passed']:
        dump(destination/'verification.json',{'status':'failed','historical_complete_dicts_identical':False})
        raise AssertionError('historical sixteen complete result dictionaries differ')
    if source_manifest(ROOT)!=manifest:
        raise ValueError('sources changed during study; refusing an inconsistent frozen bundle')
    summary = {'schema_version':1,'study':STUDY_ID,'profile_id':LN64_LOCAL64K_PROFILE_ID,
               'measurement_boundary':'first_weight_request_to_final_internal_int8_output_ack',
               'config_sha256':config_hash(config),'source_manifest':manifest,
               'authority':{'revision_spec':REVISION_SPEC,'revision_spec_sha256':manifest['files'][REVISION_SPEC],
                            'profile_path':PROFILE_PATH,'profile_definition_sha256':_hash(definition)},
               'historical_reference_summary_sha256':_hash(reference),'historical_reference_config_sha256':config_hash(old_config),
               'mappings':list(mappings),'case_count':expected,'cases':cases,
               'results':[case['result'] for case in cases],'mapping_traits':_traits(cases),'derived_metrics':derived,'pareto':pareto,
               'claims':{'metadata_behavior_simulation':True,'sram_changed_in_this_study':False,'local_sram_bytes':65536,
                         'historical_results_ranked_with_new_conditions':False,'multiplier_count_capped':False,
                         'rtl_verified':False,'area_synthesized':False,'power_measured':False,
                         'fixed_point_ln_bit_accuracy_verified':False,'checkpoint_accuracy_measured':False}}
    dump(destination/'summary.json',summary)
    _tables(destination,cases,derived)
    _csv(destination/'pareto.csv',pareto['records'])
    analysis = _analysis(cases,derived,pareto)
    (destination/'analysis.md').write_text(analysis,encoding='utf-8')
    (destination/'report.html').write_text(_html(config,cases,derived,pareto,manifest,analysis),encoding='utf-8')
    dump(destination/'verification.json',{'status':'simulation_and_historical_pass_replay_pending','case_count':expected,
                                         'historical_complete_dicts_identical':True,'historical_case_count':16,
                                         'source_bundle_consistent':True,'exact_replay':False})
    print('LN64 / local64k study: '+str(destination/'report.html'),flush=True)
    return 0


def replay_study(report):
    report = Path(report).absolute()
    saved = _load(report/'summary.json')
    if saved.get('study')!=STUDY_ID:
        raise ValueError('not an LN64 / local64k study report')
    mappings = tuple(ALL_MAPPINGS)
    if saved['mappings']!=list(mappings) or saved['case_count']!=len(mappings)*len(OPERATORS):
        raise ValueError('saved candidate registry or case count differs from the frozen study')
    config = load_config(report/'run_config.json')
    if config_hash(config)!=saved['config_sha256']:
        raise ValueError('saved revision config hash differs')
    manifest = source_manifest(ROOT)
    if manifest!=saved['source_manifest'] or _load(report/'source_manifest.json')!=manifest:
        raise ValueError('source/runtime differs; replay from frozen source/ with the same Python')
    definition = _load(report/'references'/'profile_definition.json')
    authority = saved['authority']
    if _hash(definition)!=authority['profile_definition_sha256']:
        raise ValueError('saved profile-definition hash differs')
    import hashlib
    revision_hash = hashlib.sha256((report/'references'/'user_revision.md').read_bytes()).hexdigest()
    if revision_hash!=authority['revision_spec_sha256'] or revision_hash!=manifest['files'][REVISION_SPEC]:
        raise ValueError('saved user-revision specification hash differs')
    reference = _load(report/'references'/'historical_summary.json')
    old_config = load_config(report/'references'/'historical_run_config.json')
    _validate_profile(config,old_config)
    if _hash(reference)!=saved['historical_reference_summary_sha256'] or config_hash(old_config)!=saved['historical_reference_config_sha256']:
        raise ValueError('historical reference hashes differ')
    cases = run_cases(config,mappings=mappings)
    derived = _augment_derived(cases)
    historical = historical_regression(reference,old_config)
    checks = {'case_envelopes_exact':cases==saved['cases'],
              'complete_results_exact':[case['result'] for case in cases]==saved['results'],
              'traits_exact':_traits(cases)==saved['mapping_traits'],'derived_metrics_exact':derived==saved['derived_metrics'],
              'pareto_exact':pareto_fronts(derived)==saved['pareto'],
              'case_count_exact':len(cases)==saved['case_count'],
              'historical_complete_dicts_identical':historical['passed'],
              'historical_replay_exact':historical==_load(report/'historical_regression.json'),
              'source_bundle_consistent':True,'reference_hashes_verified':True}
    passed = all(checks.values())
    dump(report/'verification.json',dict(checks,status='pass' if passed else 'failed',exact_replay=passed,
                                         case_count=len(cases),historical_case_count=16,python_version=manifest['python_version']))
    if not passed:
        raise AssertionError('LN64 study differs in complete results, traits, Pareto or historical regression')
    print('Exact LN64 / local64k replay PASS: %d new complete dictionaries and 16 historical cases' % len(cases),flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='LN64-only revision; retain 64 KiB SRAM and compare the full candidate registry')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out',help='new immutable report directory')
    group.add_argument('--replay',help='existing LN64 / local64k report directory')
    args = parser.parse_args(argv)
    try:
        return replay_study(args.replay) if args.replay else build_study(args.out)
    except (ValueError,AssertionError,FileExistsError,FileNotFoundError) as error:
        parser.exit(2,'error: %s\n' % error)


if __name__=='__main__':
    raise SystemExit(main())
