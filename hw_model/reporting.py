"""Schema-2 fixed-contract reports; no bandwidth or LN sweeps."""
import csv
import hashlib
from html import escape
import json
from pathlib import Path
from .fixed_schedule import fixed_l0_benchmarks


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def _flat(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix or 'value': value}
    result = {}
    for key in sorted(value):
        name = prefix+'.'+str(key) if prefix else str(key)
        item = value[key]
        if isinstance(item, dict):
            result.update(_flat(item,name))
        elif isinstance(item,(list,tuple)):
            result[name]=json.dumps(item,ensure_ascii=False,sort_keys=True)
        else:
            result[name]=item
    return result


def _csv(path, rows):
    leading=('mapping','architecture','operator')
    fields=set()
    for row in rows:
        fields.update(row)
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(leading)+sorted(fields-set(leading)))
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    return 'unknown' if value is None else '%.6g'%value if isinstance(value,float) else str(value)


def _table(columns,rows):
    return '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>'%(
        ''.join('<th>%s</th>'%escape(label) for _,label in columns),
        ''.join('<tr>%s</tr>'%''.join('<td>%s</td>'%escape(_number(row.get(key))) for key,_ in columns) for row in rows))


def _bar_chart(results):
    maximum=max([row.get('total_cycles',0) for row in results] or [1]) or 1
    items=[]
    for index,row in enumerate(results):
        y,width=15+index*35,620*row.get('total_cycles',0)/maximum
        color='#197b75' if row.get('mapping')=='A' else '#c57936' if row.get('mapping')=='B0' else '#506fb2'
        label='%s · %s'%(row.get('operator'),row.get('mapping'))
        items.append('<text x="165" y="%d" text-anchor="end">%s</text><rect x="180" y="%d" width="%.2f" height="23" rx="3" fill="%s"/><text x="%.2f" y="%d">%s</text>'%
                     (y+16,escape(label),y,max(1,width),color,190+width,y+16,escape(_number(row.get('total_cycles')))))
    return '<svg role="img" aria-label="fixed 300 MB/s completion cycles" viewBox="0 0 940 %d">%s</svg>'%(max(70,len(results)*35+30),''.join(items))


def _timeline(trace):
    groups={}
    for event in trace:
        begin=event.get('cycle')
        if not isinstance(begin,(int,float)) or not 0<=begin<1000:
            continue
        kind=str(event.get('event','')).upper()
        if not any(word in kind for word in ('ISSUE','LOCAL_READ','LOCAL_WRITE','LN_START','LN_ACCEPT','LN_DONE','OUTPUT_CONFIRM')):
            continue
        end=event.get('complete_cycle',event.get('done_cycle',begin+1))
        if not isinstance(end,(int,float)):
            end=begin+1
        lane=0 if 'WEIGHT' in kind else 1 if 'LOCAL' in kind else 2 if 'ISSUE' in kind else 3
        group=(str(event.get('mapping','unlabeled')),str(event.get('operator','unlabeled')))
        groups.setdefault(group,[]).append((lane,begin,min(1000,max(begin+1,end)),kind))
    parts=[]
    labels,colors=['权重端口','局部 SRAM','矩阵发射','LN / 输出'],['#8b9fbd','#527da2','#197b75','#a16b9b']
    for group in sorted(groups):
        items=['<text x="110" y="%d" text-anchor="end">%s</text>'%(31+i*30,label) for i,label in enumerate(labels)]
        for tick in range(0,1001,200):
            x=130+tick*.75
            items.append('<line x1="%.1f" y1="10" x2="%.1f" y2="140" stroke="#dde5e9"/><text x="%.1f" y="159" text-anchor="middle">%d</text>'%(x,x,x,tick))
        for lane,begin,end,kind in groups[group]:
            items.append('<rect x="%.2f" y="%d" width="%.2f" height="18" fill="%s" opacity=".7"><title>%s [%s,%s)</title></rect>'%
                         (130+begin*.75,17+lane*30,max(1,(end-begin)*.75),colors[lane],escape(kind),begin,end))
        parts.append('<details><summary>%s · %s</summary><svg role="img" aria-label="first 1000 cycles" viewBox="0 0 940 175">%s</svg></details>'%(escape(group[0]),escape(group[1]),''.join(items)))
    return ''.join(parts) or '<p class="muted">本次未启用详细轨迹；需要逐事件窗口时使用 --trace。</p>'


def _readable(results):
    rows=[]
    for result in results:
        row=dict(result)
        port,local,resource=result.get('weight_port',{}),result.get('local_memory',{}),result.get('resources',{})
        row.update(weight_bytes=port.get('useful_weight_bytes'),physical_weight_bytes=port.get('physical_weight_bytes'),
                   weight_busy=port.get('bus_busy_cycles'),weight_service=port.get('occupied_service_cycles'),
                   fills=port.get('fill_count'),reuse=port.get('token_reuse_beyond_first',port.get('reuse_count')),
                   token_consumers=port.get('logical_token_consumers'),
                   weight_peak_reserved=port.get('register_peak_reserved_bytes'),weight_peak_live=port.get('register_peak_live_bytes'),
                   local_read=local.get('physical_read_bytes'),local_write=local.get('physical_write_bytes'),
                   local_conflicts=local.get('denied_cycles'),register_bytes=resource.get('total_register_payload_bytes'),
                   weight_register_bytes=resource.get('register_payload_bytes',{}).get('weight_register_bytes'),
                   partial_sram_bytes=resource.get('b1_partial_sram_bytes'),local_sram_bytes=resource.get('local_sram_bytes'),
                   area_status=resource.get('area',{}).get('status','unknown'))
        rows.append(row)
    return rows


ASSUMPTIONS=[
'修改条款优先于原文；当前配置 schema_version=2。本轮不运行其他权重带宽和 LN 间隔扫描。',
'主窗口：第一次权重请求至最后内部 INT8 LN 输出确认。激活、残差和 bias/gamma/beta 已在局部就绪；权重寄存器为空。',
'初装、共享权重源预排布和参数初始化单列；外部初始化时间未定义的部分为 unknown。本结果不是系统冷启动时间。',
'主窗口不含最后共享 SRAM 写回；不得与旧报告的共享写回终点直接混比。',
'150 MHz；权重 4 B/beat、接受间隔 2、响应延迟 3、最多 2 outstanding；同周期允许返回并接受新拍。',
'权重端口稳态 2 B/cycle = 300,000,000 B/s；只应用一次，不再叠加权重 SRAM 的 bank 供数限制。',
'A 仅两组 16 B 权重寄存器，B 仅两组 128 B；完整有效掩码到齐后才可消费，最后一个 token 完成前不能覆盖。',
'公共局部存储固定 8×4096×16-bit 单端口宏（64 KiB），读延迟 1，所有激活/残差/后处理/LN/输出/部分和请求共享 bank 仲裁。',
'A 激活以 k-major 存放，token 数填充为偶数；B 为 row-major。源权重按映射打包，原始矩阵值相同，打包/初始化成本单列。',
'B1_G9 和 B1_G99 是同一 B 硬件的跨 token 权重复用调度；部分和 8×G B 使用同一局部 SRAM。容量不足应判不可行，不增加共享激活访问。',
'INT8 A/B、INT32 累加和 bias、INT16 LN 输入与 gamma/beta、INT8 输出；nearest-even 舍入与饱和。',
'后处理每周期最多 8 通道、结果延迟 2；局部 SRAM 读取/写入仍独立参与仲裁。',
'LN 核心是固定行事件：L=128，II=128，最多一行。全部输入准备到 LN 工作寄存器且预留输出空间后启动；输入提取、输出写入确认另计。',
'数值参考与时序模型分开；LN 浮点参考不是位精确定点 LN，未把统计/rsqrt/affine 延迟再叠加到 128。',
'寄存器有效载荷不包含全部控制、标签、地址重排、互连和物理实现开销；面积缺少宏或标准单元数据时保持 unknown。',
'尚未测量真实 checkpoint 识别准确率、全网时延、功耗、官方库面积或 RTL 逐事件一致性。']


def _html(config,results,source,trace,digest):
    rows=_readable(results)
    times=[('operator','子图'),('mapping','映射'),('total_cycles','完成周期'),('first_matrix_fragment','首矩阵片段'),
           ('first_matrix_row','首完整矩阵行'),('first_ln_row_done','首 LN 行完成'),('last_internal_output_confirm','最后内部输出确认')]
    port=[('operator','子图'),('mapping','映射'),('weight_bytes','权重有效 B'),('physical_weight_bytes','权重物理 B'),
          ('weight_busy','接受 beat 周期数'),('weight_service','服务占用周期'),('fills','填充次数'),('token_consumers','token 消费次数'),('reuse','额外 token 复用次数'),
          ('weight_peak_reserved','权重预留峰值 B'),('weight_peak_live','权重活跃峰值 B')]
    usage=[('operator','子图'),('mapping','映射'),('issue_lane_utilization','发射有效比例'),('issue_duty_cycle','发射占空比'),('compute_utilization','端到端利用率')]
    memory=[('operator','子图'),('mapping','映射'),('local_read','局部物理读 B'),('local_write','局部物理写 B'),('local_conflicts','局部冲突周期')]
    resource=[('operator','子图'),('mapping','映射'),('local_sram_bytes','局部 SRAM B'),('weight_register_bytes','权重寄存器 B'),
              ('register_bytes','总寄存器载荷 B'),('partial_sram_bytes','B1 部分和 SRAM B'),('area_status','面积状态')]
    details=[]
    for result in results:
        data={key:result.get(key,{}) for key in ('initialization','weight_port','local_memory','physical_local_layout','buffers','resources','cycle_breakdown','milestones')}
        details.append('<details><summary>%s · %s</summary><pre>%s</pre></details>'%(escape(str(result.get('operator'))),escape(str(result.get('mapping'))),escape(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True))))
    l0rows=[]
    for item in fixed_l0_benchmarks()['operators']:
        for mapping in ('A','B0'):
            l0rows.append(dict(operator=item['operator'],mapping=mapping,**item[mapping]))
    style="""
    :root{color-scheme:light;--ink:#22343e;--muted:#667b87;--line:#d8e3e8}*{box-sizing:border-box}
    body{margin:0;background:#edf3f5;color:var(--ink);font:15px/1.7 system-ui,'Microsoft YaHei',sans-serif}
    main{max-width:1280px;margin:28px auto;padding:0 25px 45px}header{padding:26px 32px;background:#173e4b;color:white;border-radius:12px}
    h1{font-size:29px;line-height:1.4;margin:6px 0 12px}.eyebrow{font-size:12px;letter-spacing:2px;color:#a2d7d3}
    header p{color:#d5e8ed;margin:8px 0}section{background:white;border:1px solid var(--line);border-radius:10px;margin:20px 0;padding:24px 28px}
    h2{font-size:21px;margin:0 0 14px}.notice{padding:15px 20px;background:#fff7e7;border-left:4px solid #c88c38;margin:18px 0}
    .muted{font-size:13px;color:var(--muted)}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}
    th,td{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}th{background:#f2f7f8;font-weight:600}
    svg{width:100%;height:auto;display:block}svg text{font:12px system-ui,'Microsoft YaHei',sans-serif;fill:#5f7581}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:16px;font:12px/1.6 Consolas,monospace}
    code{font:12px Consolas,monospace;overflow-wrap:anywhere}details{padding:12px 0;border-top:1px solid var(--line)}
    summary{cursor:pointer;color:#197b75;font-weight:600}li{margin:8px 0}footer{font-size:12px;color:var(--muted)}
    @media(max-width:700px){main{padding:0 12px;margin:12px auto}header,section{padding:18px}h1{font-size:24px}}
    """
    return ''.join([
'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KWT 固定 300 MB/s 行为模型</title><style>',style,'</style></head><body><main>',
'<header><div class="eyebrow">KWT · FIXED 300 MB/s · SCHEMA 2</div><h1>同一权重通路下的两种架构与四种调度</h1><p>A：9×16 阵列；B0 / B1：同一 128 乘法器单元的调度变体</p><p>',escape(config['provenance']['scenario_id']),'</p></header>',
'<div class="notice"><strong>新测量边界：</strong>第一次权重请求开始，最后一行 INT8 LN 输出写入内部输出缓冲并确认。激活、残差和参数预先就绪，权重寄存器为空。主窗口不含激活初装与共享 SRAM 输出写回；不能与旧报告的共享写回终点直接混比，也不能称为系统冷启动时间。</div>',
'<section><h2>固定条件</h2><p>150 MHz；权重 4 B/beat，每两周期接受一拍，响应延迟 3 周期，最多两个未完成请求；稳态 2 B/cycle＝十进制 300 MB/s。只应用这一份权重带宽，没有额外权重 SRAM bank 限制。</p><p>局部 SRAM：8×4096×16-bit 单端口宏，64 KiB，最多 16 B/cycle 读写，延迟 1。LN 核心 L=128、II=128、最多一行；输入提取与输出存储确认另计。</p><p class="muted">INT8 输入/权重、INT32 累加/bias、INT16 LN 输入和 gamma/beta、INT8 输出。128 周期 LN 是行级抽象事件，浮点 LN 参考独立，尚非位精确定点电路。</p></section>',
'<section><h2>详细完成时刻</h2>',_bar_chart(results),_table(times,rows),'<p class="muted">A / B 分别有 144 / 128 个乘法器。B1 为 B 的调度变体，不是第三种硬件架构。</p></section>',
'<section><h2>权重填充与复用</h2>',_table(port,rows),'<p class="muted">接受 beat 周期数每接受一拍计 1；服务占用每拍计 2 周期，不含响应首尾。128 B 包经过 32 个真实 4 B 拍，完整有效掩码到齐后才能消费；所有消费者完成前不可覆盖。</p></section>',
'<section><h2>利用率的三个分母</h2>',_table(usage,rows),'<p>发射有效比例＝有效 MAC /（物理乘法器 × 发射次数）；发射占空比＝发射次数 / 完成周期；端到端 MAC 利用率＝有效 MAC /（物理乘法器 × 完成周期）。表中为 0 至 1 比例。</p></section>',
'<section><h2>局部访问与资源</h2>',_table(memory,rows),_table(resource,rows),'<p class="muted">局部 SRAM 不保存权重。B1 部分和 8×G B 已在同一 64 KiB 宏中，不另加宏，不重复算寄存器。寄存器预留与活跃峰值见详情。面积缺失保持 unknown，不判定满足 2.25 mm²。</p></section>',
'<section><h2>调度与数据流</h2><ul><li>A：九行一组、十六通道一块，每个 k 的 16 B 权重广播九行；两组权重寄存器 32 B，144 个 INT32 累加器 576 B。</li><li>B0：严格逐行，先载入当前输入行；128 B 包用于当前 token 的两个 64 项点积。权重双寄存器 256 B，基本部分和 8 B。</li><li>B1_G9 / B1_G99：保留当前包供 G 个 token 依次消费；从公共 SRAM 重读激活，部分和在 SRAM 读改写，完整行到齐后才进入后处理和 LN。</li><li>源权重按映射预先打包，原始矩阵值相同，排列成本在初始化单列。A 激活 k-major 并填充偶数 token，B 为 row-major。</li></ul></section>',
'<section><h2>初始化、逐 bank 计数、物理预留与活跃峰值</h2><p class="muted">局部初始化写入操作数与周期单列；外部首次搬运来源和带宽未定义时保持 unknown。主窗口共享激活访问为零；容量不足的候选不可行。</p>',''.join(details),'</section>',
'<section><h2>L0 手算检查：单独列示</h2>',_table([('operator','子图'),('mapping','映射'),('weight_bytes','权重 B'),('weight_cycles','稳态权重周期'),('total_cycles','重叠 LN 后周期')],l0rows),'<p class="muted">只用稳定 2 B/cycle 权重流与 LN128，省略请求首尾、局部访问、部分和、排空和后处理，不能替代详细结果。</p></section>',
'<section><h2>前 1,000 周期事件窗口</h2>',_timeline(trace),'</section>',
'<section><h2>适用边界与假设</h2><ul>',''.join('<li>%s</li>'%escape(text) for text in ASSUMPTIONS),'</ul></section>',
'<footer><p>配置 SHA-256：<code>',escape(digest),'</code><br>源码集合：<code>',escape(source.get('aggregate_sha256','unknown')),'</code><br>修改条款：<code>',escape(config['provenance']['amendment_sha256']),'</code></p><p>summary.json 保留全部原始字段；operators.csv / memory.csv / buffers.csv / initialization.csv 保存分项；配置、source/ 快照与 invocation.json 用于精确重放。</p></footer></main></body></html>'])


def write_report(path,config,results,source,trace,sweep=None):
    if sweep:
        raise ValueError('bandwidth/LN sweeps are excluded by the fixed-300 contract')
    if config['schema_version']!=2:
        raise ValueError('report requires schema_version 2')
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    results,trace=list(results),list(trace)
    digest=hashlib.sha256(json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    claims={'rtl_calibrated':False,'full_kwt_inference_timed':False,'checkpoint_inference_accuracy_measured':False,
            'area_synthesized':False,'fixed_point_ln_bit_accuracy_verified':False,
            'timing_simulation_uses_metadata_not_tensor_values':True,'weight_bandwidth_swept':False,
            'main_window_includes_activation_initialization':False,'main_window_includes_shared_output_writeback':False}
    summary={'schema_version':2,'scope':config['scope'],'measurement_boundary':'first_weight_request_to_last_internal_output_confirmation',
             'scenario_id':config['provenance']['scenario_id'],'config_sha256':digest,'source_manifest':source,'results':results,'claims':claims}
    _dump(path/'run_config.json',config)
    _dump(path/'source_manifest.json',source)
    _dump(path/'summary.json',summary)
    operators,memory,buffers,initialization=[],[],[],[]
    for result in results:
        identity={key:result.get(key,'') for key in ('mapping','architecture','operator')}
        operators.append(_flat({key:value for key,value in result.items() if key not in ('weight_port','local_memory','buffers','resources','initialization','trace')}))
        row=dict(identity)
        for key in ('weight_port','local_memory','physical_local_layout','resources'):
            row.update(_flat(result.get(key,{}),key))
        memory.append(row)
        buffers.append(dict(identity,**_flat(result.get('buffers',{}))))
        initialization.append(dict(identity,**_flat(result.get('initialization',{}))))
    _csv(path/'operators.csv',operators)
    _csv(path/'memory.csv',memory)
    _csv(path/'buffers.csv',buffers)
    _csv(path/'initialization.csv',initialization)
    with (path/'trace.jsonl').open('w',encoding='utf-8') as stream:
        for event in trace:
            stream.write(json.dumps(event,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    assumptions=['# 固定 300 MB/s 的测量假设','']+ASSUMPTIONS+[
        '','配置 SHA-256：'+digest,'源码集合 SHA-256：'+source.get('aggregate_sha256','unknown'),
        '修改条款 SHA-256：'+config['provenance']['amendment_sha256'],'']
    (path/'assumptions.md').write_text('\n'.join(assumptions),encoding='utf-8')
    lines=['# 固定 300 MB/s 架构与调度比较','','主窗口从第一次权重请求到最后内部输出确认；激活初装单列，不含共享输出写回。','',
           '| 子图 | 映射 | 架构 | 完成周期 | 首矩阵片段 | 首完整矩阵行 | 首 LN 行完成 | 最后内部输出确认 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for result in results:
        lines.append('| '+' | '.join(_number(result.get(key)) for key in ('operator','mapping','architecture','total_cycles','first_matrix_fragment','first_matrix_row','first_ln_row_done','last_internal_output_confirm'))+' |')
    lines+=['','B1_G9 / B1_G99 为同一 B 硬件的调度变体。A/B 分别有 144/128 个乘法器。',
            '三类利用率分别列分母，见 report.html。固定 8 个局部 SRAM 宏，无权重 SRAM；面积缺失保持 unknown。',
            'L0 单列于 arithmetic_baselines.json；不能替代详细周期。时序模型只运行元数据，未实现位精确定点 LN 或 RTL 校准。','']
    (path/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    (path/'report.html').write_text(_html(config,results,source,trace,digest),encoding='utf-8')
    return [str(item) for item in sorted(path.iterdir()) if item.is_file()]
