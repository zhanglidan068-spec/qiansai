"""40nm proxy area budget for the requested SMIC40-2025, 26x4 array.

Not a result from the SMIC library. Reads architecture constants only; no model
inference, dataset, labels or predictions are opened. Old runs are preserved.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from estimate_kwt_area import GE, PLANS, SOURCES


ROOT = Path(__file__).absolute().parent
ROWS, COLS = 26, 4
PE_COUNT = ROWS * COLS
NODE_SCALE = (40.0 / 45.0) ** 2
PROCESS_FACTORS = (.70, 1.0, 1.30)
ADDITIONAL_MACROS = {128: 44, 1024: 5, 4096: 0}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def endpoints(pair):
    return pair[0], sum(pair) / 2.0, pair[1]


def calculate():
    ge_area = [.798 * NODE_SCALE * factor for factor in PROCESS_FACTORS]
    parts = {'pe_array': [PE_COUNT * value for value in endpoints(GE['pe_per_instance'])]}
    for key in ('nonlinear', 'fp32_add_convert', 'dma_control', 'requant_channel_control'):
        parts[key] = list(endpoints(GE[key]))
    raw_ge = [sum(values[i] for values in parts.values()) for i in range(3)]
    allowance = endpoints(GE['clock_timing_allowance'])
    logic_ge = [raw_ge[i] * allowance[i] for i in range(3)]
    logic = [logic_ge[i] * ge_area[i] / 1e6 for i in range(3)]
    part_area = {key: [values[i] * ge_area[i] / 1e6 for i in range(3)]
                 for key, values in parts.items()}
    part_area['clock_timing_allowance'] = [logic[i] - raw_ge[i] * ge_area[i] / 1e6
                                           for i in range(3)]
    results, plans = [], []
    for base in PLANS:
        plan = dict(base)
        plan['base_plan'] = base['name']
        plan['name'] = base['name'].replace('27kib', '48kib').replace('107kib', '128kib').replace('155kib', '176kib')
        for depth, count in ADDITIONAL_MACROS.items():
            plan['macro_' + str(depth)] += count
        plans.append(plan)
        # Additional full-layer weight cache is optional, not part of the base.
        for weight_bits, extra_4096 in ((0, 0), (4, 3), (8, 6)):
            counts = {depth: plan['macro_' + str(depth)] + (extra_4096 if depth == 4096 else 0)
                      for depth in (128, 1024, 4096)}
            capacity = sum(counts[d] * d * 2 for d in counts)
            correction = endpoints(plan['correction45'])
            memory = [capacity * 8 / 826000.0 * NODE_SCALE * correction[i] * PROCESS_FACTORS[i]
                      for i in range(3)]
            total = [logic[i] + memory[i] for i in range(3)]
            result = dict(plan=plan['name'], pe_rows=ROWS, pe_cols=COLS, pe_count=PE_COUNT,
                          full_layer_weight_cache_bits=weight_bits,
                          full_layer_weight_cache_kib=extra_4096 * 8,
                          internal_capacity_kib=capacity / 1024.0,
                          macro_128=counts[128], macro_1024=counts[1024], macro_4096=counts[4096])
            for name, values in (('logic_mm2', logic), ('memory_mm2', memory), ('total_mm2', total)):
                for i, suffix in enumerate(('low', 'nominal', 'high')):
                    result[name + '_' + suffix] = values[i]
            result.update(assumed_range_below_limit=total[2] <= 2.25,
                          smic_library_used=False, synthesized=False,
                          area_compliance_verified=False, timing_verified=False)
            results.append(result)
    # M,N,K and count per encoder block: Q/K/V/O, FFN1/2, QK, AV.
    shapes = [(99, 64, 64, 4), (99, 256, 64, 1), (99, 64, 256, 1),
              (99, 99, 64, 1), (99, 64, 99, 1)]
    macs = 12 * sum(m * n * k * count for m, n, k, count in shapes)
    tile_cycles = 12 * sum(math.ceil(m / ROWS) * math.ceil(n / COLS) * k * count
                           for m, n, k, count in shapes)
    performance = dict(macs=macs, ideal_mac_cycles=math.ceil(macs / PE_COUNT),
                       ideal_mac_ms_at_150mhz=math.ceil(macs / PE_COUNT) / 150000.0,
                       ideal_rectangular_tile_cycles=tile_cycles,
                       ideal_rectangular_tile_ms_at_150mhz=tile_cycles / 150000.0,
                       internal_array_operand_bytes_per_active_cycle=ROWS + COLS,
                       internal_array_operand_gbyte_per_second_at_150mhz=(ROWS + COLS) * .15,
                       measured=False)
    return dict(results=results, plans=plans, logic_ge=logic_ge,
                ge_area_um2=ge_area, logic_parts_mm2=part_area, performance=performance)


def report(data):
    base_rows = [r for r in data['results'] if r['full_layer_weight_cache_bits'] == 0]
    lines = ['# SMIC40-2025目标、26×4阵列面积预算', '',
        '**结果是40nm公开参考缩放预算；没有使用SMIC40-2025实际标准单元库或存储宏。不能作为综合面积或达标证明。**', '',
        '指定工艺名称和阵列形状已记录。延续现有精度方案：104个signed INT8×INT8 PE，每PE含32-bit矩阵累加和必要寄存器；共享N16非线性单元、宽整数中间算术、FP32 bias/残差加法与定浮转换、重缩放、DMA和控制。适用于W8A8/P8F7或逐通道W4A8/P8F7。W4权重符号扩展进入同一INT8阵列，不使阵列面积自动减半。', '',
        '| 内部存储方案 | SRAM/ROM | 低值 mm² | 中间假设 mm² | 高值 mm² |',
        '|---|---:|---:|---:|---:|']
    labels = ['大张量使用既有共享SRAM', '内部缓存主要激活', '四张大张量均按FP32保存']
    for label, row in zip(labels, base_rows):
        lines.append('| {} | {:.0f} KiB | {:.3f} | {:.3f} | {:.3f} |'.format(
            label, row['internal_capacity_kib'], row['total_mm2_low'], row['total_mm2_nominal'], row['total_mm2_high']))
    lines += ['',
        '中间假设只是参数中点组合，不是统计期望；低/高值也不是置信区间或物理界。48KiB方案适合作为面积收敛起点；128KiB高值略超2.25mm²；176KiB预算跨过上限，不宜先假定达标。既有共享SRAM的容量、总线竞争和逐层权重装载仍需调度验证。', '',
        '## 104个PE和其余逻辑', '',
        '| 项目 | 低值 mm² | 中间假设 mm² | 高值 mm² |', '|---|---:|---:|---:|']
    names = {'pe_array': '104个INT8 MAC PE（含32-bit累加）', 'nonlinear': '共享N16及宽整数非线性',
             'fp32_add_convert': '共享FP32加法及定浮转换', 'dma_control': 'DMA、接口和控制',
             'requant_channel_control': '重缩放及逐通道F控制', 'clock_timing_allowance': '时钟及修复预留'}
    for key, values in data['logic_parts_mm2'].items():
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(names[key], *values))
    lines += ['',
        '`GE_logic = (104×GE_PE + GE_N16 + GE_FP32 + GE_DMA + GE_requant) × clock_factor`', '',
        '沿用上一预算的工程门数：每PE850–1500 GE；共享非线性12–35kGE，FP32加法/转换8–20kGE，DMA/控制10–30kGE，重缩放/通道控制5–15kGE；逻辑预留15%–30%。PE门数已含矩阵累加器，不再另加104份累加器重复计数。N16单元不是每PE复制一份，FP32也不是104个并行FPU。', '',
        '## 40nm参考面积参数', '',
        '公开Nangate45的NAND2_X1为0.798μm²。以 `0.798×(40/45)² = 0.630519 μm²/GE` 作参考中心，再乘0.70/1.00/1.30工艺转移因子，得到0.441363/0.630519/0.819674μm²/GE。这些是本次估算假设，不是SMIC40-2025库数据，也不代表2025版本具有某个已知面积。', '',
        '公开OpenRAM FreePDK45完整宏密度为0.826Mbit/mm²。参考模型：', '',
        '`A_memory = bits / 826000 × (40/45)² × small_macro_factor × process_factor`', '',
        '48KiB方案沿用多小宏修正1.25–2.00；128/176KiB沿用1.10–1.60；工艺因子0.70–1.30。完整宏密度本已包含外围，小宏修正针对尺寸/bank差异，不另叠加一次完整外围。方平方缩放、小宏因子及±30%均为工程假设；真实SMIC SP/DP宏、冗余、工艺选项可能超出区间。公开资料网络复查本次返回HTTP429，未取得新的SMIC库参数；两公开锚值沿用上一轮已核验的记录。', '',
        '## 为26行阵列增加的缓存', '',
        '假设26行×4列为输出驻留的GEMM tile：每个有效周期提供26个A8和4个B8，分别沿行/列广播；104个PE保留各自的32-bit部分和。若实际26×4采用K维归约树、每PE做多路乘法或不同数据流，必须重新核算。', '',
        '1. 激活tile按13个16-bit读bank供26条8-bit lane；双缓冲共26个16×128小宏。原预算8个，净增18个=4.5KiB。必须按同一k的两行配对打包/转置，普通行主序不自动满足此带宽；打包/地址逻辑预算在控制项中，具体RTL仍需验证。权重保留原8个小宏，不从面积表删掉未使用bank。',
        '2. score和probability各一张26×99×16-bit阵列，各5148B、分配5376B（5.25KiB），合计10.5KiB；相对旧4KiB净增26个小宏=6.5KiB。这是两个阵列槽，**不是score和prob各自都有ping/pong**。若两者各自双缓冲，还需10.5KiB及相应宏。',
        '3. FFN隐层26×256×16-bit有效13KiB，按7个16×1024宏分配14KiB。相对旧4KiB净增5个宏=10KiB。7个宏只证明容量，不能直接供26路N16读取；需分批量化/搬到13-bank A tile后计算。两组A tile共6.5KiB可存26×256个A8，亦可分两组K=128执行，但尚未证明装载与计算能无停顿重叠。当前容量表保留原存储预算，不假设跨阶段复用。', '',
        '合计净增21KiB，把上一版27/107/155KiB更新为48/128/176KiB。', '',
        '| 内部容量 | 16×128宏 | 16×1024宏 | 16×4096宏 |', '|---:|---:|---:|---:|']
    for row in base_rows:
        lines.append('| {:.0f} KiB | {} | {} | {} |'.format(row['internal_capacity_kib'], row['macro_128'], row['macro_1024'], row['macro_4096']))
    lines += ['',
        '共享exp/GELU查表按可装载SRAM宏估算，已经包含。26行后续结果可按N16原始整数码保存；首层残差输入FP32需要保留。K/V的FP32 bias与后续量化的先后次序必须与数值模型一致。所有参数逐层/逐tile装载；基础容量不含整层权重常驻，不含全模型权重常驻，也没有新增DDR控制器。', '',
        '## 可选整层权重缓存', '',
        '每层W4权重24KiB，可加3个16×4096宏；W8权重48KiB，可加6个。下面沿用各自基础方案的小宏修正，属于偏保守增量情景；换成真实大宏面积后应逐宏相加。', '',
        '| 基础方案 | 新增权重缓存 | 总内部容量 | 总面积低/中/高 mm² |',
        '|---|---:|---:|---:|']
    for row in data['results']:
        if row['full_layer_weight_cache_bits']:
            lines.append('| {} | W{}，{} KiB | {:.0f} KiB | {:.3f} / {:.3f} / {:.3f} |'.format(
                row['plan'], row['full_layer_weight_cache_bits'], row['full_layer_weight_cache_kib'],
                row['internal_capacity_kib'], row['total_mm2_low'], row['total_mm2_nominal'], row['total_mm2_high']))
    perf = data['performance']
    lines += ['', '## 性能含义及面积口径', '',
        '26×4广播供数需内部30B/有效周期，在150MHz对应4.5GB/s。这是tile到阵列的内部带宽，不能直接当作外部共享总线必须持续提供的带宽；外部需求取决于tile复用、装载/计算重叠和调度。', '',
        '12层共{:,}MAC。104PE满利用的纯MAC下界{:,}周期（{:.5f}ms）；计99行/列尾块的理想矩形tile为{:,}周期（{:.5f}ms）。均不含阵列填排空、非线性、GELU插值、LN开方/除法、FP32加法/转换、重排、DMA和停顿，未证明150MHz时序或端到端延迟。'.format(
            perf['macs'], perf['ideal_mac_cycles'], perf['ideal_mac_ms_at_150mhz'],
            perf['ideal_rectangular_tile_cycles'], perf['ideal_rectangular_tile_ms_at_150mhz']), '',
        '题目面积按新增映射标准单元加新增内部存储宏核算，既有E203/512KiB共享SRAM/固定总线按审计边界排除；新增接口仍计入。这里没有把总面积除以布局利用率；core/die面积、宏halo和布线空白另列。实际读端口/复制、ECC、CDC或更宽/更多并行单元可能增加面积。', '',
        '精确替换公式：`A_total = A_cells_from_SMIC40_synthesis + N128*A128 + N1024*A1024 + N4096*A4096`。宏若已包含在综合cell area里，只加一次。需SMIC40-2025标准单元.lib/.db、实际SP/DP宏LEF/面积和相同配置完整RTL，才能判断2.25mm²及150MHz合规。', '',
        '## 来源和复现', '',
        '- [Nangate45 NAND2_X1厂商Databook镜像](' + SOURCES['nangate45_nand2']['url'] + ')：0.798μm²/GE。',
        '- [OpenRAM官方结果](' + SOURCES['openram45_density']['url'] + ')：FreePDK45完整宏密度0.826Mbit/mm²。',
        '- 项目hardware_precision_spec.md及area_requirements_audit.json：模型规模、硬件边界、宏容量和赛题面积口径。', '',
        '本次新建独立run，脚本/输入架构常量副本及SHA256均保留，不改旧实验，不访问音频、标签或预测文件。', '',
        '```powershell', '& D:\\codex\\kwt_fp32\\.conda\\python.exe -B D:\\codex\\kwt_fp32\\estimate_kwt_smic40_26x4.py', '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    data = calculate()
    base_rows = [r for r in data['results'] if not r['full_layer_weight_cache_bits']]
    assert len(data['results']) == 9
    assert [r['internal_capacity_kib'] for r in base_rows] == [48, 128, 176]
    assert data['performance']['macs'] == 73446912
    assert data['performance']['ideal_rectangular_tile_cycles'] == 742656
    assert all(0 < r['total_mm2_low'] < r['total_mm2_nominal'] < r['total_mm2_high'] for r in data['results'])
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out = (args.out or ROOT / 'runs' / ('area_smic40_26x4_' + stamp)).absolute()
    out.mkdir(parents=True, exist_ok=False)
    data.update(kind='40nm_proxy_budget_not_smic_synthesis', requested_process='SMIC40-2025',
                area_limit_mm2=2.25, node_area_scale=NODE_SCALE, process_factors=PROCESS_FACTORS,
                engineering_ge_assumptions=GE, sources={key: SOURCES[key] for key in ('nangate45_nand2', 'openram45_density')},
                smic_library_used=False, testing_samples_used=0, model_inference_performed=False,
                old_experiments_modified=False, area_compliance_verified=False, timing_verified=False)
    dump(out / 'area_estimate.json', data)
    with (out / 'area_estimate.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['results'][0]))
        writer.writeheader()
        writer.writerows(data['results'])
    (out / 'README_面积估算.md').write_text(report(data), encoding='utf-8')
    source = out / 'source'
    source.mkdir()
    hashes = {}
    for name in ('estimate_kwt_smic40_26x4.py', 'estimate_kwt_area.py', 'hardware_precision_spec.md', 'area_requirements_audit.json'):
        shutil.copyfile(ROOT / name, source / name)
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    dump(out / 'verification.json', dict(checks_passed=True, result_count=9, source_sha256=hashes,
         macro_capacity_checked=True, prior_runs_preserved=True, physical_verification=False))
    print('AREA_ESTIMATE_DIRECTORY=' + str(out))
    for row in base_rows:
        print('{}: {:.3f} / {:.3f} / {:.3f} mm2'.format(row['plan'], row['total_mm2_low'], row['total_mm2_nominal'], row['total_mm2_high']))


if __name__ == '__main__':
    main()
