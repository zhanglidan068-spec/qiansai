"""Architecture area budget, not synthesis or official area qualification.

Standard library only. Does not read audio, labels, predictions, or old trials.
Public data anchor the assumptions; GE counts and SRAM corrections are estimates.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).absolute().parent
SOURCES = {
    'nangate45_nand2': {
        'url': 'https://www.cs.upc.edu/~jpetit/CellRouting/nangate/Front_End/Doc/Databook/Cells/NAND2_X1_NangateOpenCellLibrary_typical_typical.html',
        'verified_value': 'NAND2_X1 cell area 0.798 um2',
        'kind': 'vendor databook mirrored by university'},
    'openram45_density': {
        'url': 'https://github.com/VLSIDA/OpenRAM/blob/stable/docs/source/results.md',
        'verified_value': 'FreePDK45 generated memory density 0.826 Mbit/mm2',
        'kind': 'compiler authors results; not measurements of our 16-bit macros'},
    'intel65_bitcell': {
        'url': 'https://doi.org/10.1109/IEDM.2004.1419253',
        'verified_value': '65 nm process SRAM 6T bitcell 0.57 um2',
        'kind': 'original process paper; bitcell excludes macro peripherals'},
    'sky130_macro_comparison': {
        'url': 'https://escholarship.org/uc/item/9dc0v8g3',
        'verified_value': 'SRAM Design with OpenRAM in SkyWater 130nm; macro size and port examples',
        'kind': 'qualitative cross-check of peripheral and port costs only'},
}

# Macro counts are capacity/banking budgets, not a demonstrated RTL schedule.
PLANS = [
    dict(name='shared_sram_27kib', label='共享SRAM搬运，内部仅tile/ROM',
         macro_128=44, macro_1024=4, macro_4096=1,
         correction45=[1.25, 2.0],
         note='X/K/V reside in existing shared SRAM; extra traffic and stalls must be timed'),
    dict(name='internal_107kib', label='内部激活缓存，首层X为FP32',
         macro_128=44, macro_1024=44, macro_4096=1,
         correction45=[1.1, 1.6],
         note='First residual FP32; later N16 outputs can be stored losslessly as N16 codes'),
    dict(name='conservative_155kib', label='四张大张量均按FP32保存',
         macro_128=44, macro_1024=68, macro_4096=1,
         correction45=[1.1, 1.6],
         note='Storage upper scenario; preserving every intermediate FP32 tensor is unnecessary'),
]

GE = dict(pe_per_instance=[850, 1500], nonlinear=[12000, 35000],
          fp32_add_convert=[8000, 20000], dma_control=[10000, 30000],
          requant_channel_control=[5000, 15000], clock_timing_allowance=[1.15, 1.30])


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def calculate(efficiency=(.20, .40)):
    if not 0 < efficiency[0] <= efficiency[1] < 1:
        raise ValueError('SRAM array efficiency must satisfy 0 < low <= high < 1')
    rows = []
    for plan in PLANS:
        capacity = sum(plan['macro_' + str(depth)] * depth * 2 for depth in (128, 1024, 4096))
        bits = capacity * 8
        for pe in (16, 32, 64):
            logic_ge = []
            for end in (0, 1):
                fixed = sum(GE[key][end] for key in ('nonlinear', 'fp32_add_convert', 'dma_control', 'requant_channel_control'))
                logic_ge.append((pe * GE['pe_per_instance'][end] + fixed) * GE['clock_timing_allowance'][end])
            for node in (45, 65):
                if node == 45:
                    ge_area = [.798, .798]
                    memory = [bits / 826000. * factor for factor in plan['correction45']]
                else:
                    nominal_ge = .798 * (65. / 45.) ** 2
                    ge_area = [nominal_ge * .70, nominal_ge * 1.30]
                    memory = [bits * .57 / efficiency[1] / 1e6,
                              bits * .57 / efficiency[0] / 1e6]
                logic = [logic_ge[end] * ge_area[end] / 1e6 for end in (0, 1)]
                total = [logic[end] + memory[end] for end in (0, 1)]
                rows.append(dict(node_nm_scenario=node, plan=plan['name'], pe_count=pe,
                    macro_128=plan['macro_128'], macro_1024=plan['macro_1024'], macro_4096=plan['macro_4096'],
                    internal_capacity_kib=capacity / 1024., logic_ge_low=logic_ge[0], logic_ge_high=logic_ge[1],
                    logic_mm2_low=logic[0], logic_mm2_high=logic[1],
                    sram_rom_mm2_low=memory[0], sram_rom_mm2_high=memory[1],
                    total_mm2_low=total[0], total_mm2_high=total[1],
                    full_assumed_range_below_2_25=total[1] <= 2.25,
                    synthesized=False, area_compliance_verified=False, timing_verified=False))
    return rows


def report(rows, efficiency):
    lines = ['# KWT 加速器常见工艺面积粗估', '',
        '**这是架构预算，不是综合结果。没有证明题目指定工艺面积或150 MHz时序达标。**', '',
        '面向已验证的W8A8/P8F7或逐通道W4A8/P8F7、N16编码器方案，统一采用共享signed INT8 PE、32bit矩阵累加器、共享多周期非线性单元。编码器中的FP32 bias/残差加法与定浮转换已计入逻辑预算。', '',
        '## 32个PE的主要结果', '',
        '| 内部存储方案 | 容量 | 45nm公开参考情景 / mm² | 65nm工程情景 / mm² |',
        '|---|---:|---:|---:|']
    for plan in PLANS:
        values = [next(row for row in rows if row['pe_count'] == 32 and row['plan'] == plan['name'] and row['node_nm_scenario'] == node) for node in (45, 65)]
        lines.append('| {} | {:.0f} KiB | {:.2f}–{:.2f} | {:.2f}–{:.2f} |'.format(
            plan['label'], values[0]['internal_capacity_kib'], values[0]['total_mm2_low'], values[0]['total_mm2_high'], values[1]['total_mm2_low'], values[1]['total_mm2_high']))
    lines += ['', '表中两种工艺使用不同公开资料和假设，不是同一设计移植后的工艺缩放实测。上下界是条件预算，不是置信区间或承诺上界。', '',
        '## 逻辑模型', '',
        '`GE = (PE数量 × 每PE门当量 + 共享固定逻辑门当量) × 时钟/时序预留`', '',
        '每个PE假设850–1500 GE；共享N16单元12–35kGE，FP32加法/转换8–20kGE，DMA/控制10–30kGE，输出重缩放和通道F控制5–15kGE。逻辑预留15%–30%，已在结果中加入。GE以一个二输入NAND为1门当量，这些门数是工程假设，未由RTL综合获得。', '',
        '45nm采用Nangate NAND2_X1的0.798 μm²/GE。65nm暂以 `0.798×(65/45)²` 为中心再取±30%；这是跨工艺投影假设，不是65nm实际库数据。面积单位μm²除以1,000,000得到mm²。', '',
        '## 存储模型与外围成本', '',
        '45nm：`A_memory = 容量bit / 826000 × 小宏修正系数`。OpenRAM的0.826 Mbit/mm²已经是完整宏密度；27KiB小宏多，修正取1.25–2.0；另两方案取1.1–1.6。这些修正不是实测宏表，不再重复添加一次完整外围。', '',
        '65nm：`A_memory = 容量bit × 0.57 / η / 1e6`。0.57 μm²仅为公开65nm 6T bitcell；η取{:.2f}–{:.2f}，表示阵列有效面积占完整宏面积的比例。该效率是面向小宏、bank、译码、读写、冗余和间隙的工程预算，已经计入外围，不另加通用外围面积。实际双端口、bank复制、测试/冗余及工艺库差异可能超出该范围。'.format(*efficiency), '',
        '| 缓存方案 | 16×128宏 | 16×1024宏 | 16×4096宏 | 容量 |', '|---|---:|---:|---:|---:|']
    for plan in PLANS:
        size = sum(plan['macro_' + str(d)] * d * 2 for d in (128, 1024, 4096)) / 1024.
        lines.append('| {} | {} | {} | {} | {:.0f} KiB |'.format(plan['label'], plan['macro_128'], plan['macro_1024'], plan['macro_4096'], size))
    lines += ['',
        '107KiB修正自原85.75KiB样例：首层FP32残差输入增加16KiB；8行score/prob双缓冲增加3.5KiB；每层576个F按宏取整增加0.75KiB；DMA FIFO增加1KiB。后续层的N16输出可按原始码无损保存；K/V在FP32 bias后立即按下一矩阵格式量化，必须按真实数值顺序验证。155KiB情景把另外三张量也扩为FP32，是宽裕存储情景。', '',
        '27KiB方案把大张量放在题目既有共享512KiB SRAM，只新增tile、行、共享查表、FIFO。共享SRAM排除其已有物理面积，新增接口、控制和内部宏仍计入。首层FP32 X、其余三个N16张量的逻辑负载约61.875KiB；再加一层W8权重48KiB约109.875KiB，容量上可行，但总线竞争、装载和重复读取必须计任务时间。', '',
        '两张共享非线性ROM以可装载SRAM估算，已包含在容量内。每层权重/F逐层装载，不把全模型W4的288KiB、W8的576KiB全部算成新增内部SRAM；每层F只需576B，宏容量768B。若全层权重常驻，W4/W6/W8分别额外增加24/36/48KiB；若全部F常驻，在已有层F预算上净增6KiB。', '',
        '## 如何选架构', '',
        '先以32PE、27KiB内部缓存为面积收敛起点，再测共享SRAM带宽和总周期。若搬运成本太高，再比较64PE与107KiB内部缓存。32PE在150MHz下、理想单遍纯MAC下界为15.30144ms；64PE为7.65072ms，均未包含非线性、访存、浮点转换和停顿，不能作为最终性能结论。', '',
        'W4和W8若共用同一INT8乘法阵列和相同tile缓存，逻辑/内部宏面积基本相同，W4主要降低装载量。W4不会自动把芯片面积减半；若保留整层权重缓存，两者的容量差才会直接影响内部SRAM面积。QK/AV两侧仍是激活，不能只用原生4bit权重乘法器替代所有INT8运算。', '',
        '面积口径为新增标准单元面积加新增内部存储宏面积；既有E203、共享SRAM与固定总线按赛题排除。布局core面积还含放置利用率、宏halo和布线空白，不能把cell+macro总面积统一除以利用率后与题目口径混用。实际新增双端口/多读端口、复制、复杂DMA、ECC、接口或更宽组合算术会超出此预算，届时必须更新模型。', '',
        '## 来源与复现', '',
        '以下公开数值用于锚定参数；所有GE数量、效率、小宏修正和65nm缩放均是本次工程假设：', '']
    for key, source in SOURCES.items():
        lines.append('- [{}]({})：{}。'.format(key, source['url'], source['verified_value']))
    lines += ['', '同目录area_estimate.csv/json保存16/32/64 PE全部18组结果与假设；source/保存本次脚本。运行不访问数据集，不改变旧精度实验，也不重新执行模型。', '',
        '```powershell', '& D:\\codex\\kwt_fp32\\.conda\\python.exe -B D:\\codex\\kwt_fp32\\estimate_kwt_area.py', '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--sram65-efficiency-low', type=float, default=.20)
    parser.add_argument('--sram65-efficiency-high', type=float, default=.40)
    args = parser.parse_args()
    efficiency = (args.sram65_efficiency_low, args.sram65_efficiency_high)
    rows = calculate(efficiency)
    assert len(rows) == 18
    assert {row['internal_capacity_kib'] for row in rows} == {27., 107., 155.}
    assert all(0 < row['total_mm2_low'] <= row['total_mm2_high'] for row in rows)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out = (args.out or ROOT / 'runs' / ('area_estimate_' + stamp)).absolute()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'area_estimate.json', dict(kind='engineering_budget_not_synthesis', area_limit_mm2=2.25,
         sources=SOURCES, engineering_ge_assumptions=GE, memory_plans=PLANS,
         sram65_array_efficiency_range=efficiency, results=rows,
         testing_samples_used=0, area_compliance_verified=False, timing_verified=False))
    with (out / 'area_estimate.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'README_面积粗估.md').write_text(report(rows, efficiency), encoding='utf-8')
    source = out / 'source'
    source.mkdir()
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    dump(out / 'verification.json', dict(checks_passed=True, result_count=len(rows), macro_capacity_sums_checked=True,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         all_lower_bounds_positive_and_ordered=True, old_experiments_modified=False))
    print('AREA_ESTIMATE_DIRECTORY=' + str(out))
    for row in rows:
        if row['pe_count'] == 32:
            print('{} nm {}: {:.3f}--{:.3f} mm2'.format(row['node_nm_scenario'], row['plan'], row['total_mm2_low'], row['total_mm2_high']))


if __name__ == '__main__':
    main()
