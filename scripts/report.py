import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from nanocode.common import get_model_dir

# -----------------------------------------------------------------------------
# log parsing

def parse_tokens_per_step(lines):
    batch_size, seq_len, world_size = None, None, None
    for line in lines:
        m = re.match(r'  batch_size: (\d+)', line)
        if m:
            batch_size = int(m.group(1))
        m = re.match(r'  config: GPTConfig\(sequence_len=(\d+)', line)
        if m:
            seq_len = int(m.group(1))
        m = re.match(r'[Ww]orld.size: (\d+)', line)
        if m:
            world_size = int(m.group(1))
    if batch_size and seq_len and world_size:
        return batch_size * seq_len * world_size
    return None

def parse_training_steps(lines):
    steps, loss, tkps, mfu = [], [], [], []
    for line in lines:
        m = re.match(r'Step: (\d+)/\d+ \| Loss: ([\d.]+) .* tkps: (\d+) \| mfu: ([\d.]+)', line)
        if m:
            steps.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            tkps.append(int(m.group(3)))
            mfu.append(float(m.group(4)))
    return steps, {'loss': loss, 'tkps': tkps, 'mfu': mfu}

def parse_memory(lines):
    steps, used = [], []
    prev_step = 0
    for line in lines:
        m = re.match(r'Step: (\d+)/', line)
        if m:
            prev_step = int(m.group(1))
        m = re.match(r'\tPeak bytes reserved/limit: ([\d.]+)/([\d.]+)', line)
        if m:
            steps.append(prev_step)
            used.append(float(m.group(1)))
    return steps, {'memory (GB)': used}

def parse_base_bpb(lines):
    steps, fwe, sv2, avg = [], [], [], []
    prev_step = 0
    for line in lines:
        m = re.match(r'Step: (\d+)/', line)
        if m:
            prev_step = int(m.group(1))
        m = re.match(r'\tfwe_bpb: ([\d.]+) \| sv2_bpb: ([\d.]+) \| avg_bpb: ([\d.]+)', line)
        if m:
            steps.append(prev_step)
            fwe.append(float(m.group(1)))
            sv2.append(float(m.group(2)))
            avg.append(float(m.group(3)))
    return steps, {'fwe_bpb': fwe, 'sv2_bpb': sv2, 'avg_bpb': avg}

def parse_core(lines):
    steps, core = [], []
    prev_step = 0
    for line in lines:
        m = re.match(r'Step: (\d+)/', line)
        if m:
            prev_step = int(m.group(1))
        m = re.match(r'  CORE metric: ([\d.]+)', line)
        if m:
            steps.append(prev_step)
            core.append(float(m.group(1)))
    return steps, {'CORE': core}

def parse_sft_bpb(lines):
    steps, rollout, chat, avg = [], [], [], []
    prev_step = 0
    for line in lines:
        m = re.match(r'Step: (\d+)/', line)
        if m:
            prev_step = int(m.group(1))
        m = re.match(r'\trollout_bpb: ([\d.]+) \| chat_bpb: ([\d.]+) \| avg_bpb: ([\d.]+)', line)
        if m:
            steps.append(prev_step)
            rollout.append(float(m.group(1)))
            chat.append(float(m.group(2)))
            avg.append(float(m.group(3)))
    return steps, {'rollout_val_bpb': rollout, 'chat_val_bpb': chat, 'avg_val_bpb': avg}

def parse_dpo_metrics(lines):
    steps, loss, acc, margins, chosen, rejected = [], [], [], [], [], []
    for line in lines:
        m = re.match(r'Step: (\d+)/\d+ \| Loss: ([\d.]+) \| Acc: ([\d.]+) \| Margins: ([-\d.]+) \| Rewards \(chosen/rejected\): ([-\d.]+)/([-\d.]+)', line)
        if m:
            steps.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            acc.append(float(m.group(3)))
            margins.append(float(m.group(4)))
            chosen.append(float(m.group(5)))
            rejected.append(float(m.group(6)))
    return steps, {'loss': loss, 'accuracy': acc, 'margins': margins, 'chosen_reward': chosen, 'rejected_reward': rejected}

def parse_dpo_eval(lines):
    steps, bpb, val_loss, val_acc, val_margins = [], [], [], [], []
    prev_step = 0
    for line in lines:
        m = re.match(r'Step: (\d+)/', line)
        if m:
            prev_step = int(m.group(1))
        m = re.match(r'\tval bpb: ([\d.]+) \| val loss: ([\d.]+) \| val acc: ([\d.]+) \| val margins: ([-\d.]+)', line)
        if m:
            steps.append(prev_step)
            bpb.append(float(m.group(1)))
            val_loss.append(float(m.group(2)))
            val_acc.append(float(m.group(3)))
            val_margins.append(float(m.group(4)))
    return steps, {'rollout_val_bpb': bpb, 'val_loss': val_loss, 'val_acc': val_acc, 'val_margins': val_margins}

def parse_training_time(lines):
    for line in lines:
        m = re.match(r'Total training time: ([\d.]+)min', line)
        if m:
            return float(m.group(1))
    return None

def parse_tok_train(path):
    for line in path.read_text().split('\n'):
        m = re.match(r'Training time: ([\d.]+)s', line)
        if m:
            return float(m.group(1))
    return None

def parse_tok_eval(path):
    return re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())

def parse_eval_csv(path):
    tasks = {}
    for line in path.read_text().strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3 and parts[0] != 'Task':
            name = parts[0].strip()
            acc = float(parts[1]) if parts[1] else None
            cen = float(parts[2]) if parts[2] else None
            tasks[name] = (acc, cen)
    return tasks

def parse_eval_bpb(path):
    for line in path.read_text().split('\n'):
        m = re.match(r'fwe_bpb: ([\d.]+) \| sv2_bpb: ([\d.]+) \| avg_bpb: ([\d.]+)', line)
        if m:
            return {'fwe_bpb': float(m.group(1)), 'sv2_bpb': float(m.group(2)), 'avg_bpb': float(m.group(3))}
    return None

METRIC_LINE = re.compile(r'^\t(fwe_bpb|sv2_bpb|rollout_bpb|chat_bpb|val bpb|Peak bytes)')

def parse_generations(lines):
    # generations are multiline: start with \t<|bos|>, continue on subsequent
    # lines until the next \t<|bos|>, a Step: line, or a metric line.
    # we collect all batches (between Step: lines) and keep the last one.
    all_batches = []
    current_batch = []
    current_gen = None
    for line in lines:
        if re.match(r'Step: \d+/', line):
            if current_gen is not None:
                current_batch.append(current_gen)
                current_gen = None
            if current_batch:
                all_batches.append(current_batch)
                current_batch = []
        elif line.startswith('\t<|bos|>'):
            if current_gen is not None:
                current_batch.append(current_gen)
            current_gen = line.strip()
        elif current_gen is not None:
            if METRIC_LINE.match(line) or line.startswith('Evaluating:') or line.startswith('Total training') or line.startswith('Model ('):
                current_batch.append(current_gen)
                current_gen = None
            else:
                current_gen += '\n' + line
    if current_gen is not None:
        current_batch.append(current_gen)
    if current_batch:
        all_batches.append(current_batch)
    return all_batches[-1] if all_batches else []

# -----------------------------------------------------------------------------
# plotting

def plot(steps, series, ylabel, title, path, xlabel='step'):
    fig, ax = plt.subplots(figsize=(8, 4))
    if xlabel == 'tokens':
        steps = [s / 1e9 for s in steps]
        xlabel = 'tokens (B)'
    for label, values in series.items():
        ax.plot(steps, values, linewidth=0.8, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# phase configs

PHASES = {
    'base': {
        'log': 'base_log.txt',
        'evals': [('base_eval', 'eval')],
        'parsers': [
            (parse_training_steps, [('loss', 'loss', 'loss')]),
            (parse_base_bpb, [('*', 'bpb', 'bpb')]),
            (parse_core, [('*', 'CORE', 'core')]),
            (parse_training_steps, [('mfu', 'MFU (%)', 'mfu'), ('tkps', 'tkps', 'tkps')]),
            (parse_memory, [('*', 'GB', 'memory')]),
        ],
    },
    'sft': {
        'log': 'chat_sft_log.txt',
        'evals': [],
        'parsers': [
            (parse_training_steps, [('loss', 'loss', 'loss')]),
            (parse_sft_bpb, [('*', 'bpb', 'bpb')]),
            (parse_training_steps, [('mfu', 'MFU (%)', 'mfu'), ('tkps', 'tkps', 'tkps')]),
            (parse_memory, [('*', 'GB', 'memory')]),
        ],
    },
    'dpo': {
        'log': 'dpo_log.txt',
        'evals': [],
        'parsers': [
            (parse_dpo_metrics, [('loss', 'loss', 'loss'), ('accuracy', 'accuracy', 'acc'), ('margins', 'margin', 'margins'),
                                  ({'chosen_reward', 'rejected_reward'}, 'reward', 'rewards')]),
            (parse_dpo_eval, [('*', 'value', 'val')]),
            (parse_training_steps, [('mfu', 'MFU (%)', 'mfu'), ('tkps', 'tkps', 'tkps')]),
            (parse_memory, [('*', 'GB', 'memory')]),
        ],
    },
}

def generate_phase_report(phase, model_dir, report_dir):
    cfg = PHASES[phase]
    log_path = model_dir / cfg['log']
    if not log_path.exists():
        return None
    lines = log_path.read_text().split('\n')
    plots = []
    all_series = {}

    # for base phase, plot x-axis in tokens instead of steps
    tokens_per_step = parse_tokens_per_step(lines) if phase == 'base' else None
    xlabel = 'tokens' if tokens_per_step else 'step'

    for parser_fn, plot_specs in cfg['parsers']:
        steps, series = parser_fn(lines)
        if not steps:
            continue
        all_series.update(series)
        x_axis = [s * tokens_per_step for s in steps] if tokens_per_step else steps
        for key_spec, ylabel, name in plot_specs:
            if key_spec == '*':
                plot_series = series
            elif isinstance(key_spec, set):
                plot_series = {k: v for k, v in series.items() if k in key_spec}
            else:
                plot_series = {key_spec: series[key_spec]}
            path = report_dir / f'{phase}_{name}.png'
            plot(x_axis, plot_series, ylabel, f'{phase} {name}', path, xlabel=xlabel)
            plots.append((name, path))

    training_time = parse_training_time(lines)

    # collect eval results for this phase
    evals = []
    for eval_dir, eval_name in cfg['evals']:
        eval_path = model_dir / eval_dir
        if not eval_path.exists():
            continue
        for csv_file in sorted(eval_path.glob('*.csv')):
            tasks = parse_eval_csv(csv_file)
            txt_file = csv_file.with_suffix('.txt')
            bpb = parse_eval_bpb(txt_file) if txt_file.exists() else None
            if tasks:
                evals.append({'name': eval_name, 'tasks': tasks, 'bpb': bpb})

    generations = parse_generations(lines)
    return {'name': phase, 'plots': plots, 'training_time': training_time, 'evals': evals, 'series': all_series, 'generations': generations}

def collect_final_bpb(sections):
    bpb_rows = []
    for s in sections:
        for es in s.get('evals', []):
            if es.get('bpb'):
                bpb_rows.append((s['name'], es['bpb']))
    return bpb_rows

REPORT_CSS = """\
<style>
body { max-width: 960px; margin: 0 auto; padding: 2em; font-family: sans-serif; }
table { border-collapse: collapse; margin: 1em 0; }
td, th { padding: 6px 12px; border: 1px solid #ccc; }
th { background: #f5f5f5; }
img { max-width: 100%; }
pre { background: #f5f5f5; padding: 1em; overflow-x: auto; }
</style>
"""

def fmt_metric(name, value):
    if name == 'tkps': return f'{int(value)}'
    return f'{value:.4f}'

def write_report(sections, tok_info, report_dir, report_path):
    with open(report_path, 'w') as f:
        f.write(REPORT_CSS + '\n')
        motd_path = Path('motd.txt')
        if motd_path.exists():
            f.write('```\n')
            f.write(motd_path.read_text())
            f.write('```\n\n')
        f.write('# nanocode training report\n\n')

        # table of contents
        f.write('## contents\n\n')
        if tok_info:
            f.write('- [tokenizer](#tokenizer)\n')
        base = next((s for s in sections if s['name'] == 'base'), None)
        if base and base.get('evals'):
            f.write('- [eval](#eval)\n')
        f.write('- [plots](#plots)\n')
        for s in sections:
            f.write(f'  - [{s["name"]}](#{s["name"]})\n')
            for name, _ in s['plots']:
                f.write(f'    - [{s["name"]} {name}](#{s["name"]}-{name})\n')
        has_gens = any(s.get('generations') for s in sections)
        if has_gens:
            f.write('- [sample generations](#sample-generations)\n')
        f.write('\n')

        # summary table
        f.write('## summary\n\n')
        f.write(f'| {"phase":<10} | {"time (min)":<12} | {"tkps":<10} | {"MFU":<10} |\n')
        f.write(f'|{"-"*12}|{"-"*14}|{"-"*12}|{"-"*12}|\n')
        if tok_info and tok_info.get('training_time'):
            t = f"{tok_info['training_time'] / 60:.1f}"
            f.write(f'| {"tokenizer":<10} | {t:<12} | {"-":<10} | {"-":<10} |\n')
        for s in sections:
            t = f"{s['training_time']:.1f}" if s['training_time'] else '-'
            ss = s.get('series', {})
            tkps = f'{int(ss["tkps"][-1])}' if 'tkps' in ss and ss['tkps'] else '-'
            mfu = f'{ss["mfu"][-1]:.2f}' if 'mfu' in ss and ss['mfu'] else '-'
            f.write(f'| {s["name"]:<10} | {t:<12} | {tkps:<10} | {mfu:<10} |\n')
        total = sum(s['training_time'] for s in sections if s['training_time'])
        if tok_info and tok_info.get('training_time'):
            total += tok_info['training_time'] / 60
        f.write(f'| {"total":<10} | {total:.1f}{"":<6} | {"":<10} | {"":<10} |\n')
        f.write('\n')

        # pretrain summary
        if base and base.get('series'):
            f.write('### pretrain\n\n')
            bs = base['series']
            bpb_rows = collect_final_bpb(sections)
            f.write(f'| {"metric":<15} | {"value":<10} |\n')
            f.write(f'|{"-"*17}|{"-"*12}|\n')
            for _, bpb in bpb_rows:
                f.write(f'| {"fineweb bpb":<15} | {bpb["fwe_bpb"]:<10.4f} |\n')
                f.write(f'| {"the stack v2 bpb":<15} | {bpb["sv2_bpb"]:<10.4f} |\n')
                f.write(f'| {"avg bpb":<15} | {bpb["avg_bpb"]:<10.4f} |\n')
            for es in base.get('evals', []):
                core = es['tasks'].get('CORE')
                if core and core[1] is not None:
                    f.write(f'| {"CORE":<15} | {core[1]:<10.4f} |\n')
            for m in ['MFU', 'tkps']:
                k = m.lower()
                if k in bs and bs[k]:
                    f.write(f'| {m:<15} | {fmt_metric(k, bs[k][-1]):<10} |\n')
            f.write('\n')

        # sft summary
        sft = next((s for s in sections if s['name'] == 'sft'), None)
        if sft and sft.get('series'):
            f.write('### sft\n\n')
            ss = sft['series']
            f.write(f'| {"metric":<15} | {"last step":<10} |\n')
            f.write(f'|{"-"*17}|{"-"*12}|\n')
            for m in ['loss', 'tkps', 'MFU']:
                k = m.lower()
                if k in ss and ss[k]:
                    f.write(f'| {m:<15} | {fmt_metric(k, ss[k][-1]):<10} |\n')
            for m in ['rollout_val_bpb', 'chat_val_bpb', 'avg_val_bpb']:
                if m in ss and ss[m]:
                    f.write(f'| {m:<15} | {ss[m][-1]:<10.4f} |\n')
            f.write('\n')

        # dpo summary
        dpo = next((s for s in sections if s['name'] == 'dpo'), None)
        if dpo and dpo.get('series'):
            f.write('### dpo\n\n')
            ds = dpo['series']
            f.write(f'| {"metric":<15} | {"first step":<12} | {"last step":<12} |\n')
            f.write(f'|{"-"*17}|{"-"*14}|{"-"*14}|\n')
            for m in ['loss', 'accuracy', 'margins']:
                if m in ds and ds[m]:
                    f.write(f'| {m:<15} | {ds[m][0]:<12.4f} | {ds[m][-1]:<12.4f} |\n')
            if 'rollout_val_bpb' in ds and ds['rollout_val_bpb']:
                f.write(f'| {"rollout_val_bpb":<15} | {ds["rollout_val_bpb"][0]:<12.4f} | {ds["rollout_val_bpb"][-1]:<12.4f} |\n')
            for m in ['MFU', 'tkps']:
                k = m.lower()
                if k in ds and ds[k]:
                    f.write(f'| {m:<15} | {fmt_metric(k, ds[k][0]):<12} | {fmt_metric(k, ds[k][-1]):<12} |\n')
            f.write('\n')

        # tokenizer section
        if tok_info:
            f.write('## tokenizer\n\n')
            if tok_info.get('training_time'):
                f.write(f'training time: {tok_info["training_time"]:.1f}s\n\n')
            if tok_info.get('eval_text'):
                f.write('```\n')
                f.write(tok_info['eval_text'])
                f.write('```\n\n')

        # eval section (base eval tables)
        if base and base.get('evals'):
            f.write('## eval\n\n')
            for es in base['evals']:
                if es.get('bpb'):
                    b = es['bpb']
                    f.write(f'| {"metric":<10} | {"value":<10} |\n')
                    f.write(f'|{"-"*12}|{"-"*12}|\n')
                    f.write(f'| {"fwe_bpb":<10} | {b["fwe_bpb"]:<10.4f} |\n')
                    f.write(f'| {"sv2_bpb":<10} | {b["sv2_bpb"]:<10.4f} |\n')
                    f.write(f'| {"avg_bpb":<10} | {b["avg_bpb"]:<10.4f} |\n')
                    f.write('\n')
                f.write(f'| {"task":<35} | {"accuracy":<10} | {"centered":<10} |\n')
                f.write(f'|{"-"*37}|{"-"*12}|{"-"*12}|\n')
                for task, (acc, cen) in es['tasks'].items():
                    acc_s = f'{acc:.4f}' if acc is not None else '-'
                    cen_s = f'{cen:.4f}' if cen is not None else '-'
                    f.write(f'| {task:<35} | {acc_s:<10} | {cen_s:<10} |\n')
                f.write('\n')

        # plots section - all phases under one heading
        f.write('## plots\n\n')
        for s in sections:
            f.write(f'### {s["name"]}\n\n')
            for name, p in s['plots']:
                f.write(f'#### {s["name"]} {name}\n\n')
                rel = p.relative_to(report_path.parent)
                f.write(f'![]({rel})\n\n')

        # sample generations
        if has_gens:
            f.write('## sample generations\n\n')
            for s in sections:
                gens = s.get('generations', [])
                if not gens:
                    continue
                f.write(f'### {s["name"]}\n\n')
                for g in gens:
                    f.write(f'```\n{g}\n```\n\n')

    print(f'report written to {report_path}')

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='generate nanocode training report')
parser.add_argument('--model-dir', type=str, default=None)
args = parser.parse_args()

model_dir = Path(args.model_dir) if args.model_dir else get_model_dir()
report_dir = Path('reports') / model_dir.name
report_dir.mkdir(parents=True, exist_ok=True)

# tokenizer info
tok_info = {}
tok_train_path = model_dir / 'tok_train.txt'
if tok_train_path.exists():
    tok_info['training_time'] = parse_tok_train(tok_train_path)
tok_eval_path = model_dir / 'tok_eval.txt'
if tok_eval_path.exists():
    tok_info['eval_text'] = parse_tok_eval(tok_eval_path)
if not tok_info:
    tok_info = None

sections = []
for phase in PHASES:
    result = generate_phase_report(phase, model_dir, report_dir)
    if result:
        sections.append(result)

report_path = report_dir / 'report.md'
write_report(sections, tok_info, report_dir, report_path)
