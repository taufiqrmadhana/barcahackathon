"""Render vector research figures from aggregate metrics; never read match assets."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

BLUE, ORANGE, GRAY, INK = '#185B80', '#B45E22', '#98A5AE', '#1F2D38'


def finish(fig, axes, folder, name):
    for ax in axes:
        ax.tick_params(length=0, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight', pad_inches=.06,
                metadata={'Title': name.replace('_', ' ').title(), 'Author': 'Borrow Khan'})
    fig.savefig(folder / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def ablation(report, folder):
    original = next(x for x in report['sensitivity']['results']
                    if x['folds'] == 5 and x['objective'] == 'baseline')
    motion = next(x for x in report['experiments']['motion_weight_trials']
                  if x['motion_weight'] == .5 and x['prune_maximum'] == 100
                  and x['prune_confidence'] == .5)
    trial = {x['mode']: x for x in report['experiments']['temporal_without_dead_blend']}
    final = report['checkpoint']['public']
    rows = [
        ('Previous primary (v4)', original['macroF1'], original['spellF1']),
        ('Motion mixture alone', motion['macroF1'], motion['spellF1']),
        ('v4 + oscillation removal', trial['oscillation']['macroF1'], trial['oscillation']['spellF1']),
        ('v4 + short-spell agreement', trial['consensus']['macroF1'], trial['consensus']['spellF1']),
        ('v4 + both temporal steps', trial['both']['macroF1'], trial['both']['spellF1']),
        ('Final: both + P(D) average', final['macro_f1'], final['spell_f1_mean']),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), sharey=True)
    y = np.arange(len(rows))
    for ax, index, title, color in [(axes[0], 1, 'Macro-F1', BLUE),
                                     (axes[1], 2, 'Spell-F1', ORANGE)]:
        values = [row[index] for row in rows]
        ax.barh(y, values, height=.56, color=[GRAY] * 5 + [color], zorder=3)
        for i, value in enumerate(values):
            ax.text(value + .025, i, f'{value:.6f}', va='center', fontsize=8.2,
                    color=INK, weight='bold' if i == 5 else 'normal')
        ax.set_xlim(0, 1.23)
        ax.set_xticks([0, .25, .5, .75, 1.])
        ax.set_title(title, fontsize=10, loc='left', color=color, weight='bold', pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E6EBEF', linewidth=.6)
    axes[0].set_yticks(y, [row[0] for row in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.265, right=.985, top=.84, bottom=.13, wspace=.17)
    finish(fig, axes, folder, 'ablation')
    return [dict(configuration=x[0], macroF1=x[1], spellF1=x[2]) for x in rows]


def sensitivity(report, folder):
    rows = report['sensitivity']['results']
    folds = sorted({x['folds'] for x in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 1.85))
    data = dict(folds=folds)
    for ax, metric, title, limits in [(axes[0], 'macroF1', 'Macro-F1', (.92, .97)),
                                      (axes[1], 'spellF1', 'Spell-F1', (.65, .90))]:
        for key, name, color, marker in [('baseline', 'Previous (v4)', GRAY, 'o'),
                                          ('candidate', 'Final', BLUE, 's')]:
            scores = [next(x[metric] for x in rows if x['folds'] == f and x['objective'] == key)
                      for f in folds]
            data[f'{metric}_{key}'] = scores
            ax.plot(folds, scores, color=color, marker=marker, markersize=4,
                    linewidth=1.6, label=name, zorder=3)
        ax.set_title(title, fontsize=10, loc='left', weight='bold')
        ax.set_xticks(folds)
        ax.set_xlabel('Time folds', fontsize=8)
        ax.set_ylim(*limits)
        ax.grid(axis='y', color='#E6EBEF', linewidth=.6)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='upper right', ncol=2,
               frameon=False, fontsize=8, handlelength=1.4, bbox_to_anchor=(.985,1.06))
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.26, wspace=.27)
    finish(fig, axes, folder, 'fold_sensitivity')
    return data


def match_stats(report, folder):
    checkpoint = report['checkpoint']
    rows = [(f'Half {x["half"]}', x) for x in checkpoint['halves']]
    rows.append(('Both halves', checkpoint['full']))
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 1.85), sharey=True)
    home = np.array([x['home_alive_share'] for _, x in rows])
    axes[0].barh(y, home, height=.52, color=BLUE, label='Home')
    axes[0].barh(y, 1-home, left=home, height=.52, color=ORANGE, label='Away')
    for i, h in enumerate(home):
        axes[0].text(h/2, i, f'{100*h:.2f}%', va='center', ha='center', color='white', fontsize=8.5)
        axes[0].text((1+h)/2, i, f'{100*(1-h):.2f}%', va='center', ha='center', color='white', fontsize=8.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([0, .5, 1])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title('Share of alive time', loc='left', fontsize=10, weight='bold')
    for i, (_, x) in enumerate(rows):
        h, a = x['teams']['H']['possessions'], x['teams']['A']['possessions']
        axes[1].barh(i, h, height=.52, color=BLUE)
        axes[1].barh(i, a, left=h, height=.52, color=ORANGE)
        axes[1].text(h/2, i, str(h), va='center', ha='center', color='white', fontsize=8.5)
        axes[1].text(h+a/2, i, str(a), va='center', ha='center', color='white', fontsize=8.5)
        axes[1].text(h+a+8, i, str(h+a), va='center', fontsize=8.5, color=INK, weight='bold')
    axes[1].set_xlim(0, 455)
    axes[1].set_xticks([0, 200, 400])
    axes[1].set_title('Predicted possession spells', loc='left', fontsize=10, weight='bold')
    axes[0].set_yticks(y, [name for name, _ in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(.53, -.015))
    fig.subplots_adjust(left=.13, right=.97, top=.80, bottom=.28, wspace=.25)
    finish(fig, axes, folder, 'match_statistics')
    return [dict(window=name, home_alive_share=x['home_alive_share'],
                 home_spells=x['teams']['H']['possessions'],
                 away_spells=x['teams']['A']['possessions'], total_spells=x['possessions'])
            for name, x in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': '#536573', 'ytick.color': INK,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    data = dict(ablation=ablation(report, args.out), sensitivity=sensitivity(report, args.out),
                match_statistics=match_stats(report, args.out),
                source_report='src/bench/results/' + args.report.name, uses_aggregate_report_only=True)
    (args.out / 'data.json').write_text(json.dumps(data, indent=2) + '\n')
    print('Rendered three vector figures from the aggregate checkpoint report.')


if __name__ == '__main__':
    main()
