"""Export separate evaluation tables and genuine data-size learning curves.

Reads aggregate checkpoint reports only. Does not access match assets, refit a
classifier, change a submission or modify the IEEE report.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, GRAY, INK = '#185B80', '#B45E22', '#98A5AE', '#1F2D38'


def write_csv(path, headers, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def table(fig, box, title, headers, rows, widths=None, fontsize=10):
    ax = fig.add_axes(box)
    ax.set_axis_off()
    ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=11, color=BLUE)
    cells = ax.table(cellText=rows, colLabels=headers, loc='upper left',
                     cellLoc='left', colLoc='left', colWidths=widths, bbox=[0, 0, 1, .94])
    cells.auto_set_font_size(False)
    cells.set_fontsize(fontsize)
    for (row, column), cell in cells.get_celld().items():
        cell.set_edgecolor('white')
        cell.PAD = .07
        cell.set_facecolor('#EDF3F6' if row % 2 else '#F7F9FA')
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color='white', weight='bold')
    return ax


def error_counts(report):
    c = np.array(report['checkpoint']['public']['confusion'])
    return [int(c[0, 1]+c[1, 0]), int(c[0, 2]+c[1, 2]), int(c[2, 0]+c[2, 1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'src/submission/build/evaluation')
    parser.add_argument('--report', type=Path, default=ROOT / 'src/bench/results/macro_v11_2026-11-17.json')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'src/bench/results/macro_v10_2026-11-17.json')
    parser.add_argument('--learning-report', type=Path, default=ROOT / 'src/bench/results/macro_v11_learning_curve_2026-11-17.json')
    args = parser.parse_args()
    paths = dict(checkpoint=args.report, baseline=args.baseline, learning_curve=args.learning_report)
    report, baseline, learning = (json.loads(paths[k].read_text())
                                  for k in ('checkpoint', 'baseline', 'learning_curve'))
    public = report['checkpoint']['public']
    profile = Path(report['params']['hybrid_model']).parent.name
    baseline_profile = Path(baseline['params']['hybrid_model']).parent.name
    full_count = report['checkpoint']['full']['possessions']
    assert learning['baseline']['macroF1'] == public['macro_f1']
    assert learning['baseline']['spellF1'] == public['spell_f1_mean']
    assert learning['model_sha256'] == report['checkpoint']['artifact_sha256']
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    transition = next(value for key, value in public['transitions'].items() if int(key) == 25)
    metrics = [('Macro-F1 (H/A/D)', public['macro_f1']),
               ('Spell-F1 (IoU 0.3–0.7)', public['spell_f1_mean']),
               ('Frame accuracy', public['accuracy']),
               ('Transition F1 (within 1 s)', transition['f1'])]
    scope = [('Public window (300 s)', public['n_gt_spells'], public['n_pred_spells'],
              '14 matched; 3 GT missed; 0 spurious'),
             ('Full graded match', 'Unknown', report['checkpoint']['full']['possessions'],
              'Predicted counts; no full-match accuracy')]
    classes = [(name, row['precision'], row['recall'], row['f1'], row['support'])
               for name, row in public['per_class'].items()]
    write_csv(output / 'public_evaluation.csv', ['metric', 'value'], metrics)
    write_csv(output / 'public_class_metrics.csv', ['class', 'precision', 'recall', 'f1', 'gt_frames'], classes)
    write_csv(output / 'scope_and_counts.csv', ['scope', 'gt_spells', 'predicted_spells', 'interpretation'], scope)
    sensitivity = []
    for folds in sorted({r['folds'] for r in report['sensitivity']['results']}):
        a, b = (next(r for r in report['sensitivity']['results']
                     if r['folds'] == folds and r['objective'] == key) for key in ('baseline', 'candidate'))
        sensitivity.append([folds, a['macroF1'], b['macroF1'], a['spellF1'], b['spellF1'],
                            a['possessions'], b['possessions']])
    write_csv(output / 'kfold_sensitivity.csv', ['time_folds', 'baseline_macroF1', 'final_macroF1',
        'baseline_spellF1', 'final_spellF1', 'baseline_predicted_spells', 'final_predicted_spells'], sensitivity)
    fold_rows = [[r['fold'], r['frames'], *r['gt_class_counts'], r['accuracy'],
                  r['macroF1_3classes'], *r['class_f1']] for r in learning['per_fold_final']]
    write_csv(output / 'five_fold_breakdown.csv', ['fold', 'frames', 'gt_H', 'gt_A', 'gt_D',
        'accuracy', 'macroF1_3classes', 'f1_H', 'f1_A', 'f1_D'], fold_rows)
    curve = [[r['fraction'], r['train_rows']['mean'], r['train_rows']['std'],
        r['train_log_loss']['mean'], r['train_log_loss']['std'],
        r['validation_log_loss']['mean'], r['validation_log_loss']['std'],
        r['pipeline_macroF1'], r['pipeline_spellF1'], r['pipeline_accuracy'],
        r['pipeline_predicted_spells']] for r in learning['results']]
    write_csv(output / 'learning_curve.csv', ['fraction', 'mean_train_rows', 'std_train_rows',
        'mean_train_log_loss', 'std_train_log_loss', 'mean_validation_log_loss', 'std_validation_log_loss',
        'pipeline_macroF1', 'pipeline_spellF1', 'pipeline_accuracy', 'pipeline_predicted_spells'], curve)
    errors = list(zip(['Wrong H/A team', 'Alive predicted D', 'D predicted alive'],
                      error_counts(baseline), error_counts(report)))
    write_csv(output / 'frame_errors.csv', ['error_type', 'baseline_frames', 'final_frames'], errors)

    with PdfPages(output / 'evaluation_tables.pdf') as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(f'Borrow Khan — evaluation of {profile}', x=.07, ha='left',
                     y=.965, fontsize=17, weight='bold', color=BLUE)
        table(fig, [.07, .725, .86, .18], 'Public metrics: 7,500 frames', ['Metric', 'Score'],
              [[name, f'{value:.6f}'] for name, value in metrics], widths=[.72, .28])
        table(fig, [.07, .50, .86, .16], 'Class-level metrics', ['Class', 'Precision', 'Recall', 'F1', 'GT frames'],
              [[name, f'{p:.6f}', f'{r:.6f}', f'{f:.6f}', str(n)] for name, p, r, f, n in classes])
        table(fig, [.07, .285, .86, .14], 'Keep evaluation scopes separate', ['Window', 'GT spells', 'Predicted spells'],
              [[row[0], str(row[1]), str(row[2])] for row in scope], widths=[.48, .22, .30])
        table(fig, [.07, .08, .86, .14], 'Error tradeoff on public frames', ['Error type', baseline_profile, 'Final'],
              [[name, str(a), str(b)] for name, a, b in errors], widths=[.56, .22, .22])
        fig.text(.07, .035, 'Public: 17 GT possessions; 14 predictions, all matched; 3 GT possessions missed.\n'
                 f'Full match: {full_count} predicted possessions. Private/unseen-match accuracy is unknown.', fontsize=9)
        pdf.savefig(fig)
        fig.savefig(output / 'evaluation_summary.png', dpi=180)
        plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle('Time-fold evaluation', x=.07, ha='left', y=.965,
                     fontsize=17, weight='bold', color=BLUE)
        table(fig, [.07, .70, .86, .18], 'Sensitivity to the number of time folds',
            ['K', 'Macro: base', 'Macro: final', 'Spell: base', 'Spell: final'],
            [[str(r[0])]+[f'{v:.6f}' for v in r[1:5]] for r in sensitivity], fontsize=9.5)
        table(fig, [.07, .37, .86, .22], 'Individual folds of the deployed five-fold model',
            ['Fold', 'GT H/A/D', 'Accuracy', 'Macro (3)', 'F1 H/A/D'],
            [[str(r['fold']), '/'.join(map(str, r['gt_class_counts'])), f"{r['accuracy']:.4f}",
              f"{r['macroF1_3classes']:.4f}", '/'.join(f'{v:.3f}' for v in r['class_f1'])]
             for r in learning['per_fold_final']], widths=[.08, .26, .17, .17, .32], fontsize=8.5)
        fig.text(.07, .245, 'All runs repartition the same 17 public possessions; these are not independent matches.\n'
            'Each individual fold contains 1,500 frames. The training purge is 125 frames (5 s).\n'
            'Macro (3) always averages H/A/D; an absent class with zero F1 still contributes zero.\n'
            'Folds 1 and 3 contain no GT D frames; fold 4 contains only 33 GT H frames.\n'
            'The official public Macro-F1 is computed after pooling held-out predictions,\n'
            'not by averaging these per-fold F1 scores.', fontsize=9, linespacing=1.65)
        pdf.savefig(fig)
        fig.savefig(output / 'kfold_results.png', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25))
    fig.suptitle(f'Phase learning curve — {learning["feature_count"]} features, C={learning["C"]} (training data size)',
                 x=.07, ha='left', y=.98, fontsize=12, weight='bold', color=BLUE)
    x = np.array([r['fraction']*100 for r in learning['results']])
    for key, label, color in [('train_log_loss', 'Training', BLUE),
                               ('validation_log_loss', 'Held-out', ORANGE)]:
        mean = np.array([r[key]['mean'] for r in learning['results']])
        std = np.array([r[key]['std'] for r in learning['results']])
        axes[0].plot(x, mean, marker='o', color=color, label=label)
        axes[0].fill_between(x, np.maximum(0, mean-std), mean+std, color=color, alpha=.12)
    axes[0].set_title('Phase expert: alive/dead log loss', loc='left', weight='bold')
    axes[0].set_ylabel('Log loss (lower is better)')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9)
    for key, label, color in [('pipeline_macroF1', 'Macro-F1', BLUE),
                               ('pipeline_spellF1', 'Spell-F1', ORANGE)]:
        values = [r[key] for r in learning['results']]
        axes[1].plot(x, values, marker='o', color=color, label=label)
        for xi, value in zip(x, values):
            axes[1].annotate(f'{value:.6f}', (xi, value), xytext=(0, 8 if key == 'pipeline_macroF1' else -14),
                             textcoords='offset points', ha='center', fontsize=8, color=color)
    axes[1].set_title('Final H/A/D pipeline: public scores', loc='left', weight='bold')
    axes[1].set_ylim(.84, .99)
    axes[1].legend(frameon=False, fontsize=9, loc='center left')
    for ax in axes:
        ticks = [f"{int(row['fraction']*100)}%\n(n≈{row['train_rows']['mean']:.0f})"
                 for row in learning['results']]
        ax.set_xticks(x, ticks)
        ax.set_xlabel('Phase training subset (mean rows per fold)')
        ax.grid(axis='y', color='#E6EBEF')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.07, .025, 'Only phase heads are refitted; other members/decoder stay fixed. Shading: ±1 fold SD, not a CI.\n'
        'CSV output has H/A/D/X; public Macro-F1 scores H/A/D. X on a graded frame counts as a miss.', fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.28, wspace=.28)
    fig.savefig(output / 'learning_curve.png', dpi=200)
    fig.savefig(output / 'learning_curve.pdf')
    plt.close(fig)
    (output / 'learning_curve_details.json').write_text(json.dumps(learning, indent=2)+'\n')
    (output / 'README.md').write_text(
        '# Paket evaluasi Borrow Khan (terpisah dari laporan IEEE)\n\n'
        'Public window 5 menit: 17 possession GT, 14 prediksi, 14 cocok, 3 terlewat, 0 spurious. '
        f'Angka {full_count} adalah prediksi seluruh pertandingan, bukan GT public.\n\n'
        'evaluation_tables.pdf: tabel metrik, kelas, scope, error, sensitivitas K, dan setiap fold. '
        'Semua tabel juga tersedia sebagai CSV.\n\n'
        'learning_curve.png/.pdf: kurva ukuran data latih, bukan kurva epoch. Model yang diuji '
        f'adalah {learning["model"]}, {learning["feature_count"]} fitur, C={learning["C"]}. Lima model lain dan decoder '
        'tetap. Pool ada 301 baris native 1 Hz; setelah held-out fold dan purge, 100% berarti '
        'rata-rata 232.8 baris latih per fold. Subset 25/50/75/100% bersarang dan stratified '
        'di dalam data latih, dengan seed 20261117. Preprocessing hanya di-fit di subset itu.\n\n'
        'Panel kiri: binary log loss train vs held-out pada baris native, mean ± SD lima fold. '
        'Panel kanan: Macro-F1 H/A/D dan Spell-F1 pipeline pada semua 7.500 prediksi held-out '
        'yang digabung. Keduanya tugas/metrik yang berbeda. SD bukan confidence interval. '
        'Output CSV tetap H/A/D/X: model binary hanya expert alive/dead; primary memprediksi H/A/D. '
        'X di luar window graded dikecualikan; prediksi X pada GT H/A/D tetap dihitung salah. '
        'Baca pola loss dan skor pada nilai terukur di CSV; kurva tidak dipaksakan monoton. '
        'Ini tidak membuktikan konvergensi atau generalisasi ke match lain.\n\n'
        'Tabel individual fold selalu menghitung Macro-F1 tiga kelas, termasuk F1=0 untuk kelas '
        'yang absen. Karena itu fold 1/3 terlihat rendah meskipun akurasi tinggi. Macro-F1 resmi '
        f'{public["macro_f1"]:.6f} dihitung dari seluruh prediksi OOF gabungan, bukan rata-rata skor per fold. '
        'Semua repartisi memakai public slice yang sama; skor private belum diketahui. '
        'Hasil kurva tidak mengubah submission; 100% mereproduksi seluruh label checkpoint.\n')
    manifest = dict(source_reports={key:dict(path=str(path.relative_to(ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key, path in paths.items()},
        csv_files=sorted(p.name for p in output.glob('*.csv')),
        public_gt_spells=public['n_gt_spells'], public_predicted_spells=public['n_pred_spells'],
        full_match_predicted_spells=report['checkpoint']['full']['possessions'],
        separate_from_ieee_report=True, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = output.parent / 'Borrow_Khan_Evaluation.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    print(f'Evaluation tables: {output / "evaluation_tables.pdf"}')
    print(f'Learning curve: {output / "learning_curve.png"}')
    print(f'Bundle: {archive}')


if __name__ == '__main__':
    main()
