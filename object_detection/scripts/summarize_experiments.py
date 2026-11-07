"""Build an evidence-based experiment index; never invent missing metrics."""
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/04_交付汇总'
BASE = ROOT / 'reports/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/independent_validation_20261022/metrics.json'
PROTOCOL = ['images', 'instances', 'image_size', 'batch_size', 'half',
            'validated_paths_sha256', 'data_yaml_sha256', 'confidence_threshold',
            'nms_iou_threshold', 'max_detections', 'augmented_inference']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def link(path):
    return os.path.relpath(str(path), str(OUT)).replace('\\', '/')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = read(BASE)
    records = []
    for metric_path in sorted((ROOT / 'reports').glob('*/*/*/metrics.json')):
        metric = read(metric_path)
        experiment = metric_path.parent.parent
        course = experiment.parent.name in ['01_普通模型达标训练', '02_模型优化']
        comparison = course and all(metric.get(k) == baseline.get(k) for k in PROTOCOL)
        record = dict(experiment=experiment.name, group=experiment.parent.name,
                      validation=metric_path.parent.name, metrics=str(metric_path),
                      weights=metric['weights'], weights_sha256=metric['weights_sha256'],
                      current_weights_match=Path(metric['weights']).is_file() and sha(metric['weights']) == metric['weights_sha256'],
                      images=metric['images'], instances=metric['instances'],
                      precision=metric['precision'], recall=metric['recall'],
                      map50=metric['map50'], map50_95=metric['map50_95'],
                      speed_ms_per_image=metric.get('speed_ms_per_image'),
                      per_class=metric['per_class'], course=course,
                      same_protocol_as_baseline=comparison,
                      improvement_percentage_points=(metric['map50'] - baseline['map50']) * 100 if comparison else None)
        run = Path(metric['weights']).parent.parent
        launch_path = experiment / 'launch.json'
        if launch_path.exists():
            launch = read(launch_path)
            record.update(method=launch.get('method', launch.get('intended_change', 'training')),
                          launch=str(launch_path), data=launch.get('data'),
                          dataset_snapshot=launch.get('dataset_snapshot'))
            for key in ['parameters', 'component_count', 'fraction_b']:
                if key in launch:
                    record[key] = launch[key]
        csv_path = run / 'results.csv'
        if csv_path.exists():
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                history = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]
            if history:
                best = max(history, key=lambda r: float(r['metrics/mAP_0.5']))
                record.update(completed_epochs=len(history), best_training_epoch_zero_based=int(best['epoch']),
                              best_training_map50=float(best['metrics/mAP_0.5']))
        records.append(record)

    # Compare actual image and label bytes with the manifest made before training.
    manifest_path = ROOT / 'reports/00_环境与数据核查/20261022_094416/course_manifest.json'
    changed = []
    for item in read(manifest_path):
        image = ROOT / item['file']
        label = ROOT / 'mydata/labels' / item['split'] / (image.stem + '.txt')
        for path, expected in [(image, item['image_sha256']), (label, item['labels_sha256'])]:
            if not path.is_file() or sha(path) != expected:
                changed.append(str(path))

    course_candidates = [r for r in records if r['course'] and r['same_protocol_as_baseline'] and r['current_weights_match']]
    best_course = max(course_candidates, key=lambda r: r['map50'])
    best_single = max((r for r in course_candidates if r.get('method') != 'prediction_ensemble'),
                      key=lambda r: r['map50'])
    scored_experiments = {str(Path(r['metrics']).parent.parent) for r in records}
    unscored = []
    for launch_path in sorted((ROOT / 'reports').glob('*/*/launch.json')):
        if str(launch_path.parent) not in scored_experiments:
            launch = read(launch_path)
            unscored.append(dict(experiment=launch_path.parent.name, launch=str(launch_path),
                                 status=launch.get('status'), returncode=launch.get('returncode')))
    result = dict(generated=datetime.now().astimezone().isoformat(), records=records,
                  experiments_without_formal_metrics=unscored,
                  baseline_metrics=str(BASE), best_course_experiment=best_course['experiment'],
                  best_course_map50=best_course['map50'],
                  best_single_network_experiment=best_single['experiment'],
                  best_single_network_map50=best_single['map50'],
                  course_data_unchanged_since_baseline=not changed, changed_course_files=changed,
                  strict_manual_LabelImg_annotation_complete=False,
                  annotation_note='v1: fixed bootstrap regions. v2: AI visual keyframes, interpolation and local image refinements. Neither is completed image-by-image human LabelImg annotation.',
                  course_evaluation_note='Fixed validation set reused for model selection; not an untouched course test set.')
    (OUT / 'experiment_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_recommendation = dict(experiment=best_course['experiment'], weights=best_course['weights'],
                                sha256=best_course['weights_sha256'], map50=best_course['map50'],
                                metrics=best_course['metrics'],
                                improvement_percentage_points=best_course['improvement_percentage_points'],
                                method=best_course.get('method'),
                                component_count=best_course.get('component_count', 1),
                                single_network_alternative=dict(experiment=best_single['experiment'],
                                    weights=best_single['weights'], sha256=best_single['weights_sha256'],
                                    map50=best_single['map50'], metrics=best_single['metrics']),
                                note='Highest observed fixed course validation AP50; not proof of statistical significance.')
    (OUT / '推荐模型信息.json').write_text(json.dumps(write_recommendation, ensure_ascii=False, indent=2), encoding='utf-8')
    professional_v2 = [r for r in records if not r['course']
                       and (r.get('data') or '').startswith('separation_dataset_v2_1000/')
                       and '_val_' in r['validation'] and r['current_weights_match']]
    professional_note = '尚无专业v2正式val成绩，不能推荐。'
    if professional_v2:
        chosen = max(professional_v2, key=lambda r: r['map50'])
        component_candidate = max(professional_v2, key=lambda r: next(
            c['ap50'] for c in r['per_class'] if c['name'] == 'separation_component'))
        professional = dict(experiment=chosen['experiment'], weights=chosen['weights'],
                            weights_sha256=chosen['weights_sha256'], val_map50=chosen['map50'],
                            val_metrics=chosen['metrics'],
                            component_focus_candidate=dict(experiment=component_candidate['experiment'],
                                weights=component_candidate['weights'], metrics=component_candidate['metrics'],
                                val_component_ap50=next(c['ap50'] for c in component_candidate['per_class']
                                                        if c['name'] == 'separation_component')),
                            selection='Selected using v2 validation AP50 only; test is descriptive, not used to rank candidates.',
                            annotation_limit='AI keyframe assisted annotations; manual LabelImg refinement remains incomplete.')
        (OUT / '专业v2推荐模型信息.json').write_text(json.dumps(professional, ensure_ascii=False, indent=2), encoding='utf-8')
        professional_note = ('专业v2按两类平均val AP50选择`' + chosen['experiment'] +
                             '`；部件重点候选见[专业v2推荐模型信息](专业v2推荐模型信息.json)，不按test排名。')

    # Preserve existing professional annotations BEFORE any future human edits.
    snapshot = OUT / '专业训练所用标签快照.zip'
    if not snapshot.exists():
        data = ROOT / 'separation_dataset_1000'
        hashes = []
        with zipfile.ZipFile(snapshot, 'x', zipfile.ZIP_DEFLATED) as archive:
            for filename in ['manifest.json', 'dataset_summary.json', 'classes.txt',
                             'separation_dataset.yaml', 'separation_dataset_test.yaml']:
                archive.write(data / filename, filename)
            for split in ['train', 'val', 'test']:
                for image in sorted((data / 'images' / split).glob('*.jpg')):
                    label = data / 'labels' / split / (image.stem + '.txt')
                    archive.write(label, str(label.relative_to(data)))
                    hashes.append(dict(image=str(image.relative_to(data)), image_sha256=sha(image),
                                       label=str(label.relative_to(data)), label_sha256=sha(label)))
            archive.writestr('file_hashes.json', json.dumps(hashes, ensure_ascii=False, indent=2))

    lines = ['# 实验结果与文件索引', '', '本页由真实验证报告生成；重新运行汇总脚本会更新本页。', '',
             '日期：' + result['generated'], '',
             '**专业v1是固定粗引导框；专业v2已改为关键帧辅助框和局部图像校正。两版均未完成逐图LabelImg人工精修，成绩只反映各自参考标签上的探索性实验。**', '',
             '| 实验 | 验证图/框 | P | R | mAP50 | mAP50:95 | 对普通模型的提升（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        delta = '不适用' if r['improvement_percentage_points'] is None else f"{r['improvement_percentage_points']:+.4f}"
        lines.append(f"| {r['experiment']} / {r['validation']} | {r['images']}/{r['instances']} | {r['precision']*100:.4f}% | {r['recall']*100:.4f}% | **{r['map50']*100:.4f}%** | {r['map50_95']*100:.4f}% | {delta} |")
    lines += ['', '课程数据与训练前清单逐文件核对：' + ('完全一致。' if not changed else '发现变化，查看 JSON。'), '',
              '按同一评价条件，目前最高课程模型：`' + best_course['experiment'] + '`。', '',
              '[直接找到推荐权重](' + link(Path(best_course['weights'])) + ')；[推荐模型信息](推荐模型信息.json)。', '',
              '最高单网络方案：`' + best_single['experiment'] + '`，AP50为' + f"{best_single['map50']*100:.4f}%" +
              '；[单网络权重](' + link(Path(best_single['weights'])) + ')。预测集成保留多个完整网络，推理与存储成本增加。', '',
              professional_note, '',
              '## 每份结果在哪里', '']
    for r in records:
        lines += ['### ' + r['experiment'] + ' / ' + r['validation'], '',
                  '- [正式指标 JSON](' + link(Path(r['metrics'])) + ')',
                  '- [模型权重](' + link(Path(r['weights'])) + ')',
                  '- [PR 曲线](' + link(Path(r['metrics']).parent / 'plots/PR_curve.png') + ')',
                  '- [混淆矩阵](' + link(Path(r['metrics']).parent / 'plots/confusion_matrix.png') + ')', '',
                  '当前权重与验证时的 SHA256：' + ('一致。' if r['current_weights_match'] else '不一致，请检查。'), '']
        if 'completed_epochs' in r:
            lines += [f"实际 CSV 有 {r['completed_epochs']} 轮；训练日志最佳轮是 epoch={r['best_training_epoch_zero_based']}（日常称第 {r['best_training_epoch_zero_based']+1} 轮）。训练日志与独立验证的浮点结果可能略有差别。", '']
        elif r.get('method') in ['weight_average', 'prediction_ensemble']:
            lines += ['本项由已有模型组合生成，没有额外训练轮次，因此没有训练CSV或训练曲线。方法：`' + r['method'] + '`。', '']
        if not r['course']:
            lines += ['专业数据版本：`' + str(r.get('data')) + '`。val用于选模，test用于结果描述；不同数据版本的分数不直接比较。', '']
            if r.get('dataset_snapshot'):
                lines += ['[本次训练前冻结的标签快照](' + link(Path(r['dataset_snapshot']['archive'])) + ')。', '']
        if r.get('speed_ms_per_image'):
            speeds = r['speed_ms_per_image']
            lines += [f"本次日志每图平均耗时：预处理{speeds[0]:.3f} ms，网络推理{speeds[1]:.3f} ms，NMS{speeds[2]:.3f} ms。它是本次运行记录，不是严格的硬件性能基准。", '']
        if r.get('method') == 'prediction_ensemble':
            lines += [f"该权重包含{r.get('component_count')}个完整检测器，总参数{r.get('parameters')}；增加了推理与存储成本。", '']
        lines += ['| 类别 | 真实框数 | AP50 | AP50:95 |', '|---|---:|---:|---:|']
        for c in r['per_class']:
            lines.append(f"| {c['name']} | {c['instances']} | {c['ap50']*100:.4f}% | {c['ap50_95']*100:.4f}% |")
        lines += ['']
    if unscored:
        lines += ['## 没有正式分数的运行', '',
                  '以下运行保留日志用于排错，没有metrics.json，不能当作已取得成绩的模型。', '']
        for item in unscored:
            lines += ['- `' + item['experiment'] + '`：状态`' + str(item['status']) + '`；[运行记录](' + link(Path(item['launch'])) + ')。']
        lines += ['']
    lines += ['## 保留的数据版本', '',
              '`专业训练所用标签快照.zip` 保存历史专业v1标签。v2的每次训练在对应report目录保存`training_labels_snapshot.zip`，其中有标签、配置、manifest和图像/标签SHA；没有重复压缩全部视频和图片。', '',
              '将来人工修改标签后，旧权重的报告仍对应这个快照，不能把旧分数直接当成新标签上的分数。', '',
              '## 比较图', '', '![课程模型比较](course_map50_comparison.png)', '',
              '![课程相对普通的百分点差值](course_map50_delta.png)', '',
              '![小幅微调差值详图](course_map50_delta_detail.png)', '',
              '[reports逐文件用途清单](reports逐文件用途清单.csv)：按相对路径列出所有已有报告文件及用途。', '',
              '上图只放相同课程数据与评价协议的模型。专业数据的结果应单独阅读。', '']
    (OUT / '实验结果与文件索引.md').write_text('\n'.join(lines), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [r['experiment'].split('_', 1)[0] for r in course_candidates]
    values = [r['map50'] * 100 for r in course_candidates]
    bars = ax.bar(names, values, color=['#2563eb' if r['experiment'] == best_course['experiment'] else '#94a3b8' for r in course_candidates])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.4f}%', ha='center', fontsize=10)
    ax.axhline(91, color='#dc2626', ls='--', lw=1, label='Required baseline: 91%')
    ax.set(ylim=(0, 103), ylabel='mAP@0.5 (%)', xlabel='Experiment number (see table)', title='Course validation: same 640-pixel evaluation protocol')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(str(OUT / 'course_map50_comparison.png'), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    deltas = [r['improvement_percentage_points'] for r in course_candidates]
    bars = ax.bar(names, deltas, color=['#2563eb' if d > 0 else '#94a3b8' for d in deltas])
    for bar, value in zip(bars, deltas):
        ax.annotate(f'{value:+.4f}', (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 4 if value >= 0 else -12), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(0, color='black', lw=1)
    ax.set(ylabel='Absolute change in percentage points', xlabel='Experiment number (see table)',
           title='Course AP50 change relative to baseline (including failed P2)')
    fig.tight_layout()
    fig.savefig(str(OUT / 'course_map50_delta.png'), dpi=160)
    plt.close(fig)
    # A separate view shows small refinements without hiding the failed P2 in
    # the overview. Its scale is labelled in absolute percentage points.
    near = [r for r in course_candidates if abs(r['improvement_percentage_points']) < 1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    small = [r['improvement_percentage_points'] for r in near]
    names_near = [r['experiment'].split('_', 1)[0] for r in near]
    bars = ax.bar(names_near, small, color=['#2563eb' if d > 0 else '#94a3b8' for d in small])
    for bar, value in zip(bars, small):
        ax.annotate(f'{value:+.4f}', (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 4 if value >= 0 else -12), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(0, color='black', lw=1)
    ax.set(ylim=(min(small) - .1, max(small) + .1), ylabel='Absolute change in percentage points',
           xlabel='Experiment number; P2 shown in overview',
           title='Refinement differences below 1 percentage point (not significance)')
    fig.tight_layout()
    fig.savefig(str(OUT / 'course_map50_delta_detail.png'), dpi=160)
    plt.close(fig)
    # Map every existing report file to its purpose, including historical files.
    explanations = {
        'metrics.json': '正式总指标、逐类AP、权重指纹与验证设置',
        'launch.json': '启动命令、起点权重、时间、状态与退出码',
        'comparison.json': '与普通模型的同协议比较及百分点差',
        'weight_transfer.json': '原张量继承、初始输出一致性及参数量',
        'dataset_audit.json': '数据格式/配对/指纹/分组检查，不认证语义正确',
        'environment.json': 'Python、库版本、GPU和导入检查',
        'course_dataset.json': '课程图数/框数/类别与备份处理摘要',
        'course_manifest.json': '课程逐图及标签指纹清单',
        'dataset_summary.json': '专业图数/框数/方法与局限摘要',
        'PR_curve.png': 'Precision-Recall曲线与AP',
        'P_curve.png': 'Precision随置信度变化',
        'R_curve.png': 'Recall随置信度变化',
        'F1_curve.png': 'F1随置信度变化',
        'confusion_matrix.png': '预测类别与参考类别/背景的匹配统计',
        'course_map50_comparison.png': '同课程协议各方案AP50总览',
        'course_map50_delta.png': '含失败P2的相对普通绝对百分点差总览',
        'course_map50_delta_detail.png': '差值不足1个百分点的细图，非显著性证明',
        'tutorial_validation_v6.json': '第六版教程实际链接/语法/命令与最终证据检查范围',
        'tutorial_validation_v7.json': '第七版教程实际链接/命令、图片和结果证据检查范围',
        'tutorial_validation_v8.json': '第八版教程实际链接/命令、图片和最终结果证据检查范围',
        'reports逐文件用途清单.csv': '本逐文件用途索引；自身大小列留空，避免自引用',
    }
    files = sorted(p for p in (ROOT / 'reports').rglob('*') if p.is_file())
    with (OUT / 'reports逐文件用途清单.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['相对工程路径', '字节', '用途'])
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            purpose = explanations.get(path.name)
            if purpose is None:
                if path.name.startswith('val_batch'):
                    purpose = '参考标签拼图' if '_labels' in path.name else '模型预测拼图（部分批次）'
                elif '/all_labels/' in relative:
                    purpose = '专业辅助参考框预览，每页最多15张；不是模型预测'
                elif '/raw_keyframes/' in relative:
                    purpose = '原视频关键帧，无检测框'
                elif path.suffix == '.log':
                    purpose = '训练/验证终端日志；失败时查Traceback与退出码'
                elif path.suffix == '.zip':
                    purpose = '源码快照' if '源码' in path.name else '标签/数据版本快照，保留训练答案'
                elif path.suffix == '.pt':
                    purpose = '初始模型或保存的模型参数，结合所属目录理解'
                elif path.suffix in ['.yaml', '.yml']:
                    purpose = '实际结构/超参数配置'
                elif path.suffix == '.md':
                    purpose = '说明、审查记录或自动生成的结果索引'
                elif path.suffix in ['.jpg', '.png']:
                    purpose = '检查图/比较图/教学图，结合文件名和第八版第8章阅读'
                elif path.suffix == '.json':
                    purpose = '结构化记录或检查结果，结合字段与所属目录阅读'
                else:
                    purpose = '附属检查/环境记录，结合第八版第8章阅读'
            writer.writerow([relative, '' if path.name == 'reports逐文件用途清单.csv' else path.stat().st_size, purpose])
    print(json.dumps(dict(output=str(OUT), experiments=len(records), best_course=best_course['experiment'],
                          best_map50=best_course['map50'], course_files_unchanged=not changed), ensure_ascii=True))


if __name__ == '__main__':
    main()
