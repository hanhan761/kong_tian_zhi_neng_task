"""Draw v7 explanations from actual course images and labels, never fake predictions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/06_新一轮优化_20261023/第七版教程图'


def draw_box(ax, image, row, title):
    cls, cx, cy, bw, bh = map(float, row.split())
    w, h = image.size
    box = ((cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h)
    ax.imshow(image)
    ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor='#00e34f', linewidth=1.5))
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    return box


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams['axes.unicode_minus'] = False
    audit = json.loads((ROOT / 'reports/00_环境与数据核查/20261022_094416/course_dataset.json').read_text(encoding='utf-8'))
    example = audit['examples']['0']
    original = Image.open(ROOT / example['image']).convert('RGB')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    box = draw_box(axes[0, 0], original, example['raw_line'], '原始课程图：1920×1080（绿色是参考标签）')
    resized = original.resize((640, 360), Image.Resampling.BILINEAR)
    small = draw_box(axes[0, 1], resized, example['raw_line'], '长边缩为640：有效图640×360，再补边')
    for ax, im, b, scale in [(axes[1, 0], original, box, 1), (axes[1, 1], resized, small, 1/3)]:
        draw_box(ax, im, example['raw_line'], '局部放大（仅用于看标签；不是新训练图片）')
        cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
        ax.set_xlim(cx-75*scale, cx+75*scale)
        ax.set_ylim(cy+55*scale, cy-55*scale)
    fig.suptitle('同一架四旋翼：原框15×12像素 → 640输入中约5×4像素\n放大查看不会凭空增加已经被缩小丢掉的细节', fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / '课程真实小目标_原图与640缩小.png', dpi=150)
    plt.close(fig)
    cn = {0:'四旋翼', 1:'六旋翼', 2:'固定翼', 3:'鸟', 4:'气球'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    records = []
    for cls in range(5):
        e = audit['examples'][str(cls)]
        im = Image.open(ROOT / e['image']).convert('RGB')
        box = draw_box(axes.flat[cls], im, e['raw_line'], '%d=%s\n%s' % (cls, cn[cls], Path(e['image']).name))
        # Fit context to a real crop so tiny objects remain visible.
        w, h = im.size
        cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
        radius = max(box[2], box[3], 60) * 1.8
        axes.flat[cls].set_xlim(max(0,cx-radius), min(w,cx+radius))
        axes.flat[cls].set_ylim(min(h,cy+radius), max(0,cy-radius))
        records.append(dict(class_id=cls, **e, box_pixels=box))
    axes.flat[5].axis('off')
    axes.flat[5].text(.05,.85,'这五张是课程训练集中的真实例子。\n绿色框来自已有txt参考标签，\n不是模型检测的结果。\n\n小目标放大显示仍可能模糊，\n不同图片的原始尺寸不同。', fontsize=13, va='top')
    fig.suptitle('本实验的五类目标：对照真实图片认识编号与框', fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / '课程五类真实图片与标签.png', dpi=150)
    plt.close(fig)
    baseline = json.loads((ROOT / 'reports/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/independent_validation_20261022/metrics.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    train = audit['splits']['train']['per_class']
    vals = [train[str(i)] for i in range(5)]
    bars = axes[0].bar([cn[i] for i in range(5)], vals, color='#2563eb')
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x()+b.get_width()/2, v+8, str(v), ha='center')
    axes[0].set(title='训练框数：气球仅14个', ylabel='目标框数量（不是图片数）', ylim=(0,900))
    ap = [x['ap50']*100 for x in baseline['per_class']]
    bars = axes[1].bar([cn[i] for i in range(5)], ap, color='#16a34a')
    for b,v in zip(bars,ap):
        axes[1].text(b.get_x()+b.get_width()/2, v+1, '%.2f'%v, ha='center')
    axes[1].set(title='普通模型逐类AP50：找薄弱类别', ylabel='AP50（%）', ylim=(0,110))
    fig.tight_layout()
    fig.savefig(OUT / '训练类别不均衡与普通模型薄弱项.png', dpi=150)
    plt.close(fig)
    (OUT / 'figures_sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
