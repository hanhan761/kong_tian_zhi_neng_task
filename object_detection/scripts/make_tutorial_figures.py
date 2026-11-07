"""Create explanatory schematics, not experimental predictions or measurements."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/04_交付汇总/教程示意图'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.set(xlim=(0, 1000), ylim=(500, 0), xlabel='横向像素 x', ylabel='纵向像素 y',
           title='教学算例：1000×500图片中的一个框（不是实际实验图像）')
    ax.set_aspect('equal')
    ax.add_patch(Rectangle((200, 100), 400, 200, facecolor='#dbeafe', edgecolor='#2563eb', lw=2))
    ax.scatter([200, 600, 400], [100, 300, 200], c=['#2563eb', '#2563eb', '#dc2626'])
    ax.text(205, 80, '左上角(200,100)', fontsize=11)
    ax.text(605, 320, '右下角(600,300)', fontsize=11)
    ax.text(415, 195, '中心(400,200)', color='#dc2626', fontsize=11)
    ax.text(50, 435, '框宽400，框高200；除以图片宽高后：中心(0.4,0.4)，宽高(0.4,0.4)', fontsize=11)
    ax.text(50, 475, '若类别编号为0，YOLO标签：0 0.4 0.4 0.4 0.4', fontsize=11)
    for column, size in enumerate([4, 40]):
        axis = fig.add_subplot(grid[1, column])
        shift = 2
        overlap = (size - shift) * size
        iou = overlap / (2 * size * size - overlap)
        axis.add_patch(Rectangle((0, 0), size, size, facecolor='#86efac', edgecolor='#16a34a', alpha=.55, label='参考框'))
        axis.add_patch(Rectangle((shift, 0), size, size, facecolor='#93c5fd', edgecolor='#2563eb', alpha=.55, label='预测框'))
        axis.set(xlim=(-size * .1, size * 1.1 + shift), ylim=(-size * .1, size * 1.1),
                 title=f'{size}×{size}像素的框，同样右移2像素\nIoU={iou:.4f}')
        axis.set_aspect('equal')
        axis.legend(loc='lower right', fontsize=9)
    fig.suptitle('为什么位置误差对小目标更敏感', fontsize=16)
    fig.tight_layout()
    fig.savefig(str(OUT / '标签坐标与小目标IoU.png'), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set(xlim=(0, 12), ylim=(0, 4.5))
    ax.axis('off')
    boxes = [(0.3, 2.6, '图片与参考标签\n标签决定学习目标'),
             (4.3, 2.6, '网络前向预测\n框、目标分数、类别'),
             (8.3, 2.6, '损失：与答案比较\nCIoU或混合NWD'),
             (8.3, 0.5, '反向传播\n计算各参数的梯度'),
             (4.3, 0.5, '优化器更新参数\nSGD或AdamW'),
             (0.3, 0.5, '下一批／下一轮\n在val上评分选best')]
    for x, y, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 3.3, 1.2, boxstyle='round,pad=0.08',
                                   facecolor='#eff6ff', edgecolor='#2563eb', lw=1.5))
        ax.text(x + 1.65, y + .6, label, ha='center', va='center', fontsize=11)
    for start, end in [((3.65, 3.2), (4.2, 3.2)), ((7.65, 3.2), (8.2, 3.2)),
                       ((9.95, 2.5), (9.95, 1.8)), ((8.2, 1.1), (7.7, 1.1)),
                       ((4.2, 1.1), (3.7, 1.1))]:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.8, color='#334155'))
    ax.text(6, 4.2, '训练流程示意：结构、损失、优化器是不同的修改位置', ha='center', fontsize=15)
    ax.text(6, .1, 'CA改变网络中间特征；NWD改变训练误差度量；AdamW改变参数更新。梯度可先累积再更新。',
            ha='center', fontsize=10, color='#475569')
    fig.tight_layout()
    fig.savefig(str(OUT / '训练流程与修改位置.png'), dpi=160)
    plt.close(fig)
    print('Tutorial figures:', OUT)


if __name__ == '__main__':
    main()
