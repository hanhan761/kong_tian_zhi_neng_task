# -*- coding: utf-8 -*-
"""plot_utils.py —— 用 Pillow 画实验报告插图（折线图 / 分组柱状图）

为什么不用 matplotlib：本实验的运行环境（Python 3.12 + 该课程虚拟环境）中没有安装
matplotlib，而 Pillow 随环境自带。这里用 Pillow 手写了坐标轴、刻度、网格、图例，
保证脚本换到任何装了 Pillow 的机器上都能直接出图，且中文字体不再显示成方框。

对外接口：
    line_chart(path, title, xlabel, ylabel, series, xtick_step=None)
    bar_chart(path, title, categories, series, ylabel="值", value_fmt="{:.4f}")
series 形如 [(图例名, RGB颜色, xs, ys)] 或 [(图例名, RGB颜色, values)]
"""
import os

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

BG = (255, 255, 255)
AXIS = (70, 70, 70)
GRID = (226, 230, 235)
TEXT = (45, 52, 64)


def _font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:                              # noqa: BLE001
                continue
    return ImageFont.load_default()


def _canvas(w, h):
    img = Image.new("RGB", (w, h), BG)
    return img, ImageDraw.Draw(img)


def _axes(draw, box, xlabel, ylabel, title, font_title, font_label):
    left, top, right, bottom = box
    draw.rectangle([left, top, right, bottom], outline=AXIS, width=1)
    tw = draw.textlength(title, font=font_title)
    draw.text(((left + right - tw) / 2, top - 38), title, font=font_title, fill=TEXT)
    xw = draw.textlength(xlabel, font=font_label)
    draw.text(((left + right - xw) / 2, bottom + 34), xlabel, font=font_label, fill=TEXT)
    draw.text((left - 52, top - 30), ylabel, font=font_label, fill=TEXT)


def _ylabels(draw, box, ymax, font_label, fmt="{:.2f}", n=5, ticks=None):
    left, top, right, bottom = box
    if ticks is None:
        ticks = [ymax * i / n for i in range(n + 1)]
    for v in ticks:
        y = bottom - (bottom - top) * v / ymax
        draw.line([left, y, right, y], fill=GRID, width=1)
        label = fmt.format(v)
        tw = draw.textlength(label, font=font_label)
        draw.text((left - 8 - tw, y - 8), label, font=font_label, fill=TEXT)


def _nice_ticks(vmax, target=5):
    """把最大值向上取整到「好看」的刻度"""
    import math
    raw = vmax / float(target)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    n = int(math.ceil(vmax / step - 1e-9))
    return step, [step * i for i in range(n + 1)]


def _legend(draw, items, x, y, font_label):
    for i, (label, color) in enumerate(items):
        yy = y + i * 26
        draw.line([x, yy + 8, x + 26, yy + 8], fill=color, width=4)
        draw.text((x + 34, yy), label, font=font_label, fill=TEXT)


def line_chart(path, title, xlabel, ylabel, series, xtick_step=None):
    """series: [(label, color, xs, ys)]"""
    W, H = 1000, 560
    box = (92, 66, W - 44, H - 92)
    img, draw = _canvas(W, H)
    f_title, f_label = _font(24), _font(17)
    _axes(draw, box, xlabel, ylabel, title, f_title, f_label)

    left, top, right, bottom = box
    xmax = max(max(s[2]) for s in series) or 1
    ymax = max(max(s[3]) for s in series) or 1.0
    step, ticks = _nice_ticks(ymax * 1.06)
    ymax = ticks[-1]
    _ylabels(draw, box, ymax, f_label, ticks=ticks)

    step = xtick_step or max(1, int(round(xmax / 10)))
    tick = step
    while tick <= xmax:
        x = left + (right - left) * tick / xmax
        draw.line([x, top, x, bottom], fill=GRID, width=1)
        label = str(int(tick))
        tw = draw.textlength(label, font=f_label)
        draw.text((x - tw / 2, bottom + 6), label, font=f_label, fill=TEXT)
        tick += step

    for label, color, xs, ys in series:
        pts = [(left + (right - left) * x / xmax, bottom - (bottom - top) * y / ymax)
               for x, y in zip(xs, ys)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3, joint="curve")
    _legend(draw, [(s[0], s[1]) for s in series], right - 210, top + 10, f_label)

    img.save(path)
    return path


def bar_chart(path, title, categories, series, ylabel="值", value_fmt="{:.4f}"):
    """series: [(label, color, values)]"""
    W, H = 1000, 540
    box = (92, 66, W - 44, H - 92)
    img, draw = _canvas(W, H)
    f_title, f_label = _font(23), _font(17)
    _axes(draw, box, "指标", ylabel, title, f_title, f_label)

    left, top, right, bottom = box
    ymax = max(max(s[2]) for s in series) * 1.18 or 1.0
    step, ticks = _nice_ticks(ymax)
    ymax = ticks[-1] if ticks[-1] >= ymax * 0.9 else ymax
    _ylabels(draw, box, ymax, f_label, fmt="{:.2f}", ticks=ticks)

    n_cat, n_ser = len(categories), len(series)
    slot = (right - left) / n_cat
    bar_w = slot * 0.62 / n_ser
    for ci, cat in enumerate(categories):
        cx = left + slot * (ci + 0.5)
        tw = draw.textlength(cat, font=f_label)
        draw.text((cx - tw / 2, bottom + 8), cat, font=f_label, fill=TEXT)
        for si, (label, color, values) in enumerate(series):
            v = values[ci]
            h = (bottom - top) * v / ymax
            x0 = cx - slot * 0.31 + si * bar_w
            draw.rectangle([x0, bottom - h, x0 + bar_w * 0.92, bottom], fill=color)
            text = value_fmt.format(v)
            tw2 = draw.textlength(text, font=f_label)
            draw.text((x0 + bar_w * 0.46 - tw2 / 2, bottom - h - 24), text,
                      font=f_label, fill=TEXT)
    _legend(draw, [(s[0], s[1]) for s in series], right - 230, top + 10, f_label)

    img.save(path)
    return path
