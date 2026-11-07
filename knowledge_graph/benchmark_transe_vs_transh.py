# -*- coding: utf-8 -*-
"""
benchmark_transe_vs_transh.py —— TransE / TransH 对照实验

用途
----
作业要求「模型优化提高评价指标」。仅凭一次运行的数字无法说明优化是否有效，
因此本脚本在**完全相同**的数据划分、负采样策略、学习率与训练轮数下，
分别训练 TransE 与 TransH，用 filtered 排名指标（MRR / Hits@1 / Hits@3 / Hits@10）
做对照，输出一张可直接写进实验报告的结果表。

与 chazhao.py 中实现的区别
--------------------------
chazhao.py 是「交互式应用 + 可视化」的完整版本，其训练循环逐条样本执行，
便于理解但速度慢。本脚本把同样的更新公式改写为批量矩阵运算：

    TransE:  f(h,r,t) = || h + r - t ||_2
    TransH:  f(h,r,t) = || M_r h + r - M_r t ||_2 ,  M_r = I - n_r n_r^T

其中 M_r 是投影矩阵，n_r 是关系 r 所在超平面的法向量（TransH 的核心改进：
让同一关系在不同超平面上表达，从而区分 1-to-N / N-to-1 / N-to-N 关系）。
两种模型的梯度形式一致，只差一次「去法向分量」的投影：

    M_r v = v - (v · n_r) n_r

因此「同一套训练流程换一个评分函数」即为一次干净的消融实验。

运行
----
    python benchmark_transe_vs_transh.py                # 默认 100 轮
    python benchmark_transe_vs_transh.py --epochs 200   # 指定轮数

输出
----
    benchmark_result.txt   指标对照表（同时打印到控制台）
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "output")
LOSS_PNG = os.path.join(BASE_DIR, "训练损失曲线.png")
CMP_PNG = os.path.join(BASE_DIR, "指标对比.png")


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_data(output_dir):
    """读取 entityid.txt / relationid.txt / triples.txt"""
    entities, relations = [], []
    with open(os.path.join(output_dir, "entityid.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                entities.append(parts[0])
    with open(os.path.join(output_dir, "relationid.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                relations.append(parts[0])

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    triples = []
    with open(os.path.join(output_dir, "triples.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and all(p in ent2id for p in (parts[0], parts[2])) \
                    and parts[1] in rel2id:
                triples.append((ent2id[parts[0]], rel2id[parts[1]], ent2id[parts[2]]))
    return entities, relations, np.array(triples, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class Embedding(object):
    """TransE / TransH 的共享外壳"""

    def __init__(self, n_ent, n_rel, dim=100, lr=0.01, margin=1.0, C=0.5,
                 kind="transh", seed=2026):
        rng = np.random.RandomState(seed)
        bound = 6.0 / np.sqrt(dim)
        self.n_ent, self.n_rel, self.dim = n_ent, n_rel, dim
        self.lr, self.margin, self.C, self.kind = lr, margin, C, kind

        self.E = rng.uniform(-bound, bound, (n_ent, dim))       # 实体向量
        self.R = rng.uniform(-bound, bound, (n_rel, dim))       # 关系向量
        if kind == "transh":
            self.N = rng.uniform(-bound, bound, (n_rel, dim))   # 超平面法向量
            self.N /= np.linalg.norm(self.N, axis=1, keepdims=True)

    # ---------------- 评分 ----------------
    def project(self, V, n):
        """把向量/矩阵 V 投影到法向量 n 定义的超平面上: V - (V·n) n"""
        if n.ndim == 1:                     # 单个法向量
            if V.ndim == 1:
                return V - np.dot(V, n) * n
            return V - (V @ n)[:, None] * n[None, :]
        # 批量法向量：逐行点积
        return V - np.sum(V * n, axis=1)[:, None] * n

    def score_all_tails(self, h, r):
        """给定 (h, r)，返回该实体集合作为尾实体的得分向量（越小越好）"""
        if self.kind == "transe":
            return np.sum((self.E[h] + self.R[r] - self.E) ** 2, axis=1)
        n = self.N[r]
        h_proj = self.project(self.E[h], n)
        P = self.project(self.E, n)                             # 所有实体投影
        return np.sum((h_proj + self.R[r] - P) ** 2, axis=1)

    # ---------------- 训练 ----------------
    def train_step(self, pos, neg):
        """pos / neg: (batch, 3) 的 h,r,t 数组；返回本 batch 的平均 hinge loss"""
        eps = 1e-12
        if self.kind == "transe":
            res_p = self.E[pos[:, 0]] + self.R[pos[:, 1]] - self.E[pos[:, 2]]
            res_n = self.E[neg[:, 0]] + self.R[neg[:, 1]] - self.E[neg[:, 2]]
            s_p = np.sum(res_p ** 2, axis=1)
            s_n = np.sum(res_n ** 2, axis=1)
            g_p, g_n = 2 * res_p, -2 * res_n
        else:
            n = self.N[pos[:, 1]]
            v_p = self.E[pos[:, 0]] - self.E[pos[:, 2]]
            res_p = self.project(self.E[pos[:, 0]], n) + self.R[pos[:, 1]] \
                - self.project(self.E[pos[:, 2]], n)
            v_n = self.E[neg[:, 0]] - self.E[neg[:, 2]]
            res_n = self.project(self.E[neg[:, 0]], n) + self.R[neg[:, 1]] \
                - self.project(self.E[neg[:, 2]], n)
            s_p = np.sum(res_p ** 2, axis=1)
            s_n = np.sum(res_n ** 2, axis=1)
            # 投影后残差的梯度还要再过一次投影（M_r 对称且幂等）
            g_p = 2 * self.project(res_p, n)
            g_n = -2 * self.project(res_n, n)

        loss = np.maximum(0.0, self.margin + s_p - s_n)
        active = loss > 0
        if not np.any(active):
            return 0.0

        pos, neg = pos[active], neg[active]
        g_p, g_n = g_p[active], g_n[active]
        b = len(pos)
        grad_E = np.zeros_like(self.E)
        grad_R = np.zeros_like(self.R)

        np.add.at(grad_E, pos[:, 0], g_p)
        np.add.at(grad_E, pos[:, 2], -g_p)
        np.add.at(grad_E, neg[:, 0], g_n)
        np.add.at(grad_E, neg[:, 2], -g_n)
        np.add.at(grad_R, pos[:, 1], g_p)
        np.add.at(grad_R, neg[:, 1], g_n)

        self.E -= self.lr * grad_E / b
        self.R -= self.lr * grad_R / b
        self.R /= np.maximum(np.linalg.norm(self.R, axis=1, keepdims=True), eps)

        if self.kind == "transh":
            n = self.N[pos[:, 1]]
            grad_N = np.zeros_like(self.N)
            if self.C > 0:      # 正交软约束：关系向量应落在超平面内
                r_vec = self.R[pos[:, 1]]
                orth = np.sum(r_vec * n, axis=1)[:, None]
                np.add.at(grad_N, pos[:, 1], 2 * self.C * orth * r_vec)
            np.add.at(grad_N, pos[:, 1], -2 * (
                np.sum(n * v_p[active], axis=1)[:, None] * res_p[active]
                + np.sum(res_p[active] * n, axis=1)[:, None] * v_p[active]))
            np.add.at(grad_N, neg[:, 1], -2 * (
                np.sum(n * v_n[active], axis=1)[:, None] * res_n[active]
                + np.sum(res_n[active] * n, axis=1)[:, None] * v_n[active]))
            self.N -= self.lr * grad_N / b
            self.N /= np.maximum(np.linalg.norm(self.N, axis=1, keepdims=True), eps)

        return float(loss.mean())


# --------------------------------------------------------------------------- #
# 训练 / 评估
# --------------------------------------------------------------------------- #
def train(model, train_triples, n_ent, epochs, batch_size=256, seed=2026):
    rng = np.random.RandomState(seed)
    steps = max(1, len(train_triples) // batch_size)
    history = []
    for epoch in range(epochs):
        idx = rng.permutation(len(train_triples))
        epoch_loss = 0.0
        for s in range(steps):
            pos = train_triples[idx[s * batch_size:(s + 1) * batch_size]]
            neg = pos.copy()
            corrupt_tail = rng.rand(len(pos)) > 0.5
            rand_ent = rng.randint(0, n_ent, len(pos))
            neg[corrupt_tail, 2] = rand_ent[corrupt_tail]
            neg[~corrupt_tail, 0] = rand_ent[~corrupt_tail]
            epoch_loss += model.train_step(pos, neg)
        avg = epoch_loss / steps
        history.append(avg)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  {model.kind:<7} epoch {epoch + 1:>3}/{epochs}  loss={avg:.4f}", flush=True)
    return history


def evaluate(model, test_triples, all_triples, n_ent):
    """
    filtered 尾实体预测评估：
    对每个测试三元组 (h,r,t)，对全部实体打分，把训练集中同 (h,r) 的其它真实尾实体
    的得分置为无穷大（filtered 设置），然后看正确尾实体的排名。
    """
    known = {}
    for h, r, t in all_triples:
        known.setdefault((int(h), int(r)), set()).add(int(t))

    ranks = []
    for h, r, t in test_triples:
        h, r, t = int(h), int(r), int(t)
        scores = model.score_all_tails(h, r)
        for other in known.get((h, r), ()):
            if other != t:
                scores[other] = np.inf
        rank = int(np.sum(scores < scores[t])) + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "平均排名": float(np.mean(ranks)),
        "中位数排名": float(np.median(ranks)),
    }


def main():
    ap = argparse.ArgumentParser(description="TransE vs TransH 对照实验")
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数（两个模型一致）")
    ap.add_argument("--dim", type=int, default=100, help="嵌入维度")
    ap.add_argument("--lr", type=float, default=0.02, help="学习率")
    ap.add_argument("--margin", type=float, default=1.0, help="hinge 间隔")
    ap.add_argument("--n_test", type=int, default=200, help="测试三元组条数")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子")
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="output 目录")
    ap.add_argument("--result_path", type=str,
                    default=os.path.join(BASE_DIR, "benchmark_result.txt"))
    args = ap.parse_args()

    entities, relations, triples = load_data(args.output_dir)
    print(f"实体 {len(entities)} 个，关系 {len(relations)} 个，三元组 {len(triples)} 条",
          flush=True)

    rng = random.Random(args.seed)
    idx = list(range(len(triples)))
    rng.shuffle(idx)
    test_idx = idx[:args.n_test]
    train_idx = idx[args.n_test:]
    train_triples, test_triples = triples[train_idx], triples[test_idx]
    print(f"训练 {len(train_triples)} 条 / 测试 {len(test_triples)} 条（filtered 评估）",
          flush=True)

    results = {}
    histories = {}
    for kind, label in (("transe", "TransE（基线）"), ("transh", "TransH（优化模型）")):
        print(f"\n=== {label} ===", flush=True)
        model = Embedding(len(entities), len(relations), dim=args.dim,
                          lr=args.lr, margin=args.margin, kind=kind, seed=args.seed)
        t0 = time.time()
        histories[kind] = train(model, train_triples, len(entities), args.epochs,
                                seed=args.seed)
        metrics = evaluate(model, test_triples, triples, len(entities))
        metrics["训练耗时(s)"] = round(time.time() - t0, 1)
        results[kind] = metrics
        print(f"  MRR={metrics['MRR']:.4f}  Hits@1={metrics['Hits@1']:.4f}  "
              f"Hits@3={metrics['Hits@3']:.4f}  Hits@10={metrics['Hits@10']:.4f}",
              flush=True)

    # ---------------- 报告 ----------------
    keys = ["MRR", "Hits@1", "Hits@3", "Hits@10", "平均排名", "中位数排名", "训练耗时(s)"]
    lines = []
    lines.append(f"数据：实体 {len(entities)} / 关系 {len(relations)} / "
                 f"三元组 {len(triples)}；训练 {len(train_triples)} 条，"
                 f"测试 {len(test_triples)} 条；维度 {args.dim}，轮数 {args.epochs}，"
                 f"lr={args.lr}，margin={args.margin}，seed={args.seed}，filtered 评估")
    lines.append("")
    lines.append(f"{'指标':<12}{'TransE':>12}{'TransH':>12}{'变化':>14}")
    lines.append("-" * 50)
    for k in keys:
        a, b = results["transe"][k], results["transh"][k]
        if k in ("平均排名", "中位数排名", "训练耗时(s)"):
            lines.append(f"{k:<12}{a:>12.2f}{b:>12.2f}{(b - a) / a * 100:>13.1f}%")
        else:
            lines.append(f"{k:<12}{a:>12.4f}{b:>12.4f}{(b - a) / max(a, 1e-12) * 100:>13.1f}%")
    text = "\n".join(lines)
    print("\n" + text)
    with open(args.result_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n结果已写入 {args.result_path}")

    with open(os.path.join(BASE_DIR, "benchmark_result.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "result": results,
                   "loss_history": {k: [round(v, 5) for v in hist]
                                    for k, hist in histories.items()}},
                  f, ensure_ascii=False, indent=2)

    # ---------------- 出图（Pillow 手绘，不依赖 matplotlib）----------------
    try:
        from plot_utils import line_chart, bar_chart
    except ImportError:
        print("未找到 plot_utils.py，跳过绘图")
        return

    epochs = len(histories["transe"])
    xs = list(range(1, epochs + 1))
    line_chart(LOSS_PNG, "训练损失下降曲线（TransE vs TransH）", "训练轮数", "平均损失 loss",
               [("TransE", (52, 120, 216), xs, histories["transe"]),
                ("TransH", (214, 69, 65), xs, histories["transh"])])
    bar_chart(CMP_PNG, "链接预测指标对比（filtered，测试集 %d 条）" % len(test_triples),
              ["MRR", "Hits@1", "Hits@3", "Hits@10"],
              [("TransE", (52, 120, 216), [results["transe"][k] for k in
                                          ("MRR", "Hits@1", "Hits@3", "Hits@10")]),
               ("TransH", (214, 69, 65), [results["transh"][k] for k in
                                         ("MRR", "Hits@1", "Hits@3", "Hits@10")])],
              ylabel="指标值")
    print("图已保存：%s\n           %s" % (LOSS_PNG, CMP_PNG))


if __name__ == "__main__":
    main()
