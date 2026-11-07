# -*- coding: utf-8 -*-
"""
find_tail_gui.py —— 武器装备知识图谱「头实体 + 关系 → 尾实体」查找交互界面

设计目标
--------
作业 3.3.2 的要求是：给定「头实体」与「关系」，查找可能的尾实体。
指导书中的 chazhao.py 采用命令行交互（``输入头实体和关系查找可能的尾实体(输入q退出)``），
本界面把它升级为一个可以实际演示的桌面应用：

    ┌──────────────── 查询条件 ────────────────┐ ┌──────── 候选尾实体（按得分升序）────────┐
    │ 头实体  [ 075型            ▼ ] (可输入过滤)│ │ 排名 | 尾实体 | 得分 | 来源            │
    │ 关系    [ 舰种[weight=1]   ▼ ]            │ │  1   | 直升机船坞登陆舰 | 0.00 | 已知   │
    │ Top-K   [ 10 ]      [查找尾实体] [清空]   │ │  2   | 两栖攻击舰     | 3.21 | 预测   │
    └───────────────────────────────────────────┘ └────────────────────────────────────────┘

功能
----
1. 实体名联想：输入即过滤下拉列表，避免 2000+ 实体里翻找。
2. 已知三元组优先：图中已存在的 (h, r, t) 直接标注「已知」并排在最前，
   模型补充的候选标注「预测」，两类结果一目了然（对应指导书 chazhao.py 的
   find_tail_entities 中 known_results + 模型预测的合并逻辑）。
3. 关系名自带权重后缀 ``[weight=n]``：n 表示该关系在全图中连接的不同尾实体数，
   n 越大说明这个关系越是「一对多」，模型越难预测——界面上直接展示，便于分析。
4. 结果可导出为 txt，便于写进实验报告。
5. TransH 嵌入在首次运行时训练并缓存为 model_cache.npz，之后启动秒开。

运行
----
    python find_tail_gui.py                # 自动查找 ../output，缺模型则训练
    python find_tail_gui.py --output_dir D:\\xxx\\output --top_k 20

依赖：numpy、tkinter（Python 标准库自带）。
"""
import argparse
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(os.path.dirname(BASE_DIR), "output")


# --------------------------------------------------------------------------- #
# 数据与模型
# --------------------------------------------------------------------------- #
def load_output(output_dir):
    """读取 output/{entityid,relationid,triples}.txt"""
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
            if len(parts) == 3 and parts[0] in ent2id and parts[2] in ent2id \
                    and parts[1] in rel2id:
                triples.append((ent2id[parts[0]], rel2id[parts[1]], ent2id[parts[2]]))
    return entities, relations, np.array(triples, dtype=np.int64)


class TransH(object):
    """TransH：f(h,r,t) = || M_r h + r - M_r t ||_2 ，M_r = I - n_r n_r^T"""

    def __init__(self, n_ent, n_rel, dim=100, lr=0.01, margin=1.0, C=0.5, seed=2026):
        rng = np.random.RandomState(seed)
        bound = 6.0 / np.sqrt(dim)
        self.dim = dim
        self.lr, self.margin, self.C = lr, margin, C
        self.E = rng.uniform(-bound, bound, (n_ent, dim))
        self.R = rng.uniform(-bound, bound, (n_rel, dim))
        self.N = rng.uniform(-bound, bound, (n_rel, dim))
        self.N /= np.linalg.norm(self.N, axis=1, keepdims=True)

    @staticmethod
    def project(V, n):
        """投影到超平面：V - (V·n)n（支持批量法向量）"""
        if n.ndim == 1:
            if V.ndim == 1:
                return V - np.dot(V, n) * n
            return V - (V @ n)[:, None] * n[None, :]
        return V - np.sum(V * n, axis=1)[:, None] * n

    def score_all_tails(self, h, r):
        """对全部实体计算作为尾实体的得分（越小越好）"""
        n = self.N[r]
        h_proj = self.project(self.E[h], n)
        return np.sum((h_proj + self.R[r] - self.project(self.E, n)) ** 2, axis=1)

    def train_step(self, pos, neg):
        n = self.N[pos[:, 1]]
        res_p = self.project(self.E[pos[:, 0]], n) + self.R[pos[:, 1]] \
            - self.project(self.E[pos[:, 2]], n)
        res_n = self.project(self.E[neg[:, 0]], n) + self.R[neg[:, 1]] \
            - self.project(self.E[neg[:, 2]], n)
        s_p = np.sum(res_p ** 2, axis=1)
        s_n = np.sum(res_n ** 2, axis=1)
        loss = np.maximum(0.0, self.margin + s_p - s_n)
        active = loss > 0
        if not np.any(active):
            return 0.0

        pos, neg = pos[active], neg[active]
        n = n[active]
        g_p = 2 * self.project(res_p[active], n)
        g_n = -2 * self.project(res_n[active], n)
        b = len(pos)

        grad_E = np.zeros_like(self.E)
        grad_R = np.zeros_like(self.R)
        np.add.at(grad_E, pos[:, 0], g_p)
        np.add.at(grad_E, pos[:, 2], -g_p)
        np.add.at(grad_E, neg[:, 0], g_n)
        np.add.at(grad_E, neg[:, 2], -g_n)
        np.add.at(grad_R, pos[:, 1], g_p)
        np.add.at(grad_R, neg[:, 1], g_n)

        v_p = self.E[pos[:, 0]] - self.E[pos[:, 2]]
        v_n = self.E[neg[:, 0]] - self.E[neg[:, 2]]
        r_vec = self.R[pos[:, 1]]
        grad_N = np.zeros_like(self.N)
        if self.C > 0:
            orth = np.sum(r_vec * n, axis=1)[:, None]
            np.add.at(grad_N, pos[:, 1], 2 * self.C * orth * r_vec)
        np.add.at(grad_N, pos[:, 1], -2 * (
            np.sum(n * v_p, axis=1)[:, None] * res_p[active]
            + np.sum(res_p[active] * n, axis=1)[:, None] * v_p))
        np.add.at(grad_N, neg[:, 1], -2 * (
            np.sum(n * v_n, axis=1)[:, None] * res_n[active]
            + np.sum(res_n[active] * n, axis=1)[:, None] * v_n))

        self.E -= self.lr * grad_E / b
        self.R -= self.lr * grad_R / b
        self.R /= np.maximum(np.linalg.norm(self.R, axis=1, keepdims=True), 1e-12)
        self.N -= self.lr * grad_N / b
        self.N /= np.maximum(np.linalg.norm(self.N, axis=1, keepdims=True), 1e-12)
        return float(loss.mean())

    def train(self, triples, n_ent, epochs=200, batch_size=256, seed=2026,
              progress=None, should_stop=None):
        rng = np.random.RandomState(seed)
        steps = max(1, len(triples) // batch_size)
        for epoch in range(epochs):
            if should_stop and should_stop():
                break
            idx = rng.permutation(len(triples))
            for s in range(steps):
                pos = triples[idx[s * batch_size:(s + 1) * batch_size]]
                neg = pos.copy()
                tail_mask = rng.rand(len(pos)) > 0.5
                rand_ent = rng.randint(0, n_ent, len(pos))
                neg[tail_mask, 2] = rand_ent[tail_mask]
                neg[~tail_mask, 0] = rand_ent[~tail_mask]
                self.train_step(pos, neg)
            if progress and (epoch % 10 == 0 or epoch == epochs - 1):
                progress(epoch + 1, epochs)

    def save(self, path):
        np.savez_compressed(path, E=self.E, R=self.R, N=self.N)

    def load(self, path):
        data = np.load(path)
        self.E, self.R, self.N = data["E"], data["R"], data["N"]


# --------------------------------------------------------------------------- #
# 界面
# --------------------------------------------------------------------------- #
class FindTailApp(object):
    FONT = ("Microsoft YaHei UI", 10)
    FONT_BOLD = ("Microsoft YaHei UI", 10, "bold")
    FONT_TITLE = ("Microsoft YaHei UI", 15, "bold")

    def __init__(self, root, output_dir, top_k=10, auto_train=True, epochs=200,
                 cache_path=None, demo=None):
        self.root = root
        self.output_dir = output_dir
        self.default_top_k = top_k
        self.epochs = epochs
        self.cache_path = cache_path or os.path.join(BASE_DIR, "model_cache.npz")
        self.demo = demo
        self.model = None
        self.training = False

        self.root.title("武器装备知识图谱 · TransH 尾实体查找")
        self.root.geometry("1180x660")
        self.root.minsize(1020, 600)

        self._load_data()
        self._build_ui()
        if os.path.exists(self.cache_path):
            self._load_cache()
        elif auto_train:
            self.root.after(300, self.train_model)

    # ---------------- 数据 ----------------
    def _load_data(self):
        self.entities, self.relations, self.triples = load_output(self.output_dir)
        self.ent2id = {e: i for i, e in enumerate(self.entities)}
        self.id2ent = {i: e for i, e in enumerate(self.entities)}
        self.triple_dict = {}
        for h, r, t in self.triples:
            self.triple_dict.setdefault((int(h), int(r)), []).append(int(t))
        self.rel_count = {}
        for h, r, t in self.triples:
            self.rel_count[int(r)] = self.rel_count.get(int(r), 0) + 1

    # ---------------- 界面 ----------------
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Treeview", font=self.FONT, rowheight=26)
        style.configure("Treeview.Heading", font=self.FONT_BOLD)
        style.configure("TButton", font=self.FONT)
        style.configure("TLabel", font=self.FONT)
        style.configure("TCheckbutton", font=self.FONT)

        header = tk.Frame(self.root, bg="#1f3a5f", height=54)
        header.pack(fill="x")
        tk.Label(header, text="武器装备知识图谱 · TransH 尾实体查找系统",
                 bg="#1f3a5f", fg="white", font=self.FONT_TITLE).pack(side="left", padx=16, pady=10)
        tk.Label(header, text="输入「头实体 + 关系」，输出最可能的尾实体",
                 bg="#1f3a5f", fg="#c9d8ea", font=self.FONT).pack(side="left", padx=6, pady=16)

        body = tk.Frame(self.root, padx=12, pady=10)
        body.pack(fill="both", expand=True)

        # ---- 左：查询条件 ----
        left = ttk.LabelFrame(body, text=" 查询条件 ", padding=12)
        left.pack(side="left", fill="y", padx=(0, 12))

        ttk.Label(left, text="头实体（可输入过滤）").grid(row=0, column=0, sticky="w")
        self.head_var = tk.StringVar()
        self.head_box = ttk.Combobox(left, textvariable=self.head_var, width=34,
                                     font=self.FONT, values=self.entities)
        self.head_box.grid(row=1, column=0, pady=(2, 12), sticky="we")
        self.head_box.bind("<KeyRelease>", self._filter_heads)

        ttk.Label(left, text="关系").grid(row=2, column=0, sticky="w")
        self.rel_var = tk.StringVar()
        self.rel_box = ttk.Combobox(left, textvariable=self.rel_var, width=34,
                                    font=self.FONT,
                                    values=sorted(self.relations,
                                                  key=lambda r: -self.rel_count.get(
                                                      self._rid(r), 0)))
        self.rel_box.grid(row=3, column=0, pady=(2, 12), sticky="we")

        ttk.Label(left, text="返回候选数 Top-K").grid(row=4, column=0, sticky="w")
        self.k_var = tk.IntVar(value=self.default_top_k)
        ttk.Spinbox(left, from_=3, to=50, textvariable=self.k_var, width=8,
                    font=self.FONT).grid(row=5, column=0, pady=(2, 14), sticky="w")

        btns = tk.Frame(left)
        btns.grid(row=6, column=0, sticky="we")
        ttk.Button(btns, text="查找尾实体", command=self.on_find).pack(side="left")
        ttk.Button(btns, text="清空", command=self.on_clear).pack(side="left", padx=6)
        ttk.Button(btns, text="导出结果", command=self.on_export).pack(side="left")

        ttk.Button(left, text="重新训练嵌入", command=self.train_model).grid(
            row=7, column=0, sticky="we", pady=(10, 0))

        self.info_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.info_var, foreground="#555",
                  wraplength=300, justify="left").grid(row=8, column=0, pady=(14, 0), sticky="w")

        # ---- 右：结果表 ----
        right = ttk.LabelFrame(body, text=" 候选尾实体（图中已知三元组优先，其余按模型得分升序） ",
                               padding=8)
        right.pack(side="left", fill="both", expand=True)

        self.summary_var = tk.StringVar(value="请选择头实体与关系后点击「查找尾实体」")
        ttk.Label(right, textvariable=self.summary_var, font=self.FONT_BOLD,
                  foreground="#1f3a5f").pack(anchor="w", pady=(0, 6))

        cols = ("rank", "tail", "score", "source")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=17)
        for col, text, width, anchor in (
                ("rank", "排名", 60, "center"),
                ("tail", "候选尾实体", 420, "w"),
                ("score", "得分", 140, "center"),
                ("source", "来源", 120, "center")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("known", background="#e8f4ea")
        self.tree.tag_configure("pred", background="#ffffff")

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, anchor="w",
                  relief="sunken", padding=(8, 4), font=self.FONT).pack(fill="x", side="bottom")

    # ---------------- 辅助 ----------------
    def _rid(self, rel):
        rels = getattr(self, "_rel_index", None)
        if rels is None:
            rels = {r: i for i, r in enumerate(self.relations)}
            self._rel_index = rels
        return rels.get(rel, -1)

    def _filter_heads(self, _event=None):
        text = self.head_var.get().strip()
        if not text:
            self.head_box.configure(values=self.entities)
            return
        hits = [e for e in self.entities if text in e][:200]
        if hits:
            self.head_box.configure(values=hits)

    def _set_status(self, text):
        self.status_var.set(text)

    # ---------------- 模型 ----------------
    def _load_cache(self):
        try:
            self.model = TransH(len(self.entities), len(self.relations))
            self.model.load(self.cache_path)
            self.info_var.set(f"嵌入已加载：{self.model.dim} 维（缓存）")
            self._refresh_status("模型就绪")
        except Exception as e:                              # noqa: BLE001
            self.model = None
            self._set_status(f"缓存读取失败，将重新训练：{e}")

    def train_model(self):
        if self.training:
            return
        self.training = True
        self._set_status("正在训练 TransH 嵌入……")

        def worker():
            t0 = time.time()
            model = TransH(len(self.entities), len(self.relations))
            model.train(self.triples, len(self.entities), epochs=self.epochs,
                        progress=lambda e, n: self.root.after(
                            0, self._set_status, f"正在训练 TransH 嵌入…… {e}/{n} 轮"),
                        should_stop=lambda: not self.training)
            try:
                model.save(self.cache_path)
            except Exception:
                pass
            self.model = model
            self.training = False
            self.root.after(0, lambda: self.info_var.set(
                f"嵌入已训练：{model.dim} 维，{time.time() - t0:.1f}s"))
            self.root.after(0, lambda: self._refresh_status("模型就绪"))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_status(self, prefix):
        self._set_status(
            f"{prefix} | 实体 {len(self.entities)} · 关系 {len(self.relations)} · "
            f"三元组 {len(self.triples)} | 数据目录：{os.path.abspath(self.output_dir)}")

    # ---------------- 查询 ----------------
    def on_find(self, _event=None):
        head = self.head_var.get().strip()
        rel = self.rel_var.get().strip()
        if not head or not rel:
            messagebox.showinfo("提示", "请填写头实体和关系")
            return
        if head not in self.ent2id:
            messagebox.showwarning("提示", f"图谱中不存在实体「{head}」")
            return
        if rel not in self.relations:
            messagebox.showwarning("提示", f"图谱中不存在关系「{rel}」")
            return

        h, r = self.ent2id[head], self._rid(rel)
        k = max(1, int(self.k_var.get()))
        known = self.triple_dict.get((h, r), [])

        rows = []
        if known:                       # 已知三元组优先（图中事实，不参与模型排序）
            for t in known:
                rows.append((self.id2ent[t], None, "已知"))

        if self.model is not None:      # 模型补全
            scores = self.model.score_all_tails(h, r)
            seen = set(known) | {h}
            order = np.argsort(scores)
            for t in order:
                if len(rows) >= max(k, len(known)):
                    break
                t = int(t)
                if t in seen:
                    continue
                rows.append((self.id2ent[t], float(scores[t]), "预测"))

        n_known = sum(1 for _, _, src in rows if src == "已知")
        rows = sorted(rows, key=lambda x: (x[2] != "已知",
                                          x[1] if x[1] is not None else 0.0))
        rows = rows[:max(k, len(known))]

        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, (name, score, src) in enumerate(rows, 1):
            score_text = "—" if score is None else "%.4f" % score
            self.tree.insert("", "end", values=(i, name, score_text, src),
                             tags=("known" if src == "已知" else "pred",))

        n_known_rel = self.rel_count.get(r, 0)
        self.summary_var.set(
            f"{head} ——[{rel}]——> ?    已找到 {len(rows)} 个候选（图中已知 {n_known} 条并置顶；"
            f"该关系在全图共 {n_known_rel} 条）")
        self._refresh_status("查询完成")

    def on_clear(self):
        self.head_var.set("")
        self.rel_var.set("")
        self.head_box.configure(values=self.entities)
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.summary_var.set("请选择头实体与关系后点击「查找尾实体」")
        self._refresh_status("已清空")

    def on_export(self):
        rows = [self.tree.item(i, "values") for i in self.tree.get_children()]
        if not rows:
            messagebox.showinfo("提示", "当前没有可导出的结果")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile="find_tail_result.txt",
                                            filetypes=[("文本文件", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"头实体：{self.head_var.get()}\t关系：{self.rel_var.get()}\n")
            f.write("排名\t尾实体\t得分\t来源\n")
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")
        self._set_status(f"结果已导出：{path}")

    # ---------------- 演示 ----------------
    def run_demo(self):
        """供截图使用：填入作业示例并查询"""
        head, rel = self.demo
        self.head_var.set(head)
        self.rel_var.set(rel)
        self.on_find()


def main():
    ap = argparse.ArgumentParser(description="TransH 尾实体查找交互界面")
    ap.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT, help="output 目录")
    ap.add_argument("--top_k", type=int, default=10, help="默认返回候选数")
    ap.add_argument("--epochs", type=int, default=200, help="缺少缓存时的训练轮数")
    ap.add_argument("--no_train", action="store_true", help="缺少缓存时不自动训练")
    ap.add_argument("--demo", type=str, default="", help="截图演示：'头实体|关系'")
    args = ap.parse_args()

    demo = tuple(args.demo.split("|")) if "|" in args.demo else None
    root = tk.Tk()
    app = FindTailApp(root, args.output_dir, top_k=args.top_k,
                      auto_train=not args.no_train, epochs=args.epochs, demo=demo)
    if demo:
        root.after(1500, app.run_demo)
    root.mainloop()


if __name__ == "__main__":
    main()
