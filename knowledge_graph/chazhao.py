import codecs
import numpy as np
import copy
import time
import random
from tqdm import tqdm
import os
import pickle
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 全局变量
entities2id = {}
id2entities = {}
relations2id = {}
id2relations = {}


def dataloader(file1, file2, file3):
    entity = []
    relation = []

    # 加载实体ID
    with open(file2, 'r', encoding='utf-8') as f1:
        for line in f1:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                entities2id[parts[0]] = parts[1]
                id2entities[parts[1]] = parts[0]
                entity.append(parts[1])

    # 加载关系ID
    with open(file3, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                relations2id[parts[0]] = parts[1]
                id2relations[parts[1]] = parts[0]
                relation.append(parts[1])

    # 加载三元组
    triple_list = []
    with open(file1, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                if h in entities2id and r in relations2id and t in entities2id:
                    triple_list.append([entities2id[h], relations2id[r], entities2id[t]])
                else:
                    missing = []
                    if h not in entities2id:
                        missing.append(f"头实体 '{h}'")
                    if r not in relations2id:
                        missing.append(f"关系 '{r}'")
                    if t not in entities2id:
                        missing.append(f"尾实体 '{t}'")
                    print(f"警告：跳过三元组 {line}，因为 {'、'.join(missing)} 不存在")

    print(f"\n最终统计：")
    print(f"实体字典大小: {len(entities2id)}")
    print(f"关系字典大小: {len(relations2id)}")
    print(f"有效三元组数: {len(triple_list)}")

    return entity, relation, triple_list


def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))


class TransH:
    def __init__(self, entity, relation, triple_list, embedding_dim=100, lr=0.01, margin=1.0, norm=2, C=1.0):
        self.entities = entity
        self.relations = relation
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.C = C  # 软约束权重
        self.loss = 0.0
        self.best_loss = float('inf')
        self.early_stopping_patience = 10
        self.patience_counter = 0
        self.stop_training = False
        self.triple_dict = {}
        self.loss_history = []

    def data_initialise(self):
        entityVectorList = {}
        relationVectorList = {}
        normVectorList = {}  # TransH新增：关系的法向量

        # 初始化实体向量
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension),
                                              6.0 / np.sqrt(self.dimension),
                                              self.dimension)
            entityVectorList[entity] = entity_vector

        # 初始化关系向量和法向量
        for relation in self.relations:
            # 关系向量
            relation_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension),
                                                6.0 / np.sqrt(self.dimension),
                                                self.dimension)
            relation_vector = self.normalization(relation_vector)
            relationVectorList[relation] = relation_vector

            # 法向量 (用于定义超平面)
            norm_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension),
                                            6.0 / np.sqrt(self.dimension),
                                            self.dimension)
            norm_vector = self.normalization(norm_vector)
            normVectorList[relation] = norm_vector

        self.entities = entityVectorList
        self.relations = relationVectorList
        self.norm_vectors = normVectorList  # TransH新增：存储法向量

        # 建立快速查找的三元组字典
        self._build_triple_dict()

    def _build_triple_dict(self):
        """构建用于快速查找的三元组字典"""
        self.triple_dict = {}
        for h, r, t in self.triples:
            if h not in self.triple_dict:
                self.triple_dict[h] = {}
            if r not in self.triple_dict[h]:
                self.triple_dict[h][r] = []
            self.triple_dict[h][r].append(t)

    def project_to_hyperplane(self, entity_vec, norm_vec):
        """将实体向量投影到超平面上"""
        return entity_vec - np.dot(entity_vec, norm_vec) * norm_vec

    def find_tail_entities(self, head_entity, relation, top_k=5):
        """
        根据头实体和关系查找尾实体
        参数:
            head_entity: 头实体名称(原始名称，不是ID)
            relation: 关系名称(原始名称，不是ID)
            top_k: 返回最有可能的k个结果
        返回:
            列表，包含(top_k个)可能的尾实体及其得分，按得分排序
        """
        # 转换为ID
        try:
            h_id = entities2id[head_entity]
            r_id = relations2id[relation]
        except KeyError as e:
            return [("错误: 实体或关系不存在", 0.0)]

        # 1. 首先查找已知的三元组
        known_results = []
        if h_id in self.triple_dict and r_id in self.triple_dict[h_id]:
            known_tails = self.triple_dict[h_id][r_id]
            known_results = [(id2entities[t_id], 0.0) for t_id in known_tails]
            if len(known_results) >= top_k:
                return known_results[:top_k]

        # 2. 使用模型预测
        results = []
        h_vec = self.entities[h_id]
        r_vec = self.relations[r_id]
        norm_vec = self.norm_vectors[r_id]

        # TransH: 将头实体投影到超平面
        h_proj = self.project_to_hyperplane(h_vec, norm_vec)

        # 计算所有可能的尾实体得分
        for t_id, t_vec in self.entities.items():
            # TransH: 将尾实体投影到超平面
            t_proj = self.project_to_hyperplane(t_vec, norm_vec)

            if self.norm == 1:
                score = norm_l1(h_proj, r_vec, t_proj)
            else:
                score = norm_l2(h_proj, r_vec, t_proj)

            # 添加软约束惩罚项
            orth_loss = np.sum(np.square(np.dot(r_vec, norm_vec)))
            score += self.C * orth_loss

            results.append((id2entities[t_id], score))

        # 按得分排序，分数越低越好
        results.sort(key=lambda x: x[1])

        # 合并已知结果和预测结果，去重
        final_results = []
        seen_entities = set()

        # 先添加已知结果
        for entity, score in known_results:
            if entity not in seen_entities:
                final_results.append((entity, score))
                seen_entities.add(entity)

        # 再添加预测结果
        for entity, score in results:
            if entity not in seen_entities and len(final_results) < top_k:
                final_results.append((entity, score))
                seen_entities.add(entity)

        return final_results[:top_k]

    def normalization(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def training_run(self, epochs=100, nbatches=100, out_file_title=''):
        if not self.triples:
            print("错误：没有有效的三元组可供训练")
            return

        batch_size = max(1, int(len(self.triples) / nbatches))
        print(f"Batch size: {batch_size}")

        for epoch in range(epochs):
            if self.stop_training:
                print("Early stopping triggered")
                break

            start = time.time()
            self.loss = 0.0
            epoch_loss = 0.0

            # Process batches with progress bar
            for batch in tqdm(range(nbatches), desc=f"Epoch {epoch}"):
                batch_samples = random.sample(self.triples, min(batch_size, len(self.triples)))

                for sample in batch_samples:
                    h, r, t = sample

                    # 获取投影后的向量
                    norm_vec = self.norm_vectors[r]
                    h_proj = self.project_to_hyperplane(self.entities[h], norm_vec)
                    t_proj = self.project_to_hyperplane(self.entities[t], norm_vec)
                    r_vec = self.relations[r]

                    # 计算正样本得分
                    if self.norm == 1:
                        pos_score = norm_l1(h_proj, r_vec, t_proj)
                    else:
                        pos_score = norm_l2(h_proj, r_vec, t_proj)

                    # 添加软约束
                    orth_loss = np.sum(np.square(np.dot(r_vec, norm_vec)))
                    pos_score += self.C * orth_loss

                    # 生成负样本
                    neg_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    if pr > 0.5:
                        # 替换头实体
                        neg_sample[0] = random.sample(list(self.entities.keys()), 1)[0]
                        while neg_sample[0] == sample[0]:
                            neg_sample[0] = random.sample(list(self.entities.keys()), 1)[0]
                    else:
                        # 替换尾实体
                        neg_sample[2] = random.sample(list(self.entities.keys()), 1)[0]
                        while neg_sample[2] == sample[2]:
                            neg_sample[2] = random.sample(list(self.entities.keys()), 1)[0]

                    # 获取负样本的投影向量
                    h_neg_proj = self.project_to_hyperplane(self.entities[neg_sample[0]], norm_vec)
                    t_neg_proj = self.project_to_hyperplane(self.entities[neg_sample[2]], norm_vec)

                    # 计算负样本得分
                    if self.norm == 1:
                        neg_score = norm_l1(h_neg_proj, r_vec, t_neg_proj)
                    else:
                        neg_score = norm_l2(h_neg_proj, r_vec, t_neg_proj)

                    neg_score += self.C * orth_loss

                    # 计算损失
                    loss = max(0, self.margin + pos_score - neg_score)
                    epoch_loss += loss

                    # 如果损失大于0，则更新参数
                    if loss > 0:
                        if self.norm == 1:
                            # L1范数的梯度
                            grad_pos = np.sign(h_proj + r_vec - t_proj)
                            grad_neg = np.sign(h_neg_proj + r_vec - t_neg_proj)
                        else:
                            # L2范数的梯度
                            grad_pos = 2 * (h_proj + r_vec - t_proj)
                            grad_neg = 2 * (h_neg_proj + r_vec - t_neg_proj)

                        # 更新关系向量
                        self.relations[r] -= self.learning_rate * (grad_pos - grad_neg)
                        self.relations[r] = self.normalization(self.relations[r])

                        # 更新实体向量
                        self.entities[h] -= self.learning_rate * grad_pos
                        self.entities[t] -= self.learning_rate * (-grad_pos)
                        self.entities[neg_sample[0]] -= self.learning_rate * (-grad_neg)
                        self.entities[neg_sample[2]] -= self.learning_rate * grad_neg

                        # 更新法向量
                        h_vec = self.entities[h]
                        t_vec = self.entities[t]
                        h_neg_vec = self.entities[neg_sample[0]]
                        t_neg_vec = self.entities[neg_sample[2]]

                        # 法向量的梯度
                        grad_norm_pos = -2 * np.dot(h_vec, norm_vec) * (h_proj + r_vec - t_proj) + \
                                        -2 * np.dot(t_vec, norm_vec) * (h_proj + r_vec - t_proj)
                        grad_norm_neg = -2 * np.dot(h_neg_vec, norm_vec) * (h_neg_proj + r_vec - t_neg_proj) + \
                                        -2 * np.dot(t_neg_vec, norm_vec) * (h_neg_proj + r_vec - t_neg_proj)

                        # 正交约束的梯度
                        orth_grad = 2 * np.dot(r_vec, norm_vec) * r_vec

                        self.norm_vectors[r] -= self.learning_rate * (
                                    grad_norm_pos - grad_norm_neg + self.C * orth_grad)
                        self.norm_vectors[r] = self.normalization(self.norm_vectors[r])

            end = time.time()
            avg_loss = epoch_loss / len(self.triples) if self.triples else 0
            self.loss_history.append(avg_loss)

            print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Time: {end - start:.2f}s")

            # Early stopping
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    self.stop_training = True
                    print(f"Early stopping at epoch {epoch}")

    def evaluate(self, test_triples, all_entities, n_samples=None):
        """
        评估模型性能 - 改进版本
        参数:
            test_triples: 测试三元组列表
            all_entities: 所有实体列表
            n_samples: 采样数量，如果为None则使用全部测试数据
        返回:
            包含MRR, Hits@1, Hits@3, Hits@10的字典
        """
        if not test_triples:
            print("警告：没有测试三元组可供评估")
            return {
                'MRR': float('nan'),
                'Hits@1': float('nan'),
                'Hits@3': float('nan'),
                'Hits@10': float('nan')
            }

        # 如果指定了采样数量，则随机采样
        if n_samples and n_samples < len(test_triples):
            test_triples = random.sample(test_triples, n_samples)

        print(f"开始评估，使用 {len(test_triples)} 个测试三元组")

        # 初始化指标
        ranks = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0

        # 对每个测试三元组进行评估
        for triple in tqdm(test_triples, desc="评估进度"):
            h, r, t = triple

            # 获取正样本得分
            norm_vec = self.norm_vectors[r]
            h_proj = self.project_to_hyperplane(self.entities[h], norm_vec)
            t_proj = self.project_to_hyperplane(self.entities[t], norm_vec)
            r_vec = self.relations[r]

            if self.norm == 1:
                pos_score = norm_l1(h_proj, r_vec, t_proj)
            else:
                pos_score = norm_l2(h_proj, r_vec, t_proj)

            # 评估头实体预测
            head_scores = []
            for entity in all_entities:
                if entity == h:
                    head_scores.append((entity, pos_score))
                    continue

                # 计算负样本得分
                h_neg_proj = self.project_to_hyperplane(self.entities[entity], norm_vec)
                if self.norm == 1:
                    current_score = norm_l1(h_neg_proj, r_vec, t_proj)
                else:
                    current_score = norm_l2(h_neg_proj, r_vec, t_proj)
                head_scores.append((entity, current_score))

            # 排序并计算排名
            head_scores.sort(key=lambda x: x[1])
            head_rank = 1
            for entity, score in head_scores:
                if entity == h:
                    break
                head_rank += 1
            ranks.append(head_rank)

            # 更新Hits@k
            if head_rank <= 1:
                hits_at_1 += 1
            if head_rank <= 3:
                hits_at_3 += 1
            if head_rank <= 10:
                hits_at_10 += 1

            # 评估尾实体预测
            tail_scores = []
            for entity in all_entities:
                if entity == t:
                    tail_scores.append((entity, pos_score))
                    continue

                # 计算负样本得分
                t_neg_proj = self.project_to_hyperplane(self.entities[entity], norm_vec)
                if self.norm == 1:
                    current_score = norm_l1(h_proj, r_vec, t_neg_proj)
                else:
                    current_score = norm_l2(h_proj, r_vec, t_neg_proj)
                tail_scores.append((entity, current_score))

            # 排序并计算排名
            tail_scores.sort(key=lambda x: x[1])
            tail_rank = 1
            for entity, score in tail_scores:
                if entity == t:
                    break
                tail_rank += 1
            ranks.append(tail_rank)

            # 更新Hits@k
            if tail_rank <= 1:
                hits_at_1 += 1
            if tail_rank <= 3:
                hits_at_3 += 1
            if tail_rank <= 10:
                hits_at_10 += 1

        # 计算指标
        mrr = np.mean(1.0 / np.array(ranks))
        hits_at_1 = hits_at_1 / len(ranks)
        hits_at_3 = hits_at_3 / len(ranks)
        hits_at_10 = hits_at_10 / len(ranks)

        return {
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@10': hits_at_10,
            'Ranks': ranks
        }


class TransHApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TransH 知识图谱补全系统 - 增强版")
        self.root.geometry("1000x700")

        # 设置样式
        self.setup_styles()

        # 初始化模型变量
        self.transH = None
        self.entity_set = None
        self.relation_set = None
        self.triple_list = None
        self.evaluation_results = None

        # 创建界面
        self.create_widgets()

        # 自动加载数据
        self.load_data()

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Section.TLabelframe.Label', font=('Arial', 10, 'bold'))
        style.configure('Metric.TLabel', font=('Arial', 9, 'bold'))
        style.configure('Accent.TButton', font=('Arial', 9, 'bold'))

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="TransH 知识图谱补全系统", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # 左侧输入区域
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # 输入区域
        input_frame = ttk.LabelFrame(left_frame, text="知识查询", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="头实体:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.head_entry = ttk.Entry(input_frame, width=30, font=('Arial', 10))
        self.head_entry.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="关系:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.rel_entry = ttk.Entry(input_frame, width=30, font=('Arial', 10))
        self.rel_entry.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        # 绑定回车键
        self.head_entry.bind('<Return>', lambda event: self.search_tail())
        self.rel_entry.bind('<Return>', lambda event: self.search_tail())

        # 按钮区域
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.search_btn = ttk.Button(button_frame, text="🔍 生成尾实体", command=self.search_tail,
                                     style='Accent.TButton')
        self.search_btn.pack(side=tk.LEFT, padx=5)

        self.train_btn = ttk.Button(button_frame, text="⚡ 训练模型", command=self.train_model)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.evaluate_btn = ttk.Button(button_frame, text="📊 评估模型", command=self.evaluate_model)
        self.evaluate_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(button_frame, text="🗑️ 清空", command=self.clear_inputs)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # 结果区域
        result_frame = ttk.LabelFrame(left_frame, text="查询结果", padding="10")
        result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, width=50, font=('Arial', 10))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 评估结果区域
        eval_frame = ttk.LabelFrame(left_frame, text="模型评估结果", padding="10")
        eval_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        eval_frame.columnconfigure(1, weight=1)

        metrics = [
            ('MRR:', 'mrr_var'),
            ('Hits@1:', 'hits1_var'),
            ('Hits@3:', 'hits3_var'),
            ('Hits@10:', 'hits10_var')
        ]

        for i, (label, var_name) in enumerate(metrics):
            ttk.Label(eval_frame, text=label, style='Metric.TLabel').grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            setattr(self, var_name, tk.StringVar(value="未评估"))
            ttk.Label(eval_frame, textvariable=getattr(self, var_name)).grid(row=i, column=1, sticky=tk.W, padx=5,
                                                                             pady=2)

        # 右侧可视化区域
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=2, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # 训练损失图
        self.loss_frame = ttk.LabelFrame(right_frame, text="训练损失曲线", padding="10")
        self.loss_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.loss_frame.columnconfigure(0, weight=1)
        self.loss_frame.rowconfigure(0, weight=1)

        self.fig_loss, self.ax_loss = plt.subplots(figsize=(6, 3))
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, self.loss_frame)
        self.canvas_loss.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 评估结果图
        self.eval_frame = ttk.LabelFrame(right_frame, text="评估结果可视化", padding="10")
        self.eval_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.eval_frame.columnconfigure(0, weight=1)
        self.eval_frame.rowconfigure(0, weight=1)

        self.fig_eval, self.ax_eval = plt.subplots(figsize=(6, 3))
        self.canvas_eval = FigureCanvasTkAgg(self.fig_eval, self.eval_frame)
        self.canvas_eval.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 系统信息区域
        info_frame = ttk.LabelFrame(main_frame, text="系统信息", padding="10")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        info_frame.columnconfigure(1, weight=1)

        self.info_vars = {}
        info_items = [
            ('实体数量:', 'entity_count'),
            ('关系数量:', 'relation_count'),
            ('三元组数量:', 'triple_count'),
            ('模型状态:', 'model_status'),
            ('训练轮数:', 'training_epochs')
        ]

        for i, (label, var_name) in enumerate(info_items):
            ttk.Label(info_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            self.info_vars[var_name] = tk.StringVar(value="加载中...")
            ttk.Label(info_frame, textvariable=self.info_vars[var_name]).grid(row=i, column=1, sticky=tk.W, padx=5,
                                                                              pady=2)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))

    def set_status(self, message):
        """设置状态栏消息"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_info(self):
        """更新系统信息"""
        if self.entity_set:
            self.info_vars['entity_count'].set(str(len(self.entity_set)))
        if self.relation_set:
            self.info_vars['relation_count'].set(str(len(self.relation_set)))
        if self.triple_list:
            self.info_vars['triple_count'].set(str(len(self.triple_list)))
        self.info_vars['model_status'].set("已初始化" if self.transH else "未初始化")
        if hasattr(self.transH, 'loss_history'):
            self.info_vars['training_epochs'].set(str(len(self.transH.loss_history)))

    def clear_inputs(self):
        """清空输入框"""
        self.head_entry.delete(0, tk.END)
        self.rel_entry.delete(0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.set_status("已清空输入和结果")

    def show_result(self, result):
        """在结果区域显示结果"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)
        self.result_text.see(tk.END)

    def update_evaluation_results(self, metrics):
        """更新评估结果显示"""
        self.mrr_var.set(f"{metrics['MRR']:.4f}")
        self.hits1_var.set(f"{metrics['Hits@1']:.4f}")
        self.hits3_var.set(f"{metrics['Hits@3']:.4f}")
        self.hits10_var.set(f"{metrics['Hits@10']:.4f}")

        # 更新评估结果图
        self.plot_evaluation_results(metrics)

    def plot_evaluation_results(self, metrics):
        """绘制评估结果图"""
        self.ax_eval.clear()

        metrics_names = ['Hits@1', 'Hits@3', 'Hits@10', 'MRR']
        metrics_values = [
            metrics['Hits@1'],
            metrics['Hits@3'],
            metrics['Hits@10'],
            metrics['MRR']
        ]

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = self.ax_eval.bar(metrics_names, metrics_values, color=colors, alpha=0.7)

        # 在柱子上显示数值
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            self.ax_eval.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                              f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        self.ax_eval.set_ylabel('得分')
        self.ax_eval.set_ylim(0, 1)
        self.ax_eval.set_title('模型评估指标', fontweight='bold')
        self.ax_eval.grid(True, alpha=0.3)

        self.canvas_eval.draw()

    def plot_training_loss(self):
        """绘制训练损失曲线"""
        if not hasattr(self.transH, 'loss_history') or not self.transH.loss_history:
            return

        self.ax_loss.clear()
        epochs = range(1, len(self.transH.loss_history) + 1)

        self.ax_loss.plot(epochs, self.transH.loss_history, 'b-', linewidth=2, label='训练损失')
        self.ax_loss.set_xlabel('训练轮数')
        self.ax_loss.set_ylabel('损失')
        self.ax_loss.set_title('训练损失曲线', fontweight='bold')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)

        self.canvas_loss.draw()

    def load_data(self):
        """加载数据"""
        self.set_status("正在加载数据...")

        # 文件路径
        file1 = "output/triples.txt"
        file2 = "output/entityid.txt"
        file3 = "output/relationid.txt"

        # 检查文件是否存在
        if not all(os.path.exists(f) for f in [file1, file2, file3]):
            self.set_status("错误：数据文件不存在")
            messagebox.showerror("错误", "数据文件不存在，请确保以下文件存在:\n" +
                                 f"  - {file1}\n" +
                                 f"  - {file2}\n" +
                                 f"  - {file3}")
            return

        try:
            self.entity_set, self.relation_set, self.triple_list = dataloader(file1, file2, file3)

            if not self.triple_list:
                self.set_status("错误：没有有效的三元组")
                messagebox.showerror("错误", "没有有效的三元组可供训练，请检查数据文件")
                return

            # 初始化TransH模型
            self.transH = TransH(self.entity_set, self.relation_set, self.triple_list,
                                 embedding_dim=100, lr=0.01, margin=1.0, norm=2, C=0.1)
            self.transH.data_initialise()

            self.update_info()
            self.set_status(
                f"数据加载完成! 实体: {len(self.entity_set)}, 关系: {len(self.relation_set)}, 三元组: {len(self.triple_list)}")

        except Exception as e:
            self.set_status(f"加载数据时发生错误: {str(e)}")
            messagebox.showerror("错误", f"加载数据时发生错误: {str(e)}")

    def train_model(self):
        """训练模型"""
        if not self.transH:
            messagebox.showerror("错误", "请先加载数据!")
            return

        self.set_status("开始训练TransH模型...")

        # 在新线程中运行训练，避免界面冻结
        import threading
        def train_thread():
            try:
                self.transH.training_run(epochs=100, nbatches=100)
                self.set_status("TransH模型训练完成!")
                self.update_info()
                self.plot_training_loss()
                messagebox.showinfo("完成", "TransH模型训练完成!")
            except Exception as e:
                self.set_status(f"训练过程中发生错误: {str(e)}")
                messagebox.showerror("错误", f"训练过程中发生错误: {str(e)}")

        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()

    def evaluate_model(self):
        """评估模型性能"""
        if not self.transH:
            messagebox.showerror("错误", "请先加载数据!")
            return

        self.set_status("开始评估TransH模型...")

        # 在新线程中运行评估，避免界面冻结
        import threading
        def evaluate_thread():
            try:
                # 使用训练数据的一部分作为测试数据
                test_size = min(200, len(self.triple_list) // 3)
                test_triples = random.sample(self.triple_list, test_size)

                # 获取所有实体ID
                all_entities = list(self.transH.entities.keys())

                # 评估模型
                metrics = self.transH.evaluate(test_triples, all_entities)

                # 更新界面显示
                self.evaluation_results = metrics
                self.update_evaluation_results(metrics)
                self.set_status("TransH模型评估完成!")

                # 显示详细评估结果
                result_text = f"TransH模型评估结果 (使用 {test_size} 个测试样本):\n\n"
                result_text += f"MRR: {metrics['MRR']:.4f}\n"
                result_text += f"Hits@1: {metrics['Hits@1']:.4f}\n"
                result_text += f"Hits@3: {metrics['Hits@3']:.4f}\n"
                result_text += f"Hits@10: {metrics['Hits@10']:.4f}\n\n"

                # 添加排名统计
                ranks = metrics.get('Ranks', [])
                if ranks:
                    result_text += f"平均排名: {np.mean(ranks):.2f}\n"
                    result_text += f"中位数排名: {np.median(ranks):.2f}\n"
                    result_text += f"最佳排名: {min(ranks)}\n"

                self.show_result(result_text)

            except Exception as e:
                self.set_status(f"评估过程中发生错误: {str(e)}")
                messagebox.showerror("错误", f"评估过程中发生错误: {str(e)}")

        thread = threading.Thread(target=evaluate_thread)
        thread.daemon = True
        thread.start()

    def search_tail(self):
        """查找尾实体"""
        if not self.transH:
            messagebox.showerror("错误", "请先加载数据!")
            return

        head = self.head_entry.get().strip()
        rel = self.rel_entry.get().strip()

        if not head or not rel:
            messagebox.showwarning("警告", "请输入头实体和关系!")
            return

        self.set_status(f"正在查找: 头实体='{head}', 关系='{rel}'")

        try:
            results = self.transH.find_tail_entities(head, rel, top_k=10)

            if results and not results[0][0].startswith("错误"):
                result_text = f"头实体: {head}\n关系: {rel}\n\n最可能的尾实体:\n\n"
                for i, (entity, score) in enumerate(results, 1):
                    result_text += f"{i}. {entity}\n   置信度: {1 / (1 + score):.4f}\n\n"

                self.show_result(result_text)
                self.set_status(f"找到 {len(results)} 个可能的尾实体")
            else:
                error_msg = results[0][0] if results else "没有找到匹配的尾实体"
                self.show_result(error_msg)
                self.set_status(error_msg)

        except Exception as e:
            error_msg = f"查找过程中发生错误: {str(e)}"
            self.show_result(error_msg)
            self.set_status(error_msg)
            messagebox.showerror("错误", error_msg)


if __name__ == '__main__':
    # 创建主窗口
    root = tk.Tk()
    app = TransHApp(root)
    root.mainloop()