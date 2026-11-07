# -*- coding: utf-8 -*-
"""build_dataset.py —— 把爬虫原始数据 raw_data.json 转成知识图谱数据集 + Neo4j 导入脚本

产出（默认 new_output/ 目录，格式与作业给定的 output/ 完全一致，可直接喂给 TransE/TransH 代码）：
    entityid.txt          实体名 \\t 编号
    relationid.txt        关系名 \\t 编号（关系名带 [weight=n] 后缀，n = 该关系在全图连接的不同尾实体数）
    triples.txt           头实体 \\t 关系 \\t 尾实体（均为名称）
    neo4j_import.cypher   Neo4j 批量导入语句
    relation_type_map.txt Neo4j 关系类型与中文关系名的对照表

建模约定（与原作业图谱保持一致）：
    · 节点 = 装备名 + 属性值（属性值也作为实体，才能构成「实体-关系-实体」三元组）
    · 关系 = 百科信息框的属性名，以及内链关系名（如「前型/级」「所属国家」）
    · 关系名后缀 [weight=n] 沿用原数据集写法，n 越大说明该关系越「一对多」，越难预测

用法：
    python build_dataset.py
    python build_dataset.py --raw raw_data.json --out new_output
"""
import argparse
import json
import os
import re
from collections import OrderedDict, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 属性值清洗：过长的描述性文本、纯符号不作为实体
MAX_VALUE_LEN = 40


def clean(text):
    if text is None:
        return ""
    text = str(text).replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_raw(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    return data


def as_pairs(obj):
    """兼容 dict / list-of-pairs / list-of-dict 三种存法"""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.items())
    out = []
    for item in obj:
        if isinstance(item, dict) and "rel" in item and "values" in item:
            out.append((item["rel"], item["values"]))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((item[0], item[1]))
    return out


def build(records):
    """返回 (entities, relations, triples, skipped)

    triples 中的关系名已是带 [weight=n] 后缀的最终形式。
    """
    raw = OrderedDict()          # (头, 关系基名, 尾) 保序去重
    skipped = 0

    for rec in records:
        head = clean(rec.get("name") or rec.get("query_name"))
        if not head:
            continue
        # 1) 信息框属性
        for attr, values in as_pairs(rec.get("attributes")):
            rel = clean(attr)
            if not rel:
                continue
            if isinstance(values, (str, int, float)):
                values = [values]
            for v in values or []:
                tail = clean(v)
                if not tail or tail == head or len(tail) > MAX_VALUE_LEN:
                    skipped += 1
                    continue
                raw[(head, rel, tail)] = True
        # 2) 内链关系（词条内指向其它词条的链接）
        for rel, links in as_pairs(rec.get("innerlinks")):
            rel = clean(rel)
            if not rel:
                continue
            for link in links or []:
                tail = clean(link.get("name") if isinstance(link, dict) else link)
                if not tail or tail == head or len(tail) > MAX_VALUE_LEN:
                    skipped += 1
                    continue
                raw[(head, rel, tail)] = True

    # 关系权重 = 该关系连接的不同尾实体数
    tails_of_rel = defaultdict(set)
    for h, r, t in raw:
        tails_of_rel[r].add(t)
    full_rel = {r: "%s[weight=%d]" % (r, len(tails)) for r, tails in tails_of_rel.items()}
    relations = list(full_rel.values())

    # 实体表：先装备名（按抓取顺序），再补三元组里出现的属性值
    entities = OrderedDict()
    for rec in records:
        name = clean(rec.get("name") or rec.get("query_name"))
        if name:
            entities[name] = True
    for h, r, t in raw:
        entities.setdefault(h, True)
        entities.setdefault(t, True)

    triples = [(h, full_rel[r], t) for h, r, t in raw]
    return list(entities.keys()), relations, triples, skipped


def write_output(ents, rels, triples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}

    with open(os.path.join(out_dir, "entityid.txt"), "w", encoding="utf-8") as f:
        for e in ents:
            f.write("%s\t%d\n" % (e, ent2id[e]))
    with open(os.path.join(out_dir, "relationid.txt"), "w", encoding="utf-8") as f:
        for r in rels:
            f.write("%s\t%d\n" % (r, rel2id[r]))
    with open(os.path.join(out_dir, "triples.txt"), "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write("%s\t%s\t%s\n" % (h, r, t))


def rel_type_name(base):
    """关系基名 -> 合法的 Cypher 关系类型（保留中文，替换括号等特殊字符）"""
    t = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]", "_", base)
    t = re.sub(r"_+", "_", t).strip("_")
    return t if t and not t[0].isdigit() else "R_" + t


def write_cypher(ents, rels, triples, out_dir):
    by_rel = OrderedDict()
    for h, r, t in triples:
        by_rel.setdefault(r, []).append((h, t))

    def q(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "// 由 build_dataset.py 自动生成：Neo4j 3.5 批量导入",
        "// 用法：Neo4j Browser 中 :play 或直接粘贴执行；也可 neo4j-shell -file neo4j_import.cypher",
        "// 注意：第一句会清空库中已有数据",
        "",
        "MATCH (n) DETACH DELETE n;",
        "",
        "// ---- 1. 实体节点 ----",
    ]
    chunk, size = [], 0
    for e in ents:
        item = '{"name": "%s"}' % q(e)
        chunk.append(item)
        size += len(item)
        if size > 80000:
            lines.append("UNWIND [%s] AS row CREATE (:Entity {name: row.name});" % ", ".join(chunk))
            chunk, size = [], 0
    if chunk:
        lines.append("UNWIND [%s] AS row CREATE (:Entity {name: row.name});" % ", ".join(chunk))
    lines.append("CREATE CONSTRAINT ON (e:Entity) ASSERT e.name IS UNIQUE;")
    lines.append("")
    lines.append("// ---- 2. 关系（关系类型见 relation_type_map.txt）----")

    type_map = []
    for r, pairs in by_rel.items():
        base = r.rsplit("[weight=", 1)[0]
        rtype = rel_type_name(base)
        type_map.append((rtype, r, len(pairs)))
        chunk, size = [], 0
        for h, t in pairs:
            item = '{h: "%s", t: "%s"}' % (q(h), q(t))
            chunk.append(item)
            size += len(item)
            if size > 60000:
                lines.append("UNWIND [%s] AS row MATCH (a:Entity {name: row.h}), "
                             "(b:Entity {name: row.t}) CREATE (a)-[:%s]->(b);"
                             % (", ".join(chunk), rtype))
                chunk, size = [], 0
        if chunk:
            lines.append("UNWIND [%s] AS row MATCH (a:Entity {name: row.h}), "
                         "(b:Entity {name: row.t}) CREATE (a)-[:%s]->(b);"
                         % (", ".join(chunk), rtype))
    lines.append("")
    lines.append("// ---- 3. 统计 ----")
    lines.append("MATCH (e:Entity) RETURN count(e) AS 实体数;")
    lines.append("MATCH ()-[r]->() RETURN count(r) AS 关系数;")

    with open(os.path.join(out_dir, "neo4j_import.cypher"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "relation_type_map.txt"), "w", encoding="utf-8") as f:
        f.write("关系类型\t原始关系名\t三元组数\n")
        for rtype, r, n in type_map:
            f.write("%s\t%s\t%d\n" % (rtype, r, n))


def main():
    ap = argparse.ArgumentParser(description="raw_data.json -> 知识图谱数据集 + Neo4j 导入脚本")
    ap.add_argument("--raw", default=os.path.join(BASE_DIR, "raw_data.json"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "new_output"))
    args = ap.parse_args()

    records = load_raw(args.raw)
    ents, rels, triples, skipped = build(records)
    write_output(ents, rels, triples, args.out)
    write_cypher(ents, rels, triples, args.out)

    tails_of_rel = defaultdict(set)
    for h, r, t in triples:
        tails_of_rel[r].add(t)
    top_rels = sorted(tails_of_rel.items(), key=lambda kv: -len(kv[1]))[:5]

    # 统计信息（写盘，便于实验报告与复查引用同一组数字）
    stats = {
        "词条数": len(records),
        "实体数": len(ents),
        "关系数": len(rels),
        "三元组数": len(triples),
        "过滤属性值": skipped,
        "三元组每实体": round(len(triples) / max(len(ents), 1), 2),
        "尾实体最多的关系": [{"关系": r, "尾实体数": len(t)} for r, t in top_rels],
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("词条数        : %d" % len(records))
    print("实体数        : %d" % len(ents))
    print("关系数        : %d" % len(rels))
    print("三元组数      : %d" % len(triples))
    print("过滤属性值    : %d" % skipped)
    print("三元组/实体   : %.2f" % (len(triples) / max(len(ents), 1)))
    print("尾实体最多的关系：")
    for r, tails in top_rels:
        print("    %-24s %d" % (r, len(tails)))
    print("输出目录      : %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
