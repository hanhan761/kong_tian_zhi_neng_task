# -*- coding: utf-8 -*-
"""
baike_spider.py —— 新爬虫数据集：百度百科（移动版）

与实验指导书中基于「维基百科 + lxml/XPath」的爬虫相比，本爬虫做了以下改动：

1. 目标站点改为「百度百科移动版」（https://wapbaike.baidu.com）。
   移动版是 Next.js 服务端渲染页面，词条的基本信息栏、分类路径、
   摘要等结构化数据全部以 JSON 内嵌在 <script id="__NEXT_DATA__"> 中。
   因此解析不再依赖易变的 XPath / pd.read_html，而是直接读取 JSON，
   抗页面改版能力更强、字段更规范（属性名与属性值天然成对，不会串列）。

2. 沿用指导书的两段式思想，但把「列表页 → 详情页」两级抓取改为
   「种子列表 → 详情页 → 内链滚动扩展」，只抓详情页，且每页只抓一次：
       load_seed()          读取种子装备
       get_equipment_info() 抓取并解析一个词条
       run()                顺序抓取 + 断点续存 + 内链扩展
   上一版实现里「扩展取样」会重复抓取已抓过的页面，本版改为直接复用
   已解析结果中的超链接（前型/级、后型/级、衍生型、型号…），请求量减半。

3. 输出：
       url_equipment_list.json  装备名 -> 详情页 url
       raw_data.json            结构化装备数据（后续建图直接使用）
       crawl_log.txt            抓取日志

运行：python baike_spider.py
"""
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

from config import Config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def log(msg, log_path=None, quiet=False):
    """同时打印到控制台与日志文件（flush 保证被重定向时也能实时看到）"""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    if not quiet:
        print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def make_headers(config):
    return {
        "User-Agent": config.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": config.base_url + "/",
        "Connection": "close",
    }


def get_page_text(url, config, retry=None):
    """带重试的页面下载（对应指导书 utils.get_page_text）"""
    retry = config.max_retry if retry is None else retry
    headers = make_headers(config)
    last_err = None
    for attempt in range(1, retry + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=config.timeout) as resp:
                return resp.read().decode("utf-8", "ignore")
        except Exception as e:                      # noqa: BLE001
            last_err = e
            if attempt < retry:
                time.sleep(config.long_waiting_time)
    raise RuntimeError(f"下载失败({retry}次): {url} -> {last_err}")


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def _collect_text(node, out):
    if isinstance(node, dict):
        if node.get("tag") in ("text", "innerlink") and isinstance(node.get("text"), str):
            out.append(node["text"])
            return
        for value in node.values():
            _collect_text(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_text(value, out)


def _collect_innerlinks(node, out):
    if isinstance(node, dict):
        if node.get("tag") == "innerlink" and isinstance(node.get("text"), str):
            out.append({"name": node["text"], "lemma_id": node.get("lemmaId")})
        for value in node.values():
            _collect_innerlinks(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_innerlinks(value, out)


def parse_card(card):
    """
    把 __NEXT_DATA__ 里的 modules.card（基本信息栏）转换成
    [(属性名, [值, ...]), ...] 的有序列表，并返回每条属性里的内链。
    """
    attributes, links = [], {}
    if not isinstance(card, dict):
        return attributes, links

    for item in card.get("content") or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        texts, inner = [], []
        _collect_text(item.get("data"), texts)
        _collect_innerlinks(item.get("data"), inner)

        cleaned = []
        for t in texts:
            t = t.strip().strip("、，,")
            if t and t not in cleaned:
                cleaned.append(t)
        if not cleaned:
            continue
        attributes.append((title, cleaned))
        if inner:
            links[title] = inner
    return attributes, links


def parse_lemma(html, url):
    """解析单个词条页，返回结构化字典；解析失败返回 None"""
    m = NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        page = data["props"]["pageProps"]["pageData"]
    except Exception:                                   # noqa: BLE001
        return None

    name = page.get("lemmaTitle") or ""
    if not name:
        return None

    ext = page.get("extData") or {}
    classify = ext.get("classify") or []
    kpd = ext.get("kpdClassify") or []
    category_path = classify[0].get("path") if classify else ""
    subject = kpd[0].get("name") if kpd else ""

    abstract_parts = []
    for block in page.get("abstract") or []:
        if isinstance(block, dict):
            if isinstance(block.get("text"), str):
                abstract_parts.append(block["text"])
            for sub in block.get("content") or []:
                if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                    abstract_parts.append(sub["text"])
    abstract = "".join(abstract_parts).strip()

    attributes, links = parse_card(page.get("modules", {}).get("card"))

    return {
        "name": name,
        "lemma_id": page.get("lemmaId"),
        "url": url,
        "category_path": category_path,
        "subject": subject,
        "abstract": abstract,
        "attributes": {k: v for k, v in attributes},
        "attribute_order": [k for k, _ in attributes],
        "innerlinks": links,
        "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_seed(config):
    with open(config.seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["name"], item.get("kind", "")) for item in data.get("seed") or []]


def _save(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run(config):
    random.seed(config.random_seed)
    if os.path.exists(config.log_path):
        os.remove(config.log_path)

    log("=" * 64, config.log_path)
    log(f"目标站点：{config.site_name}  ({config.base_url})", config.log_path)
    log("=" * 64, config.log_path)

    # ---------------- 种子 -> url 列表 ----------------
    seed = load_seed(config)
    url_dict, kind_dict = {}, {}
    for name, kind in seed:
        url_dict[name] = config.item_url_tpl.format(name=urllib.parse.quote(name))
        kind_dict[name] = kind
    _save(config.url_list_path, {"url_list": url_dict, "kind": kind_dict})
    log(f"种子列表载入完成：{len(url_dict)} 条装备", config.log_path)

    expand_keys = [k.strip() for k in config.expand_keys.split(",") if k.strip()]

    # ---------------- 抓取（单次遍历 + 内链滚动扩展）----------------
    results, seen = [], set()
    frontier = list(url_dict.items())        # [(name, url), ...]
    round_no, expanded, total = 0, 0, len(frontier)
    done = 0
    started = time.time()

    while frontier:
        next_frontier = []
        for name, url in frontier:
            if name in seen:
                continue
            seen.add(name)
            done += 1

            record = None
            try:
                record = parse_lemma(get_page_text(url, config), url)
            except Exception as e:                      # noqa: BLE001
                log(f"[{done}/{total}] 抓取异常 {name}: {e}", config.log_path)
                time.sleep(config.long_waiting_time)

            if record is not None:
                record["query_name"] = name
                record["kind"] = kind_dict.get(name, "")
                results.append(record)
                log(f"[{done}/{total}] OK   {name:<16} "
                    f"属性 {len(record['attributes']):>2} 条 | "
                    f"{record['category_path'] or '-'}", config.log_path)

                # 只从第一轮（种子）结果里做一次内链扩展
                if config.enable_expand and round_no == 0:
                    for key, links in record["innerlinks"].items():
                        if not any(key.startswith(k) for k in expand_keys):
                            continue
                        for link in links:
                            new_name = (link["name"] or "").strip()
                            if (not new_name or new_name in seen
                                    or not (config.min_entity_len <= len(new_name) <= 20)
                                    or expanded >= config.max_expand):
                                continue
                            next_frontier.append(
                                (new_name,
                                 config.item_url_tpl.format(
                                     name=urllib.parse.quote(new_name))))
                            seen.add(new_name)   # 先占位，防止重复入队
                            kind_dict[new_name] = kind_dict.get(name, "")
                            expanded += 1
                            if expanded >= config.max_expand:
                                break
                        if expanded >= config.max_expand:
                            break
            else:
                log(f"[{done}/{total}] SKIP {name}（页面无 __NEXT_DATA__ 或词条不存在）",
                    config.log_path)

            if len(results) and len(results) % config.save_num == 0:
                _save(config.raw_data_path, results)
                log(f"  已断点续存 {len(results)} 条 -> {os.path.basename(config.raw_data_path)}",
                    config.log_path)

            time.sleep(config.waiting_time + random.random() * config.request_delay_jitter)

        total += len(next_frontier)
        if next_frontier:
            url_dict.update(dict(next_frontier))
            _save(config.url_list_path, {"url_list": url_dict, "kind": kind_dict})
            log(f"内链扩展：第 {round_no + 1} 轮新增 {len(next_frontier)} 条 -> 队列 {total} 条",
                config.log_path)
        frontier = next_frontier
        round_no += 1

    _save(config.raw_data_path, results)

    cost = time.time() - started
    attr_total = sum(len(r["attributes"]) for r in results)
    log("-" * 64, config.log_path)
    log(f"爬取完成！成功 {len(results)} / {total} 条（内链扩展新增 {expanded} 条）",
        config.log_path)
    log(f"属性总数 {attr_total} 条，平均 {attr_total / max(1, len(results)):.1f} 条/装备",
        config.log_path)
    log(f"耗时 {cost:.1f}s，数据已写入 {config.raw_data_path}", config.log_path)
    return results


if __name__ == "__main__":
    run(Config())
