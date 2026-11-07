# -*- coding: utf-8 -*-
"""
config.py —— 新爬虫（百度百科·移动版）配置

与实验指导书中的 config.py 保持一致的设计思路：把所有可变参数集中到
一个 Config 类里，便于维护与命令行覆盖。
"""
import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    """爬虫 + 建图 的全部配置参数"""

    def __init__(self, argv=None):
        args = self.__get_config(argv)
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

    @staticmethod
    def __get_config(argv=None):
        parser = argparse.ArgumentParser(description="config for baike spider")
        # ---------------- 数据文件路径 ----------------
        parser.add_argument("--seed_path", type=str,
                            default=os.path.join(BASE_DIR, "seed_list.json"),
                            help="种子装备列表")
        parser.add_argument("--raw_data_path", type=str,
                            default=os.path.join(BASE_DIR, "raw_data.json"),
                            help="爬取到的原始数据（JSON）")
        parser.add_argument("--url_list_path", type=str,
                            default=os.path.join(BASE_DIR, "url_equipment_list.json"),
                            help="扩展得到的装备名 -> 详情页 url 列表")
        parser.add_argument("--log_path", type=str,
                            default=os.path.join(BASE_DIR, "crawl_log.txt"),
                            help="爬取日志")
        parser.add_argument("--output_dir", type=str,
                            default=os.path.join(BASE_DIR, "output"),
                            help="三元组/实体/关系文件输出目录")

        # ---------------- 站点相关参数 ----------------
        parser.add_argument("--site_name", type=str, default="百度百科(移动版)",
                            help="目标站点名称")
        parser.add_argument("--base_url", type=str,
                            default="https://wapbaike.baidu.com",
                            help="站点根地址")
        parser.add_argument("--item_url_tpl", type=str,
                            default="https://wapbaike.baidu.com/item/{name}",
                            help="词条详情页 url 模板")
        parser.add_argument("--user_agent", type=str,
                            default="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                                    "Version/16.0 Mobile/15E148 Safari/604.1",
                            help="移动端 UA（移动版页面结构稳定、无验证码）")

        # ---------------- 爬取行为参数 ----------------
        parser.add_argument("--waiting_time", type=float, default=1.0,
                            help="每次请求的基础等待时间（秒）")
        parser.add_argument("--long_waiting_time", type=float, default=5.0,
                            help="请求失败后的长等待时间（秒）")
        parser.add_argument("--max_retry", type=int, default=3,
                            help="单页最大重试次数")
        parser.add_argument("--timeout", type=int, default=20,
                            help="单次请求超时（秒）")
        parser.add_argument("--save_num", type=int, default=10,
                            help="每爬取多少条保存一次，防止数据丢失")
        parser.add_argument("--enable_expand", type=int, default=1,
                            help="是否开启内链扩展（1 开启 / 0 关闭）")
        parser.add_argument("--expand_keys", type=str,
                            default="前型/级,后型/级,衍生型,衍生机型,型号,主要型号,出口型,相关型号",
                            help="允许作为扩展入口的属性名，逗号分隔")
        parser.add_argument("--max_expand", type=int, default=40,
                            help="内链扩展最多新增的候选词条数")
        parser.add_argument("--random_seed", type=int, default=2026,
                            help="随机种子（等待时间抖动）")
        parser.add_argument("--request_delay_jitter", type=float, default=0.5,
                            help="等待时间的随机抖动上限（秒）")

        # ---------------- 数据处理参数 ----------------
        parser.add_argument("--max_value_len", type=int, default=30,
                            help="属性值超过该长度视为描述性长文本，不作为尾实体")
        parser.add_argument("--min_entity_len", type=int, default=1,
                            help="实体名最短长度")
        parser.add_argument("--relation_weight", type=int, default=1,
                            help="关系名中的权重后缀是否按尾实体数量动态计算（1 是）")

        args, _ = parser.parse_known_args(argv)
        return args

    def print_config(self):
        print("=" * 60)
        for key in self.__dict__:
            print(f"{key:<24} = {self.__dict__[key]}")
        print("=" * 60)


if __name__ == "__main__":
    Config().print_config()
