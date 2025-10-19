#!/usr/bin/env python3
"""
优雅、安全的数据替换工具
- 使用 0-based 行号索引
- 严格的内容验证，避免替换错误
- 详细的统计和日志
- 生成新文件，不覆盖原数据
"""

import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ==================== 配置 ====================

@dataclass
class Config:
    """配置类，集中管理所有路径"""
    original_file: Path = Path("/lustre/projects/polyullm/caishuo/cs_data/slime_rl/polaris-data-53K.jsonl")
    new_data_file: Path = Path("analysis/polaris-data-53K__choose_ultra.jsonl")  # 使用 ultra 版本
    old_backup_file: Path = Path("analysis/polaris-data-53K__choose_old.jsonl")
    output_file: Path = Path("/lustre/projects/polyullm/caishuo/cs_data/slime_rl/polaris-data-53K__patched_v2.jsonl")
    log_file: Path = Path("analysis/update_log.txt")

    def validate(self) -> List[str]:
        """验证必需文件是否存在"""
        errors = []
        if not self.original_file.exists():
            errors.append(f"原始数据文件不存在: {self.original_file}")
        if not self.new_data_file.exists():
            errors.append(f"新数据文件不存在: {self.new_data_file}")
        return errors


# ==================== 数据验证 ====================

class DataValidator:
    """数据验证器，确保替换的安全性"""

    @staticmethod
    def compute_signature(data: dict, keys: List[str] = None) -> str:
        """
        计算数据签名，用于快速比对

        Args:
            data: 要计算签名的数据
            keys: 用于计算签名的键列表，None 表示使用所有键

        Returns:
            MD5 签名字符串
        """
        if keys:
            filtered = {k: data.get(k) for k in keys if k in data}
        else:
            filtered = data

        content = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @staticmethod
    def compare_prompts(orig: dict, old_backup: dict) -> Tuple[bool, str]:
        """
        比较原始数据和备份数据的 prompt 是否匹配

        Returns:
            (是否匹配, 不匹配的原因)
        """
        orig_prompt = orig.get('prompt', [])
        old_prompt = old_backup.get('prompt', [])

        # 提取文本内容进行比较（忽略格式差异）
        def extract_text(prompt_list):
            return ' '.join([
                turn.get('content', '')
                for turn in prompt_list
                if isinstance(turn, dict)
            ]).strip()

        orig_text = extract_text(orig_prompt)
        old_text = extract_text(old_prompt)

        # 归一化比较（去除多余空格）
        orig_normalized = ' '.join(orig_text.split())
        old_normalized = ' '.join(old_text.split())

        # 计算相似度
        if orig_normalized == old_normalized:
            return True, "完全匹配"

        # 允许一定程度的差异（如 LaTeX 格式化）
        similarity = len(set(orig_normalized) & set(old_normalized)) / max(len(orig_normalized), len(old_normalized), 1)

        if similarity > 0.9:
            return True, f"高度相似 ({similarity:.2%})"

        return False, f"内容不匹配 (相似度: {similarity:.2%})"

    @staticmethod
    def validate_new_record(record: dict, expected_index: int) -> Tuple[bool, str]:
        """
        验证新记录的有效性

        Args:
            record: 新记录
            expected_index: 期望的索引

        Returns:
            (是否有效, 错误信息)
        """
        # 检查必需字段
        if 'prompt' not in record:
            return False, "缺少 prompt 字段"

        if 'label' not in record:
            return False, "缺少 label 字段"

        # 检查 prompt 格式
        prompt = record.get('prompt', [])
        if not isinstance(prompt, list) or not prompt:
            return False, "prompt 格式错误或为空"

        # 检查 original_index（如果存在）
        if 'original_index' in record:
            actual_index = record['original_index']
            if actual_index != expected_index:
                return False, f"索引不匹配: 期望 {expected_index}, 实际 {actual_index}"

        return True, "验证通过"


# ==================== 主处理逻辑 ====================

class DataReplacer:
    """数据替换器"""

    def __init__(self, config: Config):
        self.config = config
        self.validator = DataValidator()
        self.stats = Counter()
        self.issues = defaultdict(list)
        self.log_entries = []

    def load_new_records(self) -> Dict[int, dict]:
        """
        加载新数据，建立索引映射

        Returns:
            {行号: 数据记录} 的字典 (0-based)
        """
        new_records = {}

        with self.config.new_data_file.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)

                    # 获取原始索引 (0-based)
                    orig_index = item.get('original_index')
                    if orig_index is None:
                        self.log(f"警告: 第 {line_no} 行缺少 original_index，跳过")
                        self.stats['missing_index'] += 1
                        continue

                    # 转换为整数（确保是 0-based）
                    orig_index = int(orig_index)

                    # 验证新记录
                    valid, msg = self.validator.validate_new_record(item, orig_index)
                    if not valid:
                        self.log(f"警告: 索引 {orig_index} 的新记录验证失败: {msg}")
                        self.stats['invalid_new_record'] += 1
                        continue

                    # 移除 original_index 键（替换时不需要）
                    clean_item = {k: v for k, v in item.items() if k != 'original_index'}

                    # 检查重复
                    if orig_index in new_records:
                        self.log(f"警告: 索引 {orig_index} 重复出现")
                        self.stats['duplicate_index'] += 1

                    new_records[orig_index] = clean_item
                    self.stats['new_records_loaded'] += 1

                except json.JSONDecodeError as e:
                    self.log(f"错误: 第 {line_no} 行 JSON 解析失败: {e}")
                    self.stats['json_error'] += 1

        return new_records

    def load_old_backup(self) -> Dict[int, dict]:
        """
        加载旧备份数据（用于验证）

        Returns:
            {行号: 数据记录} 的字典 (0-based)
        """
        if not self.config.old_backup_file.exists():
            self.log("提示: 未找到旧备份文件，跳过验证")
            return {}

        old_records = {}

        with self.config.old_backup_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    orig_index = int(item.get('original_index', -1))
                    if orig_index >= 0:
                        old_records[orig_index] = item
                except (json.JSONDecodeError, ValueError):
                    pass

        self.log(f"已加载 {len(old_records)} 条旧备份记录")
        return old_records

    def process_replacement(self,
                          new_records: Dict[int, dict],
                          old_records: Dict[int, dict]) -> Tuple[int, int]:
        """
        执行替换操作

        逻辑说明:
        1. 如果行号在 new_records 中 -> 替换为新数据
        2. 如果行号在 old_records 中但不在 new_records 中 -> 删除（说明被筛选掉了）
        3. 其他情况 -> 保持原样

        Returns:
            (替换的记录数, 删除的记录数)
        """
        replaced_count = 0
        removed_count = 0

        # 计算应该被删除的索引集合
        indices_to_remove = set(old_records.keys()) - set(new_records.keys())

        if indices_to_remove:
            self.log(f"检测到 {len(indices_to_remove)} 条记录在备份中但未通过筛选，将被删除:")
            for idx in sorted(list(indices_to_remove)[:10]):  # 只显示前10个
                self.log(f"  - 索引 {idx}")
            if len(indices_to_remove) > 10:
                self.log(f"  ... 还有 {len(indices_to_remove) - 10} 个")

        with self.config.original_file.open('r', encoding='utf-8') as fin, \
             self.config.output_file.open('w', encoding='utf-8') as fout:

            for line_idx, line in enumerate(fin):
                line = line.strip()

                # 空行直接写入
                if not line:
                    fout.write('\n')
                    continue

                # 检查是否应该删除 (在备份中但不在新数据中)
                if line_idx in indices_to_remove:
                    self.log(f"删除: 第 {line_idx} 行 (未通过筛选)")
                    removed_count += 1
                    self.stats['removed'] += 1
                    self.issues['removed_indices'].append(line_idx)
                    continue  # 跳过这一行，不写入输出文件

                # 检查是否需要替换 (0-based index)
                if line_idx in new_records:
                    # 解析原始记录
                    try:
                        orig_record = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.log(f"错误: 原始文件第 {line_idx} 行 JSON 解析失败: {e}")
                        self.stats['orig_json_error'] += 1
                        fout.write(line + '\n')
                        continue

                    # 如果有旧备份，进行验证
                    if line_idx in old_records:
                        match, reason = self.validator.compare_prompts(
                            orig_record,
                            old_records[line_idx]
                        )

                        if not match:
                            self.log(f"警告: 第 {line_idx} 行内容验证失败: {reason}")
                            self.issues['validation_failed'].append({
                                'index': line_idx,
                                'reason': reason
                            })
                            self.stats['validation_warning'] += 1

                    # 执行替换
                    new_record = new_records[line_idx]
                    json.dump(new_record, fout, ensure_ascii=False)
                    fout.write('\n')

                    replaced_count += 1
                    self.stats['replaced'] += 1

                    # 记录替换信息
                    if replaced_count <= 5:  # 只记录前几条
                        self.log(f"替换: 第 {line_idx} 行")

                else:
                    # 保持原样
                    fout.write(line + '\n')
                    self.stats['unchanged'] += 1

        return replaced_count, removed_count

    def log(self, message: str):
        """记录日志"""
        print(message)
        self.log_entries.append(message)

    def save_log(self):
        """保存日志文件"""
        with self.config.log_file.open('w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        print(f"\n日志已保存到: {self.config.log_file}")

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("替换操作摘要")
        print("="*60)

        print(f"\n加载统计:")
        print(f"  新记录加载成功: {self.stats['new_records_loaded']}")
        print(f"  新记录验证失败: {self.stats['invalid_new_record']}")
        print(f"  索引缺失: {self.stats['missing_index']}")
        print(f"  索引重复: {self.stats['duplicate_index']}")

        print(f"\n替换统计:")
        print(f"  成功替换: {self.stats['replaced']}")
        print(f"  删除记录: {self.stats['removed']} (备份中有但未通过筛选)")
        print(f"  保持原样: {self.stats['unchanged']}")
        print(f"  验证警告: {self.stats['validation_warning']}")

        if self.stats['removed'] > 0:
            print(f"\n删除详情:")
            removed_indices = self.issues['removed_indices']
            print(f"  被删除的索引 (前 20 个): {sorted(removed_indices[:20])}")
            if len(removed_indices) > 20:
                print(f"  ... 还有 {len(removed_indices) - 20} 个")

        if self.issues['validation_failed']:
            print(f"\n验证失败的索引 ({len(self.issues['validation_failed'])} 个):")
            for issue in self.issues['validation_failed'][:10]:  # 只显示前10个
                print(f"  索引 {issue['index']}: {issue['reason']}")
            if len(self.issues['validation_failed']) > 10:
                print(f"  ... 还有 {len(self.issues['validation_failed']) - 10} 个")

        print(f"\n最终结果:")
        original_count = self.stats['replaced'] + self.stats['removed'] + self.stats['unchanged']
        final_count = self.stats['replaced'] + self.stats['unchanged']
        print(f"  原始文件记录数: {original_count}")
        print(f"  输出文件记录数: {final_count}")
        print(f"  净减少记录数: {self.stats['removed']}")

        print(f"\n输出文件: {self.config.output_file}")
        print("="*60)


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 初始化配置
    config = Config()

    # 验证配置
    errors = config.validate()
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return 1

    # 创建替换器
    replacer = DataReplacer(config)

    print("开始数据替换流程...")
    print(f"原始文件: {config.original_file}")
    print(f"新数据文件: {config.new_data_file}")
    print(f"输出文件: {config.output_file}")
    print()

    # 加载新数据
    replacer.log("步骤 1: 加载新数据...")
    new_records = replacer.load_new_records()
    replacer.log(f"已加载 {len(new_records)} 条新记录\n")

    # 加载旧备份（用于验证）
    replacer.log("步骤 2: 加载旧备份数据...")
    old_records = replacer.load_old_backup()
    replacer.log("")

    # 执行替换
    replacer.log("步骤 3: 执行替换操作...")
    replaced_count, removed_count = replacer.process_replacement(new_records, old_records)
    replacer.log(f"替换完成: 替换 {replaced_count} 条, 删除 {removed_count} 条\n")

    # 打印摘要
    replacer.print_summary()

    # 保存日志
    replacer.save_log()

    return 0


if __name__ == '__main__':
    exit(main())
