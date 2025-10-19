#!/usr/bin/env python3
"""
从数据集中移除 hard cases（问题样本）
- 读取 hard cases 列表
- 从源数据中移除对应的 original_index
- 使用模糊匹配验证内容一致性
- 生成清理后的数据集
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Optional
from difflib import SequenceMatcher


class HardCaseRemover:
    """Hard Case 移除器"""

    def __init__(self):
        # 配置路径
        self.hard_cases_file = Path(
            "/Users/shuocai/Downloads/home/projects/polyullm/kejing/slime_workspace/"
            "slime_polaris/slime/analysis/polaris_hard_cases_issues.jsonl"
        )
        self.source_file = Path(
            "/lustre/projects/polyullm/caishuo/cs_data/slime_rl/"
            "polaris-data-53K-indexed.jsonl"
        )
        self.output_file = Path(
            "/lustre/projects/polyullm/caishuo/cs_data/slime_rl/"
            "polaris-data-53K-indexed__clean.jsonl"
        )
        self.log_file = Path("analysis/remove_hard_cases_log.txt")

        # 模糊匹配配置
        self.similarity_threshold = 0.7  # 内容相似度阈值
        self.prompt_preview_length = 200  # 用于比较的 prompt 预览长度

        self.stats = Counter()
        self.log_entries = []
        self.fuzzy_warnings = []

    def load_hard_case_indices(self) -> Dict[int, dict]:
        """
        加载 hard cases 的 original_index 及其详细信息

        Returns:
            {original_index: hard_case_record} 字典
        """
        hard_cases = {}

        if not self.hard_cases_file.exists():
            self.log(f"错误: Hard cases 文件不存在: {self.hard_cases_file}")
            return hard_cases

        self.log(f"读取 hard cases 文件: {self.hard_cases_file}")

        with self.hard_cases_file.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    original_index = record.get('original_index')

                    if original_index is None:
                        self.log(f"警告: Hard cases 第 {line_no} 行缺少 original_index")
                        self.stats['hard_missing_index'] += 1
                        continue

                    original_index = int(original_index)
                    hard_cases[original_index] = record
                    self.stats['hard_cases_loaded'] += 1

                    # 记录问题类型（用于日志）
                    issue = record.get('issue', 'Unknown issue')
                    if line_no <= 10:  # 只记录前10个
                        self.log(f"  - index {original_index}: {issue[:80]}")

                except (json.JSONDecodeError, ValueError) as e:
                    self.log(f"错误: Hard cases 第 {line_no} 行解析失败: {e}")
                    self.stats['hard_parse_error'] += 1

        return hard_cases

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            相似度分数 (0-1)
        """
        return SequenceMatcher(None, text1, text2).ratio()

    def extract_prompt_text(self, record: dict, is_hard_case: bool = False) -> str:
        """
        从记录中提取 prompt 文本用于比较

        Args:
            record: 数据记录
            is_hard_case: 是否为 hard case 记录（使用 prompt_full 字段）

        Returns:
            提取的文本内容
        """
        if is_hard_case:
            # Hard case 使用 prompt_full 字段
            prompt_text = record.get('prompt_full', '')
            return str(prompt_text)[:self.prompt_preview_length]
        else:
            # 源数据使用 prompt 列表结构
            prompt = record.get('prompt', [])
            if isinstance(prompt, list):
                contents = []
                for turn in prompt:
                    if isinstance(turn, dict):
                        content = turn.get('content', '')
                        if content:
                            contents.append(str(content))
                return ' '.join(contents)[:self.prompt_preview_length]
            else:
                return str(prompt)[:self.prompt_preview_length]

    def verify_hard_case_match(
        self,
        original_index: int,
        source_record: dict,
        hard_case_record: dict
    ) -> Tuple[bool, Optional[str]]:
        """
        验证源记录是否与 hard case 记录匹配

        Args:
            original_index: 索引号
            source_record: 源数据记录
            hard_case_record: hard case 记录

        Returns:
            (是否匹配, 警告信息)
        """
        # 提取 prompt 内容
        source_prompt = self.extract_prompt_text(source_record, is_hard_case=False)
        hard_prompt = self.extract_prompt_text(hard_case_record, is_hard_case=True)

        # 如果都为空，认为匹配
        if not source_prompt and not hard_prompt:
            return True, None

        # 计算相似度
        similarity = self.calculate_similarity(source_prompt, hard_prompt)

        if similarity >= self.similarity_threshold:
            return True, None
        else:
            warning = (
                f"索引 {original_index} 内容相似度低 ({similarity:.2f}):\n"
                f"  源数据: {source_prompt[:100]}...\n"
                f"  Hard case: {hard_prompt[:100]}..."
            )
            return False, warning

    def remove_hard_cases(self, hard_cases: Dict[int, dict]) -> int:
        """
        从源文件中移除 hard cases（带模糊匹配验证）

        Args:
            hard_cases: {original_index: hard_case_record} 字典

        Returns:
            实际移除的数量
        """
        if not self.source_file.exists():
            self.log(f"错误: 源文件不存在: {self.source_file}")
            return 0

        self.log(f"\n读取源文件: {self.source_file}")
        self.log(f"输出文件: {self.output_file}")

        removed_count = 0
        kept_count = 0
        verified_removal_count = 0
        unverified_removal_count = 0

        with self.source_file.open('r', encoding='utf-8') as fin, \
             self.output_file.open('w', encoding='utf-8') as fout:

            for physical_line, line in enumerate(fin):
                line = line.strip()

                # 空行直接写入
                if not line:
                    fout.write('\n')
                    continue

                try:
                    record = json.loads(line)
                    original_index = record.get('original_index')

                    if original_index is None:
                        # 缺少索引，保留（并记录警告）
                        self.log(f"警告: 源文件物理行 {physical_line} 缺少 original_index，保留")
                        self.stats['source_missing_index'] += 1
                        fout.write(line + '\n')
                        kept_count += 1
                        continue

                    original_index = int(original_index)

                    # 检查是否在移除列表中
                    if original_index in hard_cases:
                        # 模糊匹配验证
                        hard_case_record = hard_cases[original_index]
                        is_match, warning = self.verify_hard_case_match(
                            original_index, record, hard_case_record
                        )

                        if is_match:
                            # 验证通过，移除
                            removed_count += 1
                            verified_removal_count += 1
                            self.stats['removed'] += 1
                            self.stats['verified_removal'] += 1

                            # 只记录前几个
                            if removed_count <= 10:
                                self.log(f"✓ 移除 (已验证): index {original_index}")
                            elif removed_count == 11:
                                self.log(f"... (后续移除不再详细记录)")

                        else:
                            # 验证未通过，记录警告但仍移除（可配置）
                            self.log(f"⚠️  警告: {warning}")
                            self.fuzzy_warnings.append(warning)
                            removed_count += 1
                            unverified_removal_count += 1
                            self.stats['removed'] += 1
                            self.stats['unverified_removal'] += 1

                            if unverified_removal_count <= 5:
                                self.log(f"⚠️  移除 (未验证): index {original_index}")

                    else:
                        # 保留
                        fout.write(line + '\n')
                        kept_count += 1
                        self.stats['kept'] += 1

                except (json.JSONDecodeError, ValueError) as e:
                    # JSON 解析错误，保留原样（并记录）
                    self.log(f"错误: 源文件物理行 {physical_line} 解析失败: {e}，保留原样")
                    self.stats['source_parse_error'] += 1
                    fout.write(line + '\n')
                    kept_count += 1

        self.log(f"\n处理完成:")
        self.log(f"  保留: {kept_count} 条")
        self.log(f"  移除总数: {removed_count} 条")
        self.log(f"    - 已验证移除: {verified_removal_count} 条")
        if unverified_removal_count > 0:
            self.log(f"    - ⚠️  未验证移除: {unverified_removal_count} 条")

        return removed_count

    def verify_removal(self, hard_cases: Dict[int, dict]):
        """
        验证移除操作是否成功

        Args:
            hard_cases: 应该被移除的 hard cases 字典
        """
        if not self.output_file.exists():
            self.log("错误: 输出文件不存在，无法验证")
            return

        self.log(f"\n验证输出文件: {self.output_file}")

        found_hard_cases = []
        total_records = 0

        with self.output_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    original_index = record.get('original_index')

                    if original_index is not None:
                        total_records += 1
                        original_index = int(original_index)

                        # 检查是否还存在应该被移除的索引
                        if original_index in hard_cases:
                            found_hard_cases.append(original_index)

                except (json.JSONDecodeError, ValueError):
                    pass

        if found_hard_cases:
            self.log(f"⚠️  警告: 发现 {len(found_hard_cases)} 个应该被移除但仍存在的索引:")
            self.log(f"  {found_hard_cases[:20]}")
            self.stats['verification_failed'] += len(found_hard_cases)
        else:
            self.log(f"✅ 验证通过: 所有 hard cases 已被移除")
            self.log(f"  输出文件总记录数: {total_records}")
            self.stats['verification_passed'] = 1

    def log(self, message: str):
        """记录日志"""
        print(message)
        self.log_entries.append(message)

    def save_log(self):
        """保存日志到文件"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open('w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        print(f"\n日志已保存: {self.log_file}")

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("Hard Cases 移除操作摘要（带模糊匹配验证）")
        print("="*60)

        print(f"\nHard Cases 加载:")
        print(f"  成功加载: {self.stats['hard_cases_loaded']}")
        print(f"  索引缺失: {self.stats['hard_missing_index']}")
        print(f"  解析错误: {self.stats['hard_parse_error']}")

        print(f"\n移除统计:")
        print(f"  成功移除: {self.stats['removed']}")
        if self.stats['verified_removal'] > 0 or self.stats['unverified_removal'] > 0:
            print(f"    - ✓ 已验证移除: {self.stats['verified_removal']}")
            if self.stats['unverified_removal'] > 0:
                print(f"    - ⚠️  未验证移除: {self.stats['unverified_removal']}")
        print(f"  保留记录: {self.stats['kept']}")

        if self.stats['source_missing_index'] > 0:
            print(f"\n源文件警告:")
            print(f"  缺少索引: {self.stats['source_missing_index']}")

        if self.fuzzy_warnings:
            print(f"\n⚠️  模糊匹配警告 (共 {len(self.fuzzy_warnings)} 个):")
            for i, warning in enumerate(self.fuzzy_warnings[:3], 1):
                print(f"\n  警告 {i}:")
                for line in warning.split('\n'):
                    print(f"    {line}")
            if len(self.fuzzy_warnings) > 3:
                print(f"\n  ... 还有 {len(self.fuzzy_warnings) - 3} 个警告")

        if self.stats['verification_failed'] > 0:
            print(f"\n⚠️  验证失败:")
            print(f"  未成功移除: {self.stats['verification_failed']}")
        elif self.stats['verification_passed'] > 0:
            print(f"\n✅ 验证通过: 所有 hard cases 已被移除")

        print(f"\n最终结果:")
        original_count = self.stats['removed'] + self.stats['kept']
        final_count = self.stats['kept']
        print(f"  原始记录数: {original_count}")
        print(f"  输出记录数: {final_count}")
        print(f"  净减少: {self.stats['removed']} 条 ({self.stats['removed'] / max(original_count, 1) * 100:.2f}%)")

        print(f"\n配置:")
        print(f"  相似度阈值: {self.similarity_threshold}")
        print(f"  比较长度: {self.prompt_preview_length} 字符")

        print(f"\n输出文件: {self.output_file}")
        print("="*60)


def main():
    """主函数"""
    remover = HardCaseRemover()

    print("开始移除 Hard Cases（带模糊匹配验证）...")
    print()

    # 步骤 1: 加载 hard cases 索引和详细信息
    remover.log("步骤 1: 加载 hard cases 索引及详细信息")
    hard_cases = remover.load_hard_case_indices()

    if not hard_cases:
        remover.log("\n错误: 未找到任何 hard cases 索引")
        return 1

    remover.log(f"\n共需要移除 {len(hard_cases)} 个 hard cases")
    remover.log(f"索引列表: {sorted(hard_cases.keys())}")

    # 步骤 2: 移除 hard cases（带模糊匹配验证）
    remover.log(f"\n步骤 2: 从源文件中移除 hard cases（相似度阈值: {remover.similarity_threshold}）")
    removed_count = remover.remove_hard_cases(hard_cases)

    if removed_count == 0:
        remover.log("\n警告: 未移除任何记录")

    # 步骤 3: 验证移除结果
    remover.log("\n步骤 3: 验证移除结果")
    remover.verify_removal(hard_cases)

    # 打印摘要
    remover.print_summary()

    # 保存日志
    remover.save_log()

    if remover.stats['unverified_removal'] > 0:
        print(f"\n⚠️  注意: 有 {remover.stats['unverified_removal']} 个索引的内容相似度低于阈值，但仍被移除")
        print(f"    请检查日志文件以获取详细信息: {remover.log_file}")

    print("\n✅ 完成!")
    return 0


if __name__ == '__main__':
    exit(main())
