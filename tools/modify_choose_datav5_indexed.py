#!/usr/bin/env python3
"""
基于索引的选择题提取工具（v5 Ultra 版本）
- 从带 original_index 的源文件中提取
- 保留 original_index 以便后续追踪
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Optional, Tuple, List

# 配置
SRC = Path('analysis/polaris-data-53K-indexed.jsonl')  # 带索引的源文件
DST = Path('analysis/polaris-data-53K__choose_ultra_indexed.jsonl')

if not SRC.exists():
    raise SystemExit(f'Missing source file: {SRC}')

skip_reason = Counter()

# ==================== Pattern Matching (与 v5 相同) ====================

WRAPPED_PAIR = [
    re.compile(r"\\textbf\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\mathrm\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\mathbf\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\textit\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\text\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\bf\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
]

WRAPPED_SINGLE = [
    re.compile(r"\\textbf\s*\{\s*([A-E])\s*\}", re.IGNORECASE),
    re.compile(r"\\mathrm\s*\{\s*([A-E])\s*\}", re.IGNORECASE),
    re.compile(r"\\mathbf\s*\{\s*([A-E])\s*\}", re.IGNORECASE),
    re.compile(r"\\textit\s*\{\s*([A-E])\s*\}", re.IGNORECASE),
    re.compile(r"\\text\s*\{\s*([A-E])\s*\}", re.IGNORECASE),
    re.compile(r"\\bf\s*\{\s*([A-E])\s*\}", re.IGNORECASE),
]

SIMPLE_CMDS = ['mathrm', 'text', 'operatorname', 'mathbf', 'textit', 'textbf', 'bf']

OPTION_PATTERNS = [
    re.compile(r"^\s*(?:\(([A-E])\)|([A-E]))[)\.:：-]?\s+", re.IGNORECASE),
    re.compile(r"^\s*\$\\textbf\{?\(([A-E])\)\}?\$\s*", re.IGNORECASE),
    re.compile(r"^\s*\(\s*([A-E])\s*\)\s*[:\-\.]?\s+", re.IGNORECASE),
    re.compile(r"^\s*([A-E])\s*[\.:：\-]\s+", re.IGNORECASE),
    re.compile(r"^\s*([A-E])\s+(?=[A-Za-z\d\$\(])", re.IGNORECASE),
    re.compile(r"^\s*\$\s*\\textbf\s*\{\s*\(([A-E])\)\s*\}\s*\$", re.IGNORECASE),
    re.compile(r"^\s*\$\s*\\mathrm\s*\{\s*\(([A-E])\)\s*\}\s*\$", re.IGNORECASE),
    re.compile(r"^\s*\$\s*\(([A-E])\)\s*\$", re.IGNORECASE),
    re.compile(r"\s+\$\\textbf\{?\(([A-E])\)\}?\$\s+", re.IGNORECASE),
    re.compile(r"^\s*\(([A-E])\)", re.IGNORECASE),
    re.compile(r"^([A-E])\)", re.IGNORECASE),
]

LABEL_PATTERNS = [
    re.compile(r"\\textbf\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\mathrm\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\\mathbf\s*\{\s*\(([A-E])\)\s*\}", re.IGNORECASE),
    re.compile(r"\(([A-E])\)", re.IGNORECASE),
    re.compile(r"^([A-E])$", re.IGNORECASE),
    re.compile(r"^([A-E])[)\.:：\-]", re.IGNORECASE),
    re.compile(r"^([A-E])\s*$", re.IGNORECASE),
    re.compile(r"^([A-E])[)\s]", re.IGNORECASE),
    re.compile(r"([A-E])(?:\s*,\s*([A-E]))*", re.IGNORECASE),
]


def strip_simple_command(cmd: str, text: str) -> str:
    """Remove LaTeX commands while preserving content."""
    pattern = re.compile(rf"\\{cmd}\s*\{{([^{{}}]*)\}}", re.IGNORECASE)
    iterations = 0
    while iterations < 10:
        new_text, count = pattern.subn(r"\1", text)
        text = new_text
        iterations += 1
        if count == 0:
            break
    return text


def normalize_prompt(raw_text: str) -> str:
    """Normalize prompt by cleaning LaTeX and formatting."""
    text = raw_text.replace('\r', '')
    text = text.replace('\\qquad', '\n').replace('\\quad', ' ')
    text = text.replace('\\\\', '\n')
    text = text.replace('\\newline', '\n')

    for _ in range(3):
        for pat in WRAPPED_PAIR:
            text = pat.sub(lambda m: f"({m.group(1)})", text)
        for pat in WRAPPED_SINGLE:
            text = pat.sub(lambda m: m.group(1), text)

    for cmd in SIMPLE_CMDS:
        text = strip_simple_command(cmd, text)

    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace('$', ' ').replace('~', ' ')
    text = text.replace('{', ' ').replace('}', ' ')

    lines = []
    for line in text.split('\n'):
        clean = re.sub(r"\s+", ' ', line).strip()
        if clean:
            lines.append(clean)
    return '\n'.join(lines)


def try_match_option(line: str):
    """Try to match option marker using multiple patterns."""
    for pattern in OPTION_PATTERNS:
        m = pattern.search(line)
        if m:
            letter = None
            for group in m.groups():
                if group:
                    letter = group.upper()
                    break
            if letter and letter in 'ABCDE':
                return letter, m.end()
    return None, None


def extract_choices_inline(clean_prompt: str):
    """Extract choices when they appear inline."""
    all_matches = []
    for pattern in OPTION_PATTERNS:
        for m in pattern.finditer(clean_prompt):
            letter = None
            for group in m.groups():
                if group:
                    letter = group.upper()
                    break
            if letter and letter in 'ABCDE':
                all_matches.append((m.start(), m.end(), letter))

    if len(all_matches) < 2:
        return None

    all_matches.sort()
    option_texts = {}
    letters = []

    for i, (start, end, letter) in enumerate(all_matches):
        if letter in letters:
            continue
        letters.append(letter)

        content_start = end
        content_end = all_matches[i + 1][0] if i + 1 < len(all_matches) else len(clean_prompt)
        content = clean_prompt[content_start:content_end].strip()

        if not content:
            content = "(blank)"
        option_texts[letter] = content

    if len(option_texts) < 2:
        return None

    stem = clean_prompt[:all_matches[0][0]].strip()
    if not stem:
        stem = "Select the correct option."

    return stem, option_texts, letters


def extract_choices(clean_prompt: str):
    """Extract multiple choice options with ultra-enhanced pattern matching."""
    lines = clean_prompt.split('\n')
    choices = []

    for idx, line in enumerate(lines):
        letter, pos = try_match_option(line)
        if letter:
            choices.append((letter, idx, pos))

    if len(choices) >= 2:
        seen = set()
        filtered = []
        for letter, idx, pos in choices:
            if letter not in seen:
                seen.add(letter)
                filtered.append((letter, idx, pos))

        if len(filtered) >= 2:
            letters = [letter for letter, _, _ in filtered]
            if is_valid_sequence(letters):
                option_texts = {}
                for i, (letter, idx, pos) in enumerate(filtered):
                    start_line = idx
                    end_line = filtered[i + 1][1] if i + 1 < len(filtered) else len(lines)
                    desc_lines = []

                    remainder = lines[idx][pos:].strip()
                    if remainder:
                        desc_lines.append(remainder)

                    for j in range(idx + 1, end_line):
                        desc_lines.append(lines[j])

                    desc = ' '.join(desc_lines).strip()
                    if not desc:
                        desc = '(blank)'
                    option_texts[letter] = desc

                stem_lines = lines[:filtered[0][1]]
                stem = ' '.join(stem_lines).strip()
                if not stem:
                    stem = "Select the correct option."

                return stem, option_texts, letters

    return extract_choices_inline(clean_prompt)


def is_valid_sequence(letters):
    """Check if letters form a valid subsequence of ABCDE."""
    if not letters:
        return False

    unique_letters = []
    for letter in letters:
        if letter not in unique_letters:
            unique_letters.append(letter)

    if len(unique_letters) < 2:
        return False

    expected = 'ABCDE'
    expected_idx = 0

    for letter in unique_letters:
        try:
            idx = expected.index(letter, expected_idx)
            expected_idx = idx + 1
        except ValueError:
            return False

    return True


def parse_label(label_raw: str):
    """Parse label with enhanced pattern matching."""
    if not label_raw:
        return None

    for pattern in LABEL_PATTERNS:
        matches = pattern.findall(label_raw.upper())
        if matches:
            if matches and isinstance(matches[0], tuple):
                letters = [m for group in matches for m in (group if isinstance(group, tuple) else [group]) if m]
            else:
                letters = matches

            for letter in letters:
                if letter and letter in 'ABCDE':
                    return letter

    simple_match = re.search(r'\b([A-E])\b', label_raw.upper())
    if simple_match:
        return simple_match.group(1)

    return None


def fuzzy_match_label(label_raw: str, available_options, option_texts):
    """Try to match label even if format is unusual."""
    if not label_raw:
        return None

    cleaned = label_raw.strip()
    letter = parse_label(cleaned)
    if letter and letter in available_options:
        return letter

    cleaned_lower = cleaned.lower()
    for opt_letter, opt_text in option_texts.items():
        opt_text_clean = opt_text.strip().lower()

        if cleaned_lower == opt_text_clean:
            return opt_letter

        if cleaned_lower in opt_text_clean or opt_text_clean in cleaned_lower:
            if len(cleaned) > 0 and len(opt_text_clean) > 0:
                similarity = min(len(cleaned), len(opt_text_clean)) / max(len(cleaned), len(opt_text_clean))
                if similarity > 0.5:
                    return opt_letter

    for opt_letter in available_options:
        if cleaned.startswith(opt_letter) or cleaned.startswith(f"({opt_letter})"):
            return opt_letter

    return None


# ==================== 主处理逻辑 ====================

processed = []
debug_samples = []

# 使用索引作为查找键
indexed_data: Dict[int, dict] = {}

# 第一遍：读取所有数据并建立索引
print("步骤 1: 读取源文件并建立索引...")
with SRC.open('r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
            original_index = record.get('original_index')

            if original_index is None:
                skip_reason['missing_index'] += 1
                continue

            indexed_data[int(original_index)] = record

        except (json.JSONDecodeError, ValueError) as e:
            skip_reason['json_error'] += 1

print(f"已加载 {len(indexed_data)} 条索引记录\n")

# 第二遍：处理每条记录
print("步骤 2: 提取选择题...")
for original_index, record in indexed_data.items():
    prompt_turns = record.get('prompt', [])
    contents = [turn.get('content', '') for turn in prompt_turns if turn.get('content')]

    if not contents:
        skip_reason['empty_prompt'] += 1
        continue

    # 提取和处理
    raw_prompt = '\n'.join(contents)
    normalized = normalize_prompt(raw_prompt)
    result = extract_choices(normalized)

    if not result:
        skip_reason['no_choice_detected'] += 1
        if len(debug_samples) < 10:
            debug_samples.append({
                'original_index': original_index,
                'raw': raw_prompt[:300],
                'normalized': normalized[:300]
            })
        continue

    stem, option_texts, letter_order = result

    # 解析标签
    label_raw = str(record.get('label', ''))
    answer_letter = fuzzy_match_label(label_raw, option_texts.keys(), option_texts)

    if not answer_letter:
        skip_reason['label_no_letter'] += 1
        if len([s for s in debug_samples if 'label_raw' in s]) < 10:
            debug_samples.append({
                'original_index': original_index,
                'label_raw': label_raw,
                'options': list(option_texts.keys()),
                'option_texts': {k: v[:50] for k, v in option_texts.items()}
            })
        continue

    if answer_letter not in option_texts:
        skip_reason['letter_not_in_options'] += 1
        continue

    # 构建最终问题
    option_lines = [f"{letter}) {option_texts[letter]}" for letter in letter_order]
    question_text = (
        f"{stem}\n\n"
        f"Options:\n" + '\n'.join(option_lines) + '\n\n'
        f"Answer with the option letter only, in the form \\boxed{{{answer_letter}}}."
    )

    # **保留 original_index**
    processed.append({
        'original_index': original_index,  # 保留索引
        'prompt': [{'role': 'user', 'content': question_text}],
        'label': answer_letter
    })

# 写入输出
with DST.open('w', encoding='utf-8') as fout:
    for item in processed:
        json.dump(item, fout, ensure_ascii=False)
        fout.write('\n')

# 打印统计
total_lines = len(indexed_data) + sum(skip_reason.values()) - skip_reason.get('missing_index', 0)
print(f'\n{"="*60}')
print(f'total records: {len(indexed_data)}')
print(f'kept lines: {len(processed)}')
print(f'success rate: {len(processed) / len(indexed_data) * 100:.1f}%')
print(f'\nskip reasons:')
for reason, count in skip_reason.most_common():
    print(f'  {reason:>20}: {count}')

# 打印调试样本
if debug_samples:
    print(f'\n{"="*60}')
    print('Debug samples (first 5 failed cases):')
    print('='*60)
    for i, sample in enumerate(debug_samples[:5], 1):
        print(f'\nSample {i}:')
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 150:
                print(f'  {key}: {value[:150]}...')
            else:
                print(f'  {key}: {value}')

print(f'\n输出文件: {DST}')
