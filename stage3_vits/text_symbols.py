"""
VITS 文本/音素符号表
=====================
使用 gruut 将英文文本转换为 IPA 音素序列。
"""

import re
import torch
from pathlib import Path

# ============================================================
# IPA 音素集合 (英文, gruut)
# ============================================================
# gruut en-US 使用的 IPA 音素（含重音标记）
# 从 LJSpeech 转录文本中收集
_ipa_consonants = [
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z',
    'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 'ʔ', 'd͡', 't͡',
]

_ipa_vowels = [
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɒ', 'ɔ', 'o', 'ʊ', 'u',
    'ə', 'ɚ', 'ɜ', 'ʌ', 'а',
]

# 双元音（双字符）
_ipa_diphthongs = [
    'aɪ', 'aʊ', 'ɔɪ', 'oʊ', 'eɪ', 'iː', 'uː', 'ɑː', 'ɔː', 'ɜː',
]

# 重音符号（附着在元音前）
_stress_marks = ['ˈ', 'ˌ']

# 标点
_punctuation = '!?,-.:;—\'"()[]{} '

# 特殊 token
_special = ['<pad>', '<eos>', '<unk>', '<bos>']


def _build_symbols() -> list[str]:
    """构建完整的音素符号列表。"""
    symbols = list(_special)
    for c in _ipa_consonants:
        symbols.append(c)
    for d in _ipa_diphthongs:
        symbols.append(d)
    # stress + vowel combos
    for s in _stress_marks:
        for v in _ipa_vowels:
            symbols.append(s + v)
    for p in _punctuation:
        symbols.append(p)
    return symbols


# 全局符号表
_symbols = _build_symbols()
_symbol_to_id = {s: i for i, s in enumerate(_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(_symbols)}

# 特殊 token ID
PAD_ID = _symbol_to_id['<pad>']
EOS_ID = _symbol_to_id['<eos>']
UNK_ID = _symbol_to_id['<unk>']
BOS_ID = _symbol_to_id['<bos>']


def text_to_sequence(text: str) -> list[int]:
    """将英文文本转换为音素 ID 序列。"""
    import gruut

    phones = []
    sentences = list(gruut.sentences(text, lang='en-US'))
    for s in sentences:
        for word in s:
            if word.phonemes:
                for ph in word.phonemes:
                    if ph:
                        phones.append(ph)
            phones.append(' ')  # 词间空格
    if phones and phones[-1] == ' ':
        phones.pop()

    ids = []
    ids.append(BOS_ID)
    for ph in phones:
        ids.append(_symbol_to_id.get(ph, UNK_ID))
    ids.append(EOS_ID)
    return ids


def sequence_to_text(ids: list[int]) -> str:
    """将音素 ID 序列转换回文本（调试用）。"""
    return '|'.join(_id_to_symbol.get(i, '<?>') for i in ids if i not in (
        PAD_ID, EOS_ID, BOS_ID))


def collect_symbols_from_ljspeech(metadata_path: str) -> set[str]:
    """从 LJSpeech metadata 收集所有出现的音素。"""
    import gruut

    all_phones = set()
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                text = parts[2]
                sentences = list(gruut.sentences(text, lang='en-US'))
                for s in sentences:
                    for word in s:
                        if word.phonemes:
                            for ph in word.phonemes:
                                if ph:
                                    all_phones.add(ph)

    print(f"收集到 {len(all_phones)} 个唯一音素")
    return all_phones


def get_symbols() -> list[str]:
    return list(_symbols)


def get_symbol_size() -> int:
    return len(_symbols)


if __name__ == '__main__':
    # 测试音素转换
    test_text = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition."
    seq = text_to_sequence(test_text)
    print(f"文本: {test_text}")
    print(f"音素序列长度: {len(seq)}")
    print(f"ID序列: {seq[:20]}...")
    print(f"词汇表大小: {len(_symbols)}")

    # 从 LJSpeech metadata 收集音素
    meta_path = Path(__file__).parent.parent / "data" / "LJSpeech-1.1" / "metadata.csv"
    if meta_path.exists():
        extra = collect_symbols_from_ljspeech(str(meta_path))
        missing = extra - set(_symbols)
        if missing:
            print(f"未覆盖的音素: {missing}")
        else:
            print("所有音素均在词汇表中 ✓")