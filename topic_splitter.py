"""把一条用户消息拆成若干个「彼此独立的话题」。

这里是纯文本启发式逻辑，不依赖 AstrBot，方便单独做单元测试。

设计原则：
1. 宁可少拆，也不要把同一句话硬拆成多个话题（比如「要求：1. 2. 3.」这种列表）；
2. 优先识别明确的问句（？/吗/呢）和明确的转折词（另外、还有、对了……）；
3. 只在「两个以上话题」时才返回结果，否则返回空列表。
"""

from __future__ import annotations

import re
from typing import List

__all__ = ["split_topics", "is_greeting"]


# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

# 以问号结尾
_QUESTION_END = re.compile(r"[?？]\s*$")
# 以「吗 / 呢」这类疑问语气词结尾
_SOFT_QUESTION_END = re.compile(r"(?:吗|呢)\s*[。.!！~～]*\s*$")
# 疑问词
_QUESTION_WORDS = re.compile(
    r"(什么|怎么|怎样|为什么|为啥|如何|哪里|哪个|哪些|哪儿|谁|多少|几点|几号|"
    r"能不能|可不可以|会不会|有没有|是不是|要不要|行不行|好不好|对不对|"
    r"请问|请教|求助|问一下|问下)"
)
# 话题转折词：必须出现在句子/分句开头，避免「另外一个人」这类误判
_TRANSITION = re.compile(
    r"(?:^|(?<=[。！？!?；;，,、\s]))"
    r"(?:另外(?!一)|还有(?!一|两|几)|对了|顺便(?:问|说|提)?|此外|再问|再帮我|"
    r"还有个|以及|同时问)"
)
# 行首的列表标记：1. / 2、/ ① / - / •
_LEAD_MARKER = re.compile(r"^\s*(?:\d{1,2}\s*[.、)）]|[①-⑩]|[-*•·])\s*")
# 「第一，」「第二、」这类序数（必须带标点，避免误伤「第二天」「第三方」）
_ORDINAL = re.compile(r"^\s*第[一二三四五六七八九十]+\s*[，,、.．:：]\s*")
# 「要求：」「如下：」「例如：」这类列表引导语
_LEAD_IN = re.compile(r"(?:如下|以下|要求|步骤|例如|比如|包括|分别是)[：:]?\s*$")

# 寒暄/招呼，不作为独立话题
_GREETING = re.compile(
    r"^(?:你好|您好|哈喽|哈啰|嗨|hi|hello|在吗|在么|在不在|"
    r"早上好|中午好|下午好|晚上好|晚安)"
    r"[\s呀啊哦呢嘛~～!！。.?？,，]*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """判断一段文字是不是纯寒暄（你好/在吗/晚安……）。"""

    return bool(_GREETING.match((text or "").strip()))


def _is_question(segment: str) -> bool:
    """粗略判断一个分句是不是「在提问 / 在提要求」。"""

    seg = (segment or "").strip()
    if not seg:
        return False
    if _QUESTION_END.search(seg) or _SOFT_QUESTION_END.search(seg):
        return True
    return bool(_QUESTION_WORDS.search(seg)) and len(seg) >= 6


def _clean_segment(segment: str) -> str:
    """去掉列表序号、转折词等，得到干净的话题描述。"""

    seg = (segment or "").strip()
    seg = _LEAD_MARKER.sub("", seg)
    seg = _ORDINAL.sub("", seg)
    seg = _TRANSITION.sub("", seg, count=1)
    seg = re.sub(r"[ \t]+", " ", seg)
    return seg.strip()


def _finish(
    parts: List[str], max_topics: int, min_topic_chars: int
) -> List[str]:
    """统一收尾：清洗、去寒暄、过滤碎片、限制数量。"""

    cleaned: List[str] = []
    for part in parts:
        seg = _clean_segment(part)
        if not seg or is_greeting(seg):
            continue
        cleaned.append(seg)

    if len(cleaned) < 2:
        return []

    kept = [seg for seg in cleaned if len(seg) >= min_topic_chars]
    if len(kept) < 2:
        return []
    # 话题太多说明这多半是一段完整的叙述/清单，不做拆分，避免回答被切碎
    if len(kept) > max_topics:
        return []
    return kept


def _is_lead_in_list(lines: List[str]) -> bool:
    """判断多行文本是不是「引导语 + 编号清单」，这类不应拆成多个话题。"""

    if not lines:
        return False
    head = lines[0]
    if _LEAD_IN.search(head):
        return True

    numbered = [ln for ln in lines if _LEAD_MARKER.match(ln)]
    if len(numbered) >= 2 and len(numbered) == len(lines):
        # 全是编号条目，而且都不是问句 → 视为同一件事的步骤/要点
        if not any(_is_question(_LEAD_MARKER.sub("", ln)) for ln in lines):
            return True
    return False


def _accept_newline_chunks(
    lines: List[str],
    chunks: List[str],
    max_topics: int,
    min_topic_chars: int,
) -> bool:
    """判断「按行拆开」得到的若干段是否真的算多个话题。"""

    if len(chunks) < 2:
        return False
    # 有行以冒号结尾 → 下一行是在解释/列举这一行
    if any(ln.endswith(("：", ":")) for ln in lines):
        return False
    if _is_lead_in_list(lines):
        return False

    has_transition = any(_TRANSITION.match(ln) for ln in lines[1:])
    question_count = sum(1 for chunk in chunks if _is_question(chunk))
    if has_transition or question_count >= 2:
        return True

    # 兜底：每行都像一句完整的话（够长），通常就是两件事
    floor = max(min_topic_chars, 6)
    return all(len(chunk) >= floor for chunk in chunks)


def _split_by_transition(text: str) -> List[str]:
    """按「另外 / 还有 / 对了……」等转折词切分。"""

    matches = list(_TRANSITION.finditer(text))
    if not matches:
        return []

    parts: List[str] = []
    start = 0
    for match in matches:
        segment = text[start : match.start()].strip()
        if segment:
            parts.append(segment)
        start = match.start()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts if len(parts) >= 2 else []


def split_topics(
    text: str,
    max_topics: int = 6,
    min_topic_chars: int = 4,
) -> List[str]:
    """把一条消息拆成多个话题；拆不出来时返回空列表。

    Args:
        text: 用户消息（可以是合并后的多条消息，用换行连接）。
        max_topics: 最多允许的话题数，超过则认为不适合拆分。
        min_topic_chars: 话题的最短长度，过短的碎片会被丢弃。

    Returns:
        话题列表（长度 >= 2 才返回），否则返回 []。
    """

    raw = (text or "").strip()
    if not raw:
        return []
    # 代码块里可能全是换行和问号，直接跳过，避免把代码拆成话题
    if "```" in raw:
        return []

    # 1) 按行切
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) >= 2:
        chunks = [_clean_segment(line) for line in lines]
        chunks = [chunk for chunk in chunks if chunk]
        if _accept_newline_chunks(lines, chunks, max_topics, min_topic_chars):
            result = _finish(chunks, max_topics, min_topic_chars)
            if result:
                return result

    # 2) 按句子切：至少要有两个问句才认为是在问两件事
    sentences = [
        seg.strip()
        for seg in re.split(r"(?<=[。！？!?；;])\s*", raw)
        if seg.strip()
    ]
    if len(sentences) >= 2:
        questions = [seg for seg in sentences if _is_question(seg)]
        if len(questions) >= 2:
            result = _finish(questions, max_topics, min_topic_chars)
            if result:
                return result

    # 3) 按转折词切
    parts = _split_by_transition(raw)
    if len(parts) >= 2:
        result = _finish(parts, max_topics, min_topic_chars)
        if result:
            return result

    return []
