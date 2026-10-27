"""AstrBot 插件：多话题分答（Multi-Topic Reply）。

要解决的问题
------------
在 QQ 里连着发两条消息（例如「你下午吃了什么？」「你晚上想干点啥？」），
AstrBot 会把后一条当成前一条生成过程中的 follow-up，导致最终只回答了后一条。

本插件在消息进入大模型之前做两件事：

1. **短暂合并**：把短时间内连续发来的消息放进同一个「待回复」队列，一起处理；
2. **分话题作答**：让模型按话题逐条回复，并把回复拆成多条 QQ 消息；
   每个话题的第一条回复会**引用**对应的原消息，像真人聊天一样。

它不写 ``1. 2. 3.`` 这类编号，也不改消息链里的其他内容；只在需要分话题回复时
接管发送，其余情况完全走 AstrBot 原生流程。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, List

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:  # MessageChain 在不同版本里可能从不同位置导出
    from astrbot.api.event import MessageChain  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from astrbot.core.message.message_event_result import MessageChain  # type: ignore

try:
    import astrbot.api.message_components as Comp
except Exception:  # pragma: no cover
    from astrbot.core.message import components as Comp  # type: ignore

try:
    from astrbot.core.message.message_event_result import ResultContentType
except Exception:  # pragma: no cover
    ResultContentType = None  # type: ignore

try:  # 仅用于类型标注
    from astrbot.api.provider import ProviderRequest  # noqa: F401
except Exception:  # pragma: no cover
    ProviderRequest = Any  # type: ignore

try:  # 优先使用临时内容块，避免把分答提示写进历史
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover
    TextPart = None  # type: ignore

try:
    from .topic_splitter import is_greeting, split_topics
except Exception:  # pragma: no cover - 兼容按单文件加载
    import importlib.util as _ilu
    import os as _os

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _spec = _ilu.spec_from_file_location(
        "_astrbot_multi_topic_splitter", _os.path.join(_here, "topic_splitter.py")
    )
    _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    is_greeting = _mod.is_greeting
    split_topics = _mod.split_topics


# 只允许纯文本/表情段参与合并，图片、语音、文件等一律走原流程
_TEXT_SAFE_COMPONENTS = {"Plain", "At", "Reply", "Quote", "Face"}

# 模型输出用的分隔标记。
# 模型经常把 [[T1]] 写成各种「看起来一样」的括号（全角、数学括号、书名号、
# 无括号单独一行等），所以统一用宽松匹配，宁可多认，也不能漏到 QQ 里。
_BRACKET_OPEN = r"[\[［【（(｛{⟦〚❲⁅＜〈《「『〖〘]"
_BRACKET_CLOSE = r"[\]］】）)｝}⟧〛❳⁆＞〉》」』〗〙]"
_TOPIC_RE = re.compile(
    rf"{_BRACKET_OPEN}{{1,3}}\s*[*_`]*\s*(?:T|TOPIC|话题|問題|问题)\s*"
    rf"[-_:#：\s]*\s*(\d+)\s*[*_`]*\s*{_BRACKET_CLOSE}{{1,3}}",
    re.IGNORECASE,
)
_SEG_RE = re.compile(
    rf"{_BRACKET_OPEN}{{1,3}}\s*[*_`]*\s*(?:S|SEG|SEGMENT|分段|分条|分段回复)\s*"
    rf"[*_`]*\s*{_BRACKET_CLOSE}{{1,3}}",
    re.IGNORECASE,
)
_ANY_MARKER_RE = re.compile(
    rf"{_BRACKET_OPEN}{{1,3}}\s*[*_`]*\s*"
    rf"(?:(?:T|TOPIC|话题|問題|问题)\s*[-_:#：\s]*\s*\d+|"
    rf"(?:S|SEG|SEGMENT|分段|分条|分段回复))\s*[*_`]*\s*"
    rf"{_BRACKET_CLOSE}{{1,3}}",
    re.IGNORECASE,
)

# 完全没写括号、单独一行只放标记的情况
_LINE_TOPIC_RE = re.compile(
    r"^\s*[*_`]*(?:T|TOPIC|话题|問題|问题)\s*[-_:#：\s]*\s*(\d+)\s*[*_`]*\s*$",
    re.IGNORECASE,
)
_LINE_SEG_RE = re.compile(
    r"^\s*[*_`]*(?:S|SEG|SEGMENT|分段|分条|分段回复)\s*[*_`]*\s*$",
    re.IGNORECASE,
)

# 最后的兜底：任何括号包着 T1 / 话题1 / S 的碎片都清掉
_LOOSE_MARKER_RE = re.compile(
    r"(?<![\w])[^\w\s]{0,4}\s*(?:(?:T|TOPIC|话题)\s*[-_:#：]?\s*\d+|"
    r"(?:S|SEG|分段|分条))\s*[^\w\s]{0,4}(?![\w])",
    re.IGNORECASE,
)

# 零宽字符：有些模型会在标记里插入，肉眼看不见但会让匹配失败
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _normalize(text: str) -> str:
    """去掉零宽字符，便于识别标记（不改动正常标点）。"""

    return _ZW_RE.sub("", text or "")


@dataclass
class _Topic:
    """一条等待/需要回复的消息。"""

    text: str
    message_id: str = ""
    nickname: str = ""
    ts: float = 0.0


@dataclass
class _SessionState:
    """用于给同一会话的消息编号，判断谁是最新一条。"""

    seq: int = 0
    topics: List[_Topic] = field(default_factory=list)


@dataclass
class _ReplyBlock:
    """模型输出里的一个话题回复块。"""

    index: int
    segments: List[str]


def _all_message_listener(priority: int = 1000):
    """注册「所有消息」监听器，尽量带高优先级。"""

    def decorator(func):
        try:
            return filter.event_message_type(
                filter.EventMessageType.ALL, priority=priority
            )(func)
        except TypeError:  # 旧版本可能不接受 priority
            return filter.event_message_type(filter.EventMessageType.ALL)(func)

    return decorator


def _safe_on_agent_done():
    """on_agent_done 是较新的钩子，旧版本没有时跳过注册而不是加载失败。"""

    factory = getattr(filter, "on_agent_done", None)

    def decorator(func):
        if factory is None:
            return func
        try:
            return factory()(func)
        except Exception:
            return func

    return decorator


def _collect_segments(body: str) -> List[str]:
    """把一段回复按分段标记切成多条，并清理残留标记。"""

    segments: List[str] = []
    for seg in _SEG_RE.split(body):
        seg = seg.strip()
        # 模型偶尔会把整段包进代码块，去掉围栏
        seg = re.sub(r"^```[a-zA-Z]*\s*", "", seg).strip()
        seg = re.sub(r"\s*```$", "", seg).strip()
        # 再清一次残留标记，确保不会漏到 QQ 里
        seg = _strip_markers(seg)
        if seg:
            segments.append(seg)
    return segments


def _parse_blocks_line_mode(text: str) -> List[_ReplyBlock]:
    """兜底：标记单独占一行（没有括号）时，按行解析。"""

    blocks: List[_ReplyBlock] = []
    current: _ReplyBlock | None = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        if current is not None:
            segments = _collect_segments("\n".join(buffer))
            if segments:
                current.segments.extend(segments)
        buffer = []

    for line in text.splitlines():
        topic_match = _LINE_TOPIC_RE.match(line)
        if topic_match:
            flush()
            if current is not None:
                blocks.append(current)
            current = _ReplyBlock(index=int(topic_match.group(1)), segments=[])
            continue
        if _LINE_SEG_RE.match(line):
            if current is not None:
                segments = _collect_segments("\n".join(buffer))
                if segments:
                    current.segments.extend(segments)
                buffer = []
            continue
        if current is not None:
            buffer.append(line)

    flush()
    if current is not None:
        blocks.append(current)
    return [block for block in blocks if block.segments]


def _parse_blocks(text: str) -> List[_ReplyBlock]:
    """把模型输出解析成「话题 -> 若干条消息」。"""

    text = _normalize(text)
    matches = list(_TOPIC_RE.finditer(text))
    if not matches:
        return _parse_blocks_line_mode(text)

    blocks: List[_ReplyBlock] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        segments = _collect_segments(body)
        if not segments:
            continue
        blocks.append(_ReplyBlock(index=int(match.group(1)), segments=segments))
    return blocks


def _strip_markers(text: str) -> str:
    """去掉模型可能写进正文的分隔标记（包括各种括号变体）。"""

    original = text or ""
    text = _normalize(text)
    cleaned = _ANY_MARKER_RE.sub("", text)
    # 再兜一层：任何括号包着 T1 / 话题1 / S 之类的碎片
    cleaned = _LOOSE_MARKER_RE.sub("", cleaned)

    # 兜底：整行只剩下标记/序号外壳的，直接丢掉
    kept_lines = []
    for line in cleaned.splitlines():
        if _LINE_TOPIC_RE.match(line) or _LINE_SEG_RE.match(line):
            continue
        if re.fullmatch(
            r"\s*[\[\]［］【】（）()｛｝{}⟦⟧〚〛❲❳⁅⁆＜＞〈〉《》「」『』〖〗〘〙]{1,4}"
            r"\s*[*_`]*(?:T|S|TOPIC|SEG|话题|分段)?\s*\d?\s*[*_`]*\s*"
            r"[\[\]［］【】（）()｛｝{}⟦⟧〚〛❲❳⁅⁆＜＞〈〉《》「」『』〖〗〘〙]{0,4}\s*",
            line,
            re.IGNORECASE,
        ):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = cleaned.strip()
    # 如果经过上述清理后什么都没了，至少不要把原文原样放出去
    if not cleaned and original.strip() and _ANY_MARKER_RE.search(_normalize(original)):
        return ""
    return cleaned


def _split_long_text(text: str, limit: int) -> List[str]:
    """把过长的单条回复按标点拆成几条，保持聊天感。"""

    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return [text] if text else []

    parts = re.split(r"(?<=[。！？!?~～…；;])", text)
    result: List[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if buffer and len(buffer) + len(part) > limit:
            result.append(buffer.strip())
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        result.append(buffer.strip())
    return result or [text]


class MultiTopicPlugin(Star):
    """多话题分答插件。"""

    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self._states: dict[str, _SessionState] = {}

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _enabled(self) -> bool:
        return bool(self._cfg("enabled", True))

    def _log(self, message: str) -> None:
        """调试开关打开时用 INFO 输出，保证在 WebUI 日志页默认可看到。"""

        if bool(self._cfg("debug", False)):
            logger.info(f"[multi_topic] {message}")
        else:
            logger.debug(f"[multi_topic] {message}")

    def _int_cfg(self, key: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(self._cfg(key, default))))
        except (TypeError, ValueError):
            return default

    def _float_cfg(self, key: str, default: float) -> float:
        try:
            return max(0.0, float(self._cfg(key, default)))
        except (TypeError, ValueError):
            return default

    def _streaming_enabled(self, event: AstrMessageEvent) -> bool:
        """判断当前配置是否开启了流式输出（QQ 上会退化成逐句分段发送）。

        流式发送是在模型生成过程中就把文字发出去，插件拿不到完整文本，
        也就无法按话题拆分、引用，标记还会被原样发到 QQ。因此检测到流式时
        宁可跳过分条，也不要把标记发出去。
        """

        if not bool(self._cfg("streaming_guard", True)):
            return False
        try:
            umo = event.unified_msg_origin
        except Exception:
            umo = None
        try:
            cfg = self.context.get_config(umo=umo)
        except Exception:
            try:
                cfg = self.context.get_config()
            except Exception:
                return False

        settings = {}
        try:
            settings = cfg.get("provider_settings", {}) or {}
        except Exception:
            return False
        streaming = bool(settings.get("streaming_response", False))
        segmented = bool(settings.get("streaming_segmented", False))
        return streaming and segmented

    @staticmethod
    def _active_agent_runners():
        """拿到 AstrBot 当前的「正在生成的 Agent」表（拿不到就返回 None）。"""

        try:
            from astrbot.core.pipeline.process_stage import follow_up

            return getattr(follow_up, "_ACTIVE_AGENT_RUNNERS", None)
        except Exception:
            return None

    async def _wait_for_idle(self, umo: str) -> None:
        """如果这个会话正有回复在生成，先等它结束。

        AstrBot 的 follow-up 机制会把生成过程中收到的新消息并进当前这次生成，
        模型往往只回应最新一句，前面的消息就丢了。等当前生成结束后再处理，
        这些消息就会各自排队、按正常流程回答，而不是被吞掉。
        """

        if not bool(self._cfg("wait_active_agent", True)):
            return
        runners = self._active_agent_runners()
        if runners is None:
            return
        deadline = time.monotonic() + float(self._cfg("max_wait_seconds", 60))
        waited = False
        while umo in runners and time.monotonic() < deadline:
            if not waited:
                self._log("检测到上一次回复还在生成，先排队等待，避免被 follow-up 吞掉")
                waited = True
            await asyncio.sleep(0.3)

    # ------------------------------------------------------------------
    # 第一步：收集 / 合并连续消息
    # ------------------------------------------------------------------
    @_all_message_listener(priority=1000)
    async def _collect_message(self, event: AstrMessageEvent):
        """把短时间内的连续消息合并成一次「分话题回复」。

        较早的消息会在等待后发现后面还有消息，于是 ``stop_event()`` 放弃自己的
        回复；最后一条消息带上全部待回复内容，继续走 AstrBot 原生的 LLM 流程。
        """

        if not self._enabled():
            return
        try:
            if (event.get_platform_name() or "") != "aiocqhttp":
                return
        except Exception:
            return

        text = (event.message_str or "").strip()
        if not text or len(text) < 2:
            return
        if self._looks_like_command(text):
            return
        if self._has_media(event):
            return

        try:
            umo = event.unified_msg_origin
        except Exception:
            return
        if not umo:
            return

        now = time.monotonic()
        state = self._states.get(umo)
        if state is None:
            state = _SessionState()
            self._states[umo] = state

        state.topics.append(
            _Topic(
                text=text,
                message_id=str(getattr(event.message_obj, "message_id", "") or ""),
                nickname=self._safe_sender_name(event),
                ts=now,
            )
        )
        state.seq += 1
        my_seq = state.seq

        # 先确保上一次生成已经结束，否则这条消息会被 follow-up 并进旧的生成里
        await self._wait_for_idle(umo)

        # 防抖：每条新消息都会让「最新一条」重新计时，
        # 直到安静满 merge_window_seconds 才把整批消息交给模型。
        delay = self._float_cfg("merge_window_seconds", 6.0)
        max_chars = self._int_cfg("debounce_max_chars", 200, 10, 4000)
        if delay > 0 and len(text) <= max_chars:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            # 等待期间可能刚好有新的生成开始，再确认一次
            await self._wait_for_idle(umo)
            if state.seq != my_seq:
                # 后面还有消息，由更新的那条重新计时
                self._log(f"收到新消息，重新计时: {text[:40]}")
                event.stop_event()
                return

        ttl = self._float_cfg("pending_ttl", 120.0)
        pending = [t for t in state.topics if now - t.ts <= ttl]
        state.topics = []
        if not pending:
            return

        max_topics = self._int_cfg("max_topics", 6, 2, 12)
        messages = pending[-max_topics:]
        if not messages:
            return

        # 只有一条消息、而且这句话本身只聊一件事 → 不接管，走原生流程
        multi_in_one = (
            len(messages) == 1
            and len(split_topics(messages[0].text, max_topics, 2)) >= 2
        )
        if len(messages) < 2 and not multi_in_one:
            return

        merged = "\n".join(t.text for t in messages)
        try:
            event.message_str = merged  # type: ignore[misc]
            event.message_obj.message_str = merged
        except Exception:
            pass

        # 要拿到完整文本再分组、拆分发送，因此关掉流式
        try:
            event.set_extra("enable_streaming", False)
        except Exception:
            pass

        # 这里存的是「原始消息」，由模型自己判断哪些消息属于同一个话题
        try:
            event._mt_topics = messages  # type: ignore[attr-defined]
        except Exception:
            return

        self._log(
            f"防抖 {delay:g}s 结束，待回复 {len(messages)} 条消息: "
            f"{[t.text for t in messages]}"
        )

    def _build_topics(self, messages: List[_Topic]) -> List[_Topic]:
        """把消息队列展开成话题：一条消息里多个问题也会拆开。"""

        max_topics = self._int_cfg("max_topics", 6, 2, 12)
        min_chars = self._int_cfg("min_topic_chars", 2, 1, 50)
        cont_max = self._int_cfg("continuation_max_chars", 6, 0, 40)

        topics: List[_Topic] = []
        for msg in messages:
            text = (msg.text or "").strip()
            if not text:
                continue

            # 「嗯」「然后呢」这类短句视为接着上一个话题，而不是新话题
            if topics and self._is_continuation(text, cont_max):
                topics[-1].text = f"{topics[-1].text}\n{text}"
                continue

            subs = split_topics(text, max_topics, min_chars)
            if len(subs) >= 2:
                for sub in subs:
                    topics.append(
                        _Topic(sub, msg.message_id, msg.nickname, msg.ts)
                    )
            else:
                topics.append(_Topic(text, msg.message_id, msg.nickname, msg.ts))

        if len(topics) > max_topics:
            topics = topics[:max_topics]
        return topics

    def _is_continuation(self, text: str, max_chars: int) -> bool:
        if max_chars <= 0 or len(text) > max_chars:
            return False
        if "?" in text or "？" in text:
            return False
        if is_greeting(text):
            return False
        return len(split_topics(text, 6, 2)) < 2

    def _safe_sender_name(self, event: AstrMessageEvent) -> str:
        try:
            return event.get_sender_name() or ""
        except Exception:
            return ""

    def _has_media(self, event: AstrMessageEvent) -> bool:
        try:
            chain = event.message_obj.message
        except Exception:
            return False
        for component in chain or []:
            if type(component).__name__ not in _TEXT_SAFE_COMPONENTS:
                return True
        return False

    def _looks_like_command(self, text: str) -> bool:
        prefixes = self._cfg("command_prefixes", ["/", "／", "!", "！", "#"])
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not isinstance(prefixes, (list, tuple)):
            prefixes = ["/", "／"]
        return any(isinstance(p, str) and p and text.startswith(p) for p in prefixes)

    # ------------------------------------------------------------------
    # 第二步：请求大模型时注入「分话题、带引用」的格式要求
    # ------------------------------------------------------------------
    @filter.on_llm_request()
    async def _inject_topic_hint(
        self, event: AstrMessageEvent, req: "ProviderRequest"
    ):
        if not self._enabled():
            return
        topics = getattr(event, "_mt_topics", None)
        if not topics:
            return
        if getattr(event, "_mt_hinted", False):
            return
        if self._streaming_enabled(event):
            # 流式发送时拿不到完整文本，注入分隔标记只会被原样发到 QQ
            logger.warning(
                "[multi_topic] 检测到流式输出已开启（streaming_response + "
                "streaming_segmented），本次跳过分条引用。请在 WebUI 配置文件里关闭"
                "「流式输出」和「流式分段」后重试。"
            )
            try:
                event._mt_topics = None  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        try:
            self._inject(req, self._build_hint(topics))
            event._mt_hinted = True  # type: ignore[attr-defined]
            self._log(f"已向请求注入 {len(topics)} 个话题的分答提示")
        except Exception as exc:  # 出错不能影响正常回复
            logger.warning(f"[multi_topic] 注入分答提示失败: {exc}")

    def _build_hint(self, topics: List[_Topic]) -> str:
        n = len(topics)
        lines = "\n".join(
            f"消息{i}：{t.text[:300]}" for i, t in enumerate(topics, 1)
        )
        if n == 1:
            example = (
                "[[T1]]\n"
                "（回应第一个话题的第一句）\n"
                "[[S]]\n"
                "（同一话题的下一句，可省略）\n"
            )
            group_rule = (
                "2. 这一条消息里如果有多个互相独立的话题，就拆成多个 [[T1]] 块，"
                "分别回应。\n"
            )
        else:
            example = (
                "[[T1]]\n"
                "（回应从消息 1 开始的那个话题）\n"
                "[[S]]\n"
                "（同一话题的下一句，可省略）\n"
                "[[T2]]\n"
                "（回应从消息 2 开始的那个话题）\n"
                "……\n"
            )
            group_rule = (
                "2. 几条消息如果是在说同一件事，就合并成一次回应；"
                "[[Tn]] 的 n 用这个话题**第一条消息的序号**，不要从 1 连续重排。\n"
            )
        return (
            "【分话题回复】\n"
            f"用户这一轮连续发了 {n} 条消息（可能有多个话题，也可能几条在说同一件事）：\n"
            f"{lines}\n\n"
            "请先自己判断这些消息分别属于哪些话题，再按话题逐个回应。"
            "严格使用下面的格式（分隔标记必须保留，不要解释它）：\n"
            f"{example}\n"
            "要求：\n"
            "1. 不要输出“1.”“2.”、不要用列表、不要写“问题一/回答一”，"
            "不要解释你在做什么——就像随口聊天。\n"
            f"{group_rule}"
            "3. 每个话题单独回应，按话题出现的顺序来；不要把几个话题揉在一段里。\n"
            "4. 每条只写一两句口语，一个话题 1~3 条即可，不要长篇大论。\n"
            "5. 不要复述用户的原话，不要加“好的，我先回答第一个”这种过渡句。\n"
            "6. 之前已经聊过、用户这次没有追问的话题，不要再翻出来；"
            "如果某条消息是在接着上一个话题说，就顺着接下去，不要重复回答。\n"
            "7. 分隔标记必须使用英文半角方括号和半角字母（[[T1]]、[[S]]），"
            "不要改写成全角括号、中文标点或别的写法。"
        )

    def _inject(self, req: Any, hint: str) -> None:
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is not None and TextPart is not None:
            part = None
            try:
                part = TextPart(text=hint)
            except TypeError:
                try:
                    part = TextPart(hint)  # type: ignore[call-arg]
                except Exception:
                    part = None
            if part is not None:
                marker = getattr(part, "mark_as_temp", None)
                if callable(marker):
                    try:
                        marked = marker()
                        if marked is not None:
                            part = marked
                    except Exception:
                        pass
                try:
                    parts.append(part)
                    return
                except Exception:
                    pass
        try:
            req.prompt = f"{getattr(req, 'prompt', '') or ''}\n\n{hint}"
        except Exception as exc:
            logger.warning("[multi_topic] 无法追加分答提示: %s", exc)

    @_safe_on_agent_done()
    async def _clean_history_markers(self, event, run_context, resp):
        """保存历史前去掉 [[T1]] 这类只用于本次分条的分隔标记。"""

        if not getattr(event, "_mt_topics", None):
            return
        try:
            messages = run_context.messages
        except Exception:
            return

        for message in messages:
            if getattr(message, "role", None) != "assistant":
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str):
                cleaned = _strip_markers(content)
                if cleaned != content:
                    message.content = cleaned
            elif isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        cleaned = _strip_markers(text)
                        if cleaned != text:
                            part.text = cleaned

    # ------------------------------------------------------------------
    # 第三步：把回复拆成多条消息，并引用对应原消息
    # ------------------------------------------------------------------
    # 优先级要高于其它「发送前」插件：必须先拿到完整原文，否则分段/改写类
    # 插件可能已经对消息链动手（甚至先发出去），我们就只能看到残片。
    @filter.on_decorating_result(priority=10000)
    async def _send_topic_replies(self, event: AstrMessageEvent):
        if not self._enabled():
            return
        topics = getattr(event, "_mt_topics", None)
        if not topics:
            return
        if getattr(event, "_mt_sent", False):
            return

        result = event.get_result()
        if result is None or not result.chain:
            return
        if ResultContentType is not None:
            content_type = getattr(result, "result_content_type", None)
            if content_type in (
                ResultContentType.STREAMING_RESULT,
                ResultContentType.STREAMING_FINISH,
            ):
                # 已经流式发出去的内容无法再拆分，保持原样
                return
        if not all(isinstance(comp, Comp.Plain) for comp in result.chain):
            # 有图片/语音等，不能拆成多条，最多把标记清理掉
            for comp in result.chain:
                if isinstance(comp, Comp.Plain):
                    comp.text = _strip_markers(comp.text)
            return

        text = "".join(comp.text for comp in result.chain)
        blocks = _parse_blocks(text)
        if not blocks:
            self._log(
                f"回复未按格式分块，回退为普通回复；原文开头(ascii): {ascii(text[:200])}"
            )
            cleaned = _strip_markers(text)
            if cleaned and cleaned != text:
                result.chain = [Comp.Plain(cleaned)]
            elif not cleaned and text.strip():
                # 整段都是标记、没有可发送的内容，宁可什么都不发
                try:
                    event.clear_result()
                except Exception:
                    pass
            return

        try:
            await self._send_blocks(event, blocks, topics)
            total = sum(len(block.segments) for block in blocks)
            self._log(f"已按 {len(blocks)} 组话题回复分条发送 {total} 条消息")
        except Exception as exc:
            logger.warning(f"[multi_topic] 分条发送失败，回退为普通回复: {exc}")
            cleaned = _strip_markers(text)
            if cleaned:
                result.chain = [Comp.Plain(cleaned)]
            return

        event._mt_sent = True  # type: ignore[attr-defined]
        event.clear_result()

    async def _send_blocks(
        self,
        event: AstrMessageEvent,
        blocks: List[_ReplyBlock],
        topics: List[_Topic],
    ) -> None:
        segment_delay = self._float_cfg("segment_delay", 0.5)
        topic_delay = self._float_cfg("topic_delay", 0.8)
        max_chars = self._int_cfg("max_segment_chars", 120, 20, 1000)
        quote_enabled = bool(self._cfg("send_quotes", True))

        last_quote = ""
        for block_index, block in enumerate(blocks):
            topic = topics[block.index - 1] if 0 < block.index <= len(topics) else topics[-1]
            first_message = True

            for segment_index, segment in enumerate(block.segments):
                for part_index, part in enumerate(
                    _split_long_text(segment, max_chars)
                ):
                    chain: List[Any] = []
                    if (
                        quote_enabled
                        and first_message
                        and part_index == 0
                        and topic.message_id
                        and topic.message_id != last_quote
                    ):
                        try:
                            chain.append(
                                Comp.Reply(
                                    id=topic.message_id,
                                    sender_nickname=topic.nickname or "",
                                    message_str=topic.text[:100],
                                )
                            )
                        except Exception:
                            pass
                        last_quote = topic.message_id
                    chain.append(Comp.Plain(part))
                    try:
                        msg_chain = MessageChain(chain=chain)
                    except TypeError:  # 旧版本构造签名不同
                        msg_chain = MessageChain(chain)
                    await event.send(msg_chain)
                    first_message = False
                    await asyncio.sleep(segment_delay)

            if block_index < len(blocks) - 1:
                await asyncio.sleep(topic_delay)

    async def terminate(self):
        self._states.clear()
