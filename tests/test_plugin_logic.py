"""插件主体逻辑测试：用轻量桩替换 astrbot，验证合并、提示注入与分条引用发送。"""

import asyncio
import importlib
import logging
import os
import sys
import time
import types
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []

    api = types.ModuleType("astrbot.api")
    api.__path__ = []
    api.logger = logging.getLogger("multi-topic-test")

    # ---- message components -------------------------------------------
    components = types.ModuleType("astrbot.api.message_components")

    class Plain:
        def __init__(self, text: str = "", **_):
            self.text = text

    class Reply:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id")
            self.sender_nickname = kwargs.get("sender_nickname", "")
            self.message_str = kwargs.get("message_str", "")

    for cls in (Plain, Reply):
        setattr(components, cls.__name__, cls)

    # ---- message event result -----------------------------------------
    result_mod = types.ModuleType("astrbot.core.message.message_event_result")

    class MessageChain:
        def __init__(self, chain=None, **_):
            self.chain = list(chain or [])

    class ResultContentType:
        LLM_RESULT = "llm_result"
        STREAMING_RESULT = "streaming_result"
        STREAMING_FINISH = "streaming_finish"

    result_mod.MessageChain = MessageChain
    result_mod.ResultContentType = ResultContentType

    # ---- event / filter ------------------------------------------------
    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.MessageChain = MessageChain

    class AstrMessageEvent:
        pass

    class _Filter:
        class EventMessageType:
            ALL = "ALL"

        @staticmethod
        def event_message_type(_event_type, priority=0):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def on_llm_request(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def on_decorating_result(*_args, **kwargs):
            _ = kwargs.get("priority", 0)

            def decorator(func):
                return func

            return decorator

        @staticmethod
        def on_agent_done(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.filter = _Filter()

    # ---- star ----------------------------------------------------------
    star_mod = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context):
            self.context = context

    star_mod.Star = Star
    star_mod.Context = object

    # ---- provider ------------------------------------------------------
    provider_mod = types.ModuleType("astrbot.api.provider")

    class ProviderRequest:
        def __init__(self, prompt=""):
            self.prompt = prompt
            self.extra_user_content_parts = []

    provider_mod.ProviderRequest = ProviderRequest

    # ---- agent.message -------------------------------------------------
    core = types.ModuleType("astrbot.core")
    core.__path__ = []
    agent = types.ModuleType("astrbot.core.agent")
    agent.__path__ = []
    message = types.ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text=""):
            self.text = text
            self.temporary = False

        def mark_as_temp(self):
            self.temporary = True
            return self

    message.TextPart = TextPart

    message_mod = types.ModuleType("astrbot.core.message")
    message_mod.__path__ = []
    message_mod.message_event_result = result_mod

    modules = {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event_mod,
        "astrbot.api.star": star_mod,
        "astrbot.api.provider": provider_mod,
        "astrbot.api.message_components": components,
        "astrbot.core": core,
        "astrbot.core.agent": agent,
        "astrbot.core.agent.message": message,
        "astrbot.core.message": message_mod,
        "astrbot.core.message.message_event_result": result_mod,
    }
    sys.modules.update(modules)

    astrbot.api = api
    astrbot.core = core
    api.event = event_mod
    api.star = star_mod
    api.provider = provider_mod
    api.message_components = components
    core.agent = agent
    core.message = message_mod
    agent.message = message
    message_mod.message_event_result = result_mod


_install_astrbot_stubs()

main = importlib.import_module("main")
Comp = sys.modules["astrbot.api.message_components"]
ResultContentType = sys.modules["astrbot.core.message.message_event_result"].ResultContentType


class FakeEvent:
    def __init__(self, text="", message_id="1", umo="test:private:1", components=None):
        self.message_str = text
        self.message_obj = SimpleNamespace(
            message_str=text,
            message_id=message_id,
            message=components if components is not None else [Comp.Plain()],
            sender=SimpleNamespace(user_id="1", nickname="tester"),
        )
        self.unified_msg_origin = umo
        self._extras = {}
        self.sent = []
        self.stopped = False
        self._result = None

    def get_platform_name(self):
        return "aiocqhttp"

    def get_sender_name(self):
        return "tester"

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def stop_event(self):
        self.stopped = True

    async def send(self, chain):
        self.sent.append(chain)

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def clear_result(self):
        self._result = None


class FakeResult:
    def __init__(self, text):
        self.chain = [Comp.Plain(text)]
        self.result_content_type = ResultContentType.LLM_RESULT


class FakeReq:
    def __init__(self, prompt=""):
        self.prompt = prompt
        self.extra_user_content_parts = []


class FakeConfigContext:
    def __init__(self, streaming_response=False, streaming_segmented=False):
        self._config = {
            "provider_settings": {
                "streaming_response": streaming_response,
                "streaming_segmented": streaming_segmented,
            }
        }

    def get_config(self, umo=None):
        return self._config


def build_plugin(context=None, **overrides):
    config = {
        "enabled": True,
        "merge_window_seconds": 0,
        "debounce_max_chars": 80,
        "pending_ttl": 120,
        "max_topics": 6,
        "min_topic_chars": 2,
        "continuation_max_chars": 6,
        "send_quotes": True,
        "streaming_guard": True,
        "segment_delay": 0,
        "topic_delay": 0,
        "max_segment_chars": 120,
        "command_prefixes": ["/", "／"],
    }
    config.update(overrides)
    return main.MultiTopicPlugin(context, config)


def topic(text, message_id="1", nickname="tester"):
    return main._Topic(text, message_id, nickname, time.monotonic())


class CollectTest(unittest.IsolatedAsyncioTestCase):
    async def _send_pair(self, plugin, first, second, gap=0.03):
        e1 = FakeEvent(first, message_id="100")
        e2 = FakeEvent(second, message_id="101")
        t1 = asyncio.create_task(plugin._collect_message(e1))
        await asyncio.sleep(gap)
        t2 = asyncio.create_task(plugin._collect_message(e2))
        await asyncio.gather(t1, t2)
        return e1, e2

    async def test_two_messages_are_merged(self):
        plugin = build_plugin(merge_window_seconds=0.12)
        e1, e2 = await self._send_pair(
            plugin, "你下午吃了什么？", "你晚上想干点啥？"
        )

        self.assertTrue(e1.stopped)
        self.assertEqual(len(e2._mt_topics), 2)
        self.assertEqual(
            e2.message_str, "你下午吃了什么？\n你晚上想干点啥？"
        )
        self.assertFalse(e2.get_extra("enable_streaming"))

    async def test_single_message_not_taken_over(self):
        plugin = build_plugin(merge_window_seconds=0.03)
        event = FakeEvent("你下午吃了什么？")
        await plugin._collect_message(event)

        self.assertFalse(event.stopped)
        self.assertFalse(hasattr(event, "_mt_topics"))

    async def test_single_message_with_two_questions_takes_over(self):
        plugin = build_plugin(merge_window_seconds=0.05)
        event = FakeEvent("今天天气怎么样？晚饭吃什么？")
        await plugin._collect_message(event)

        # 一条消息里多个问题：消息只有 1 条，但仍然接管（由模型拆话题）
        self.assertTrue(hasattr(event, "_mt_topics"))
        self.assertEqual(len(event._mt_topics), 1)

    async def test_multiple_messages_are_kept_together(self):
        plugin = build_plugin(merge_window_seconds=0.1)
        _, e2 = await self._send_pair(plugin, "我今天去公园了", "然后呢")

        # 交给模型判断这两条是不是同一话题，插件不再自己折叠
        self.assertEqual(len(e2._mt_topics), 2)

    async def test_command_not_collected(self):
        plugin = build_plugin(merge_window_seconds=0.03)
        event = FakeEvent("/help")
        await plugin._collect_message(event)
        self.assertFalse(hasattr(event, "_mt_topics"))


class HintTest(unittest.IsolatedAsyncioTestCase):
    async def test_hint_contains_markers_and_topics(self):
        plugin = build_plugin()
        event = FakeEvent("合并后的文本")
        event._mt_topics = [topic("你下午吃了什么？", "100"), topic("你晚上想干点啥？", "101")]
        req = FakeReq("合并后的文本")

        await plugin._inject_topic_hint(event, req)

        self.assertEqual(len(req.extra_user_content_parts), 1)
        hint = req.extra_user_content_parts[0].text
        self.assertIn("[[T1]]", hint)
        self.assertIn("[[T2]]", hint)
        self.assertIn("你下午吃了什么？", hint)
        self.assertNotIn("1. 今天", hint)

    async def test_hint_injected_once(self):
        plugin = build_plugin()
        event = FakeEvent("x")
        event._mt_topics = [topic("A", "1"), topic("B", "2")]
        req = FakeReq("x")

        await plugin._inject_topic_hint(event, req)
        await plugin._inject_topic_hint(event, req)

        self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_streaming_guard_skips_hint(self):
        plugin = build_plugin(
            context=FakeConfigContext(streaming_response=True, streaming_segmented=True)
        )
        event = FakeEvent("x")
        event._mt_topics = [topic("A", "1"), topic("B", "2")]
        req = FakeReq("x")

        await plugin._inject_topic_hint(event, req)

        self.assertEqual(req.extra_user_content_parts, [])
        self.assertIsNone(event._mt_topics)


class SendRepliesTest(unittest.IsolatedAsyncioTestCase):
    async def test_splits_blocks_and_quotes_each_topic(self):
        plugin = build_plugin()
        event = FakeEvent("你下午吃了什么？\n你晚上想干点啥？")
        event._mt_topics = [
            topic("你下午吃了什么？", "100"),
            topic("你晚上想干点啥？", "101"),
        ]
        event.set_result(
            FakeResult(
                "[[T1]]\n刚吃了碗牛肉面。\n[[S]]\n有点咸。\n"
                "[[T2]]\n晚上打算出去走走。"
            )
        )

        await plugin._send_topic_replies(event)

        self.assertEqual(len(event.sent), 3)
        first = event.sent[0].chain
        self.assertEqual(first[0].id, "100")
        self.assertEqual(first[1].text, "刚吃了碗牛肉面。")
        second = event.sent[1].chain
        self.assertFalse(any(isinstance(c, Comp.Reply) for c in second))
        self.assertEqual(second[0].text, "有点咸。")
        third = event.sent[2].chain
        self.assertEqual(third[0].id, "101")
        self.assertIsNone(event.get_result())

    async def test_quotes_deduplicated_for_same_source_message(self):
        plugin = build_plugin()
        event = FakeEvent("今天天气怎么样？晚饭吃什么？")
        event._mt_topics = [
            topic("今天天气怎么样？", "100"),
            topic("晚饭吃什么？", "100"),
        ]
        event.set_result(FakeResult("[[T1]]\n多云。\n[[T2]]\n吃面。"))

        await plugin._send_topic_replies(event)

        self.assertEqual(len(event.sent), 2)
        self.assertEqual(event.sent[0].chain[0].id, "100")
        self.assertFalse(
            any(isinstance(c, Comp.Reply) for c in event.sent[1].chain)
        )

    async def test_no_markers_keeps_normal_reply(self):
        plugin = build_plugin()
        event = FakeEvent("x")
        event._mt_topics = [topic("A", "1"), topic("B", "2")]
        event.set_result(FakeResult("普通的一段回复。"))

        await plugin._send_topic_replies(event)

        self.assertEqual(event.sent, [])
        self.assertEqual(
            event.get_result().chain[0].text, "普通的一段回复。"
        )

    async def test_cn_markers_are_supported(self):
        plugin = build_plugin()
        event = FakeEvent("x")
        event._mt_topics = [topic("A", "1"), topic("B", "2")]
        event.set_result(FakeResult("【T1】\n甲。\n【S】\n乙。\n【T2】\n丙。"))

        await plugin._send_topic_replies(event)

        self.assertEqual(len(event.sent), 3)


class HistoryCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_markers_removed_before_history_save(self):
        plugin = build_plugin()
        event = FakeEvent("x")
        event._mt_topics = [topic("A", "1"), topic("B", "2")]
        message = SimpleNamespace(
            role="assistant",
            content="[[T1]]\n甲。\n[[S]]\n乙。\n[[T2]]\n丙。",
        )
        run_context = SimpleNamespace(messages=[message])

        await plugin._clean_history_markers(event, run_context, None)

        self.assertEqual(message.content, "甲。\n乙。\n丙。")


class ParseBlocksTest(unittest.TestCase):
    def test_parse_two_topics(self):
        blocks = main._parse_blocks(
            "[[T1]]\n第一句\n[[S]]\n第二句\n[[T2]]\n回答二"
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].segments, ["第一句", "第二句"])
        self.assertEqual(blocks[1].segments, ["回答二"])

    def test_strip_markers(self):
        self.assertEqual(
            main._strip_markers("[[T1]]\n你好\n[[S]]\n世界"),
            "你好\n世界",
        )

    def test_full_width_markers(self):
        blocks = main._parse_blocks(
            "［［T1］］\n甲。\n［［S］］\n乙。\n［［T2］］\n丙。"
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].segments, ["甲。", "乙。"])
        self.assertEqual(blocks[1].segments, ["丙。"])

    def test_chinese_bracket_markers(self):
        blocks = main._parse_blocks("【T1】\n甲。\n【S】\n乙。\n【T2】\n丙。")
        self.assertEqual(len(blocks), 2)

    def test_zero_width_chars_in_markers(self):
        blocks = main._parse_blocks("[[T\u200b1]]\n甲。\n[[T\u200b2]]\n乙。")
        self.assertEqual(len(blocks), 2)

    def test_strip_full_width_markers(self):
        cleaned = main._strip_markers("［［T1］］\n你好\n［［S］］\n世界")
        self.assertEqual(cleaned, "你好\n世界")

    def test_exotic_brackets(self):
        blocks = main._parse_blocks("⟦T1⟧\n甲。\n⟦S⟧\n乙。\n⟦T2⟧\n丙。")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].segments, ["甲。", "乙。"])

    def test_line_mode_markers(self):
        blocks = main._parse_blocks("T1\n甲。\nS\n乙。\nT2\n丙。")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].segments, ["甲。", "乙。"])
        self.assertEqual(blocks[1].segments, ["丙。"])

    def test_strip_exotic_and_line_markers(self):
        self.assertEqual(
            main._strip_markers("⟦T1⟧\n你好\n⟦S⟧\n世界"),
            "你好\n世界",
        )
        self.assertEqual(
            main._strip_markers("T1\n你好\nS\n世界"),
            "你好\n世界",
        )

    def test_strip_leaves_normal_text(self):
        text = "今天过得还行，晚上想吃面。"
        self.assertEqual(main._strip_markers(text), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
