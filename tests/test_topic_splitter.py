"""topic_splitter 的单元测试（不依赖 AstrBot）。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic_splitter import is_greeting, split_topics  # noqa: E402


class SplitTopicsTest(unittest.TestCase):
    def check(self, text, expected_count, **kwargs):
        topics = split_topics(text, **kwargs)
        self.assertEqual(
            len(topics),
            expected_count,
            msg=f"text={text!r} -> {topics!r}",
        )
        return topics

    def test_two_questions_in_one_message(self):
        topics = self.check("今天天气怎么样？晚饭吃什么？", 2)
        self.assertIn("今天天气怎么样？", topics)
        self.assertIn("晚饭吃什么？", topics)

    def test_single_question(self):
        self.check("今天几号？", 0)

    def test_greeting_only(self):
        self.check("你好呀", 0)

    def test_requirements_list_not_split(self):
        self.check(
            "帮我写个 Python 快排，要求：\n1. 读取文件\n2. 统计词频",
            0,
        )

    def test_numbered_questions_split(self):
        self.check("1. 明天天气怎么样？\n2. 晚饭吃什么？", 2)

    def test_numbered_requirements_without_question_not_split(self):
        self.check("帮我写个函数：\n1. 输入两个数\n2. 输出和", 0)

    def test_transition_word(self):
        self.check("我想问下明天天气，另外晚饭吃什么", 2)

    def test_statement_then_question_not_split(self):
        self.check("你知道为什么吗？因为下雨了。", 0)

    def test_narrative_not_split(self):
        self.check("我今天去公园了。然后呢，我碰到一只猫。", 0)

    def test_ordinal_questions(self):
        topics = self.check("第一，明天天气怎么样？第二，晚饭吃什么？", 2)
        self.assertTrue(all(not t.startswith("第一") for t in topics))

    def test_two_imperative_lines(self):
        self.check("帮我写个快排\n帮我查下明天北京的天气", 2)

    def test_short_greeting_plus_topic(self):
        self.check("在吗\n帮我查下明天天气", 0)

    def test_greeting_plus_question(self):
        self.check("你好\n今天天气怎么样？", 0)

    def test_code_block_skipped(self):
        self.check("```python\nprint(1)\nprint(2)\n```", 0)

    def test_multi_line_questions(self):
        self.check("今天天气怎么样？\n晚饭吃什么？", 2)

    def test_too_many_topics_not_split(self):
        text = "\n".join(f"第{i}个问题是什么情况？" for i in range(9))
        self.check(text, 0, max_topics=6)

    def test_min_chars(self):
        # 两个都短于阈值 → 不拆
        self.check("吃了吗？\n睡了吗？", 0, min_topic_chars=10)

    def test_english_questions(self):
        self.check("How is the weather? What's for dinner?", 2)


class GreetingTest(unittest.TestCase):
    def test_greetings(self):
        for text in ("你好", "在吗", "在吗？", "Hello!", "晚安~", "早上好呀"):
            self.assertTrue(is_greeting(text), text)

    def test_not_greetings(self):
        for text in ("你好，今天天气怎么样", "在吗？帮我查个东西", "晚上好，吃了吗"):
            self.assertFalse(is_greeting(text), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
