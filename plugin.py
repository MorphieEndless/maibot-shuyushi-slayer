"""属于是猎杀者 (Yushuyu Slayer)

在出站消息真正发送前，按概率随机清除「属于是 / 了属于是」口癖。

**为什么需要它**

MaiBot 会把 bot 自己的历史回复当作 few-shot 上下文回灌给模型。一旦模型说过一次
「属于是」，下一次请求里就多一条「我自己平时就是这么说话」的示范，形成自我强化的
复读回路 —— 光靠 prompt 约束压不住，因为示范样本一直躺在上下文里。

本插件挂在发送链路的最后一环，对最终可见文本做概率性清除，把上下文里自我示范的
密度逐步稀释掉。

**为什么挂在 send_service.before_send**

这是唯一能同时净化「发出去的消息」和「写库的历史」的位置：

1. 出站消息进入 Platform IO 之前的最后一站，此后不再被改写；
2. 写库（``_store_sent_message``）发生在本 Hook 之后，因此数据库里留下的
   也是净化后的文本 —— 这一点很关键，落库文本正是下一轮上下文的来源。

**实现要点**

- 只清理 ``raw_message`` 里的 ``text`` 组件与 ``processed_plain_text``，
  从不触碰 ``ReplyComponent.target_message_content``（那是被引用群友的原话，
  不是 bot 自己的口癖）。
- 两个文本视图共用一张「按出现序号惰性生成」的随机判定表，保证发出去的文本
  和落库的文本在这批位置上被清理得一致。
- 改写结果通过返回值 ``modified_kwargs`` 交回宿主。插件运行在独立进程里，
  宿主侧先把消息 ``serialize`` 再传进来，就地修改入参无法跨进程生效，
  因此这里只构造新对象、不做原地改写。
- 任何异常都被吞掉并记日志，绝不阻塞发送。
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from maibot_sdk import Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder


# 默认猎杀模式：可选的「了」一起吃棹，避免留下孤零零的「了」。
DEFAULT_PATTERNS: Tuple[str, ...] = (r"了?属于是",)

# 判定「这段文本还有没有实际内容」：空白与纯标点不算内容，
# emoji、字母数字、汉字都算 —— 所以「😅」是有效内容，「？」不是。
_PUNCTUATION_ONLY = set(
    " \t\r\n\u3000，、。！？；：,.;:!?~～…·—-_/\\|()（）[]【】{}<>《》“”‘’\"'"
)

# 清理残留：开头的标点、以及被掏空后留下的连续空白。
_LEADING_PUNCT = re.compile(r"^[\s，、。！？；：,.;:!?~～…]+")
_MULTI_SPACE = re.compile(r"[ \t\u3000]{2,}")


def _has_meaningful_text(text: str) -> bool:
    """判断文本里是否还剩下「不是标点」的字符。"""

    return any(ch not in _PUNCTUATION_ONLY for ch in text)


class PluginSectionConfig(PluginConfigBase):
    """``[plugin]`` 段：由运行时统一读取的启用开关。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    config_version: str = Field(default="1.0.0", description="配置版本")
    enabled: bool = Field(default=True, description="是否启用本插件")


class SlayerSectionConfig(PluginConfigBase):
    """``[slayer]`` 段：猎杀策略。"""

    __ui_label__ = "属于是猎杀"
    __ui_icon__ = "scissors"
    __ui_order__ = 1

    enabled: bool = Field(default=True, description="是否启用出站口癖清除")
    kill_ratio: float = Field(default=0.7, description="每个「属于是」被清除的概率 (0~1)，0.7 即杀掉七成")
    patterns: List[str] = Field(
        default_factory=lambda: list(DEFAULT_PATTERNS),
        description="猎杀用正则列表，默认 「了?属于是」；同时命中时按左起优先匹配",
    )
    cleanup_text: bool = Field(default=True, description="清除后修掉残留的开头标点与多余空白")
    guard_empty: bool = Field(default=True, description="若清除后整条消息不再含任何文字，则整条回滚")
    max_kills_per_message: int = Field(default=0, description="单条消息最多清除几处，0 表示不限")
    sync_processed_text: bool = Field(
        default=True,
        description="同步净化 processed_plain_text（落库用的纯文本），关掉则只改发出去的内容",
    )
    dry_run: bool = Field(default=False, description="只记录不修改，用于观察命中率")
    log_kills: bool = Field(default=True, description="每清除一处时输出日志")
    only_groups: List[str] = Field(default_factory=list, description="只在这些群生效，留空表示全部群")
    skip_groups: List[str] = Field(default_factory=list, description="在这些群里跳过")


class YushuyuSlayerConfig(PluginConfigBase):
    """插件总配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    slayer: SlayerSectionConfig = Field(default_factory=SlayerSectionConfig)


class _Decisions:
    """按「出现序号」惰性生成的随机判定表。

    同一次发送里，``raw_message`` 的文本组件与 ``processed_plain_text`` 是两个
    视图，但「第几个属于是」的顺序是一致的。共用同一张判定表，就能保证两个视图
    杀掉的是同一批位置，发出去的内容和落库的内容不会互相打架。
    """

    def __init__(self, kill_ratio: float) -> None:
        self._kill_ratio = min(1.0, max(0.0, float(kill_ratio)))
        self._cache: List[bool] = []

    def at(self, index: int) -> bool:
        """返回第 ``index`` 个出现位置的判定结果，同一序号恒定。"""

        while len(self._cache) <= index:
            self._cache.append(random.random() < self._kill_ratio)
        return self._cache[index]


def compile_patterns(patterns: Sequence[str]) -> Optional[re.Pattern[str]]:
    """把配置里的多个正则合并成一个左起优先的匹配器。

    Args:
        patterns: 正则字符串列表。

    Returns:
        Optional[re.Pattern[str]]: 合并后的匹配器；没有合法模式时返回 ``None``。
    """

    valid = [str(p).strip() for p in patterns if str(p or "").strip()]
    if not valid:
        return None
    try:
        return re.compile("|".join(f"(?:{p})" for p in valid))
    except re.error:
        return None


def cleanup_text(text: str) -> str:
    """清除后修掉残留：开头的孤立标点与连续空白。"""

    cleaned = _LEADING_PUNCT.sub("", text)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    return cleaned.strip()


def slay_text(
    text: str,
    combined: re.Pattern[str],
    decisions: _Decisions,
    *,
    offset: int = 0,
    max_kills: int = 0,
    cleanup: bool = True,
) -> Tuple[str, int, int]:
    """对单段文本执行概率清除。

    Args:
        text: 待处理的文本。
        combined: 合并后的匹配器。
        decisions: 共享的判定表。
        offset: 本段文本的第一个出现位置在全局的序号。
        max_kills: 本段最多清除几处，``0`` 表示不限。
        cleanup: 是否修掉残留标点与空白。

    Returns:
        Tuple[str, int, int]: ``(新文本, 清除处数, 下一个全局序号)``。
    """

    if not text:
        return text, 0, offset

    pieces: List[str] = []
    cursor = 0
    killed = 0
    index = offset

    for match in combined.finditer(text):
        pieces.append(text[cursor:match.start()])
        if (max_kills <= 0 or killed < max_kills) and decisions.at(index):
            killed += 1
        else:
            pieces.append(match.group(0))
        cursor = match.end()
        index += 1

    pieces.append(text[cursor:])
    new_text = "".join(pieces)

    if killed and cleanup:
        new_text = cleanup_text(new_text)

    return new_text, killed, index


def slay_components(
    components: Sequence[Any],
    combined: re.Pattern[str],
    decisions: _Decisions,
    *,
    max_kills: int,
    cleanup: bool,
) -> Tuple[List[Any], int]:
    """清除 ``raw_message`` 里所有 text 组件的口癖。

    只动 ``type == "text"`` 的组件；``reply`` / ``at`` / ``image`` 等一概不碰。

    Returns:
        Tuple[List[Any], int]: ``(新的组件列表, 清除处数)``。
    """

    new_components: List[Any] = []
    killed_total = 0
    index = 0  # 跨组件累加的全局出现序号

    for component in components:
        if not isinstance(component, Mapping) or component.get("type") != "text":
            new_components.append(component)
            continue

        raw_text = component.get("data")
        if not isinstance(raw_text, str) or not raw_text:
            new_components.append(component)
            continue

        new_text, killed, index = slay_text(
            raw_text,
            combined,
            decisions,
            offset=index,
            max_kills=max(0, max_kills - killed_total),
            cleanup=cleanup,
        )
        killed_total += killed

        # 被掏空的文本组件直接丢掉，避免发出空消息段。
        if killed and not new_text:
            continue

        if new_text != raw_text:
            replaced = dict(component)
            replaced["data"] = new_text
            new_components.append(replaced)
        else:
            new_components.append(component)

    return new_components, killed_total


def _joined_text(components: Sequence[Any]) -> str:
    """把 text 组件拼起来，用于判断整条消息还剩不剩内容。"""

    parts: List[str] = []
    for component in components:
        if isinstance(component, Mapping) and component.get("type") == "text":
            data = component.get("data")
            if isinstance(data, str) and data:
                parts.append(data)
    return " ".join(parts)


class YushuyuSlayerPlugin(MaiBotPlugin):
    """属于是猎杀者。"""

    config_model = YushuyuSlayerConfig

    def __init__(self) -> None:
        super().__init__()
        self._total_kills = 0
        self._total_messages = 0
        self._compiled: Optional[re.Pattern[str]] = None
        self._compiled_key: Optional[Tuple[str, ...]] = None

    # ─── 生命周期 ───────────────────────────────────────────────

    async def on_load(self) -> None:
        cfg = self._slayer_config()
        self.ctx.logger.info(
            "属于是猎杀者已上膛: "
            f"kill_ratio={cfg.kill_ratio}, patterns={list(cfg.patterns)}, "
            f"dry_run={cfg.dry_run}"
        )

    async def on_unload(self) -> None:
        self.ctx.logger.info(
            f"属于是猎杀者已卸载: 累计清除 {self._total_kills} 处，"
            f"涉及 {self._total_messages} 条消息"
        )

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        # 配置热更新时丢弃编译缓存，下次处理自动按新 patterns 重建。
        self._compiled = None
        self._compiled_key = None
        self.ctx.logger.info(f"属于是猎杀者配置已更新: scope={scope}, version={version}")

    # ─── 内部辅助 ───────────────────────────────────────────────

    def _slayer_config(self) -> SlayerSectionConfig:
        return self.config.slayer

    def _get_compiled(self, patterns: Sequence[str]) -> Optional[re.Pattern[str]]:
        key = tuple(str(p) for p in patterns)
        if self._compiled is None or self._compiled_key != key:
            self._compiled = compile_patterns(patterns)
            self._compiled_key = key
        return self._compiled

    @staticmethod
    def _extract_group_id(message: Mapping[str, Any]) -> str:
        message_info = message.get("message_info")
        if not isinstance(message_info, Mapping):
            return ""
        group_info = message_info.get("group_info")
        if not isinstance(group_info, Mapping):
            return ""
        return str(group_info.get("group_id") or "").strip()

    # ─── 核心 Hook ──────────────────────────────────────────────

    @HookHandler(
        "send_service.before_send",
        name="yushuyu_slayer",
        description="出站消息发送前按概率清除「属于是」口癖",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_send(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """在出站消息进入 Platform IO 之前清除口癖。

        Returns:
            Optional[Dict[str, Any]]: 需要改写时的 Hook 结果；无需改写时返回 ``None``。
        """

        try:
            return self._process_send(kwargs)
        except Exception as exc:  # 任何意外都不该影响发送
            self.ctx.logger.exception(f"属于是猎杀者执行异常，已放行原始消息: {exc}")
            return None

    def _process_send(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cfg = self._slayer_config()
        if not cfg.enabled or cfg.kill_ratio <= 0:
            return None

        message = kwargs.get("message")
        if not isinstance(message, dict):
            return None

        group_id = self._extract_group_id(message)
        if cfg.only_groups and group_id not in {str(g).strip() for g in cfg.only_groups}:
            return None
        if group_id and group_id in {str(g).strip() for g in cfg.skip_groups}:
            return None

        combined = self._get_compiled(cfg.patterns)
        if combined is None:
            return None

        components = message.get("raw_message")
        if not isinstance(components, list) or not components:
            return None

        original_text = _joined_text(components)
        if not combined.search(original_text):
            return None

        decisions = _Decisions(cfg.kill_ratio)

        if cfg.dry_run:
            hits = len(combined.findall(original_text))
            self.ctx.logger.info(
                f"[dry_run] 命中 {hits} 处口癖，按 {cfg.kill_ratio} 概率本应清除约 "
                f"{round(hits * cfg.kill_ratio)} 处 | 原文: {original_text[:120]}"
            )
            return None

        new_components, killed = slay_components(
            components,
            combined,
            decisions,
            max_kills=cfg.max_kills_per_message,
            cleanup=cfg.cleanup_text,
        )

        if killed <= 0:
            return None

        # 保险一：组件被清空到一条不剩，反序列化会失败，直接放弃。
        if not new_components:
            self.ctx.logger.warning(
                f"清除后消息组件为空，已回滚 | 原文: {original_text[:120]}"
            )
            return None

        # 保险二：整条消息被掏空的兜底 —— 原文有内容、改完没内容，就整条回滚。
        if cfg.guard_empty and _has_meaningful_text(original_text) and not _has_meaningful_text(
            _joined_text(new_components)
        ):
            self.ctx.logger.warning(
                f"清除后整条消息不再含文字，已回滚 | 原文: {original_text[:120]}"
            )
            return None

        # 构造改写后的副本，不改动传入的 message / kwargs。
        # 宿主把消息 serialize 后才交给插件，插件又在独立进程里运行，
        # 就地修改对象不会跨进程传回去，唯一生效的通道是返回的 modified_kwargs。
        modified_message = dict(message)
        modified_message["raw_message"] = new_components

        if cfg.sync_processed_text:
            processed = message.get("processed_plain_text")
            if isinstance(processed, str) and processed:
                # 判定表按出现序号对齐，因此这里从 0 开始，与组件视图共用同一批判定。
                new_processed, processed_killed, _ = slay_text(
                    processed,
                    combined,
                    decisions,
                    offset=0,
                    max_kills=cfg.max_kills_per_message,
                    cleanup=cfg.cleanup_text,
                )
                if processed_killed > 0:
                    modified_message["processed_plain_text"] = new_processed

        modified_kwargs = dict(kwargs)
        modified_kwargs["message"] = modified_message

        self._total_kills += killed
        self._total_messages += 1

        if cfg.log_kills:
            self.ctx.logger.info(
                f"已清除 {killed} 处「属于是」 | 原文: {original_text[:120]} "
                f"| 改后: {_joined_text(new_components)[:120]}"
            )

        return {"action": "continue", "modified_kwargs": modified_kwargs}


def create_plugin() -> YushuyuSlayerPlugin:
    """插件工厂函数。"""

    return YushuyuSlayerPlugin()
