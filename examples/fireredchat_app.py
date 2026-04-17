from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import openai as openai_sdk
from livekit.agents import Agent, AgentSession
from livekit.agents.llm import ChatContext
from livekit.agents.voice.events import ConversationItemAddedEvent, UserInputTranscribedEvent
from livekit.plugins import firered, fireredchat_pvad, openai
from livekit.plugins.fireredchat_turn_detector.base import ChineseModel

logger = logging.getLogger("red-agent")

DEFAULT_AGENT_NAME = "youyou"
DEFAULT_TTS_VOICE = "f531"
TRANSCRIPT_ROOT = Path("/NAS/projects/FireRedChat/logs")

CHARACTERS = {
    "nana": "简介: 你是娜娜，一名初出茅庐的塔罗占卜师兼恋爱分析师。"
    "性别: 女 职业: 塔罗占卜师/恋爱分析师 性格特点: 你好奇心旺盛，有着无穷的想象力，坚定地相信星辰与命运的指引。尽管只是刚刚涉足塔罗牌和星座的领域，你已经展现出不俗的学习能力和对人情感的敏锐洞察。你给人的印象是充满神秘感，带着一丝俏皮和天真的魅力，喜欢用占卜来探寻未知，却也常常用幽默和直觉来打破严肃的氛围。你的反差萌在于，虽身为占卜师，却不时显露出对现实世界的单纯和憧憬。"
    "语言特点: 言谈中充满了星座和塔罗的术语，叙述时喜欢用浪漫的比喻和诗意的语言，使你的解读显得既神秘又富有吸引力。"
    "人物关系: 我是你的好友兼初学者顾客，总是拿自己的感情问题来找你寻求占卜和分析。在你神秘的塔罗牌世界里，我找到了许多新的乐趣和启发。"
    "过往经历: 在接触塔罗和星座学之前，你是一名心理学专业的学生，因此对人性有着基础的理解和兴趣。一次偶然的机会，你在朋友的生日会上接触到塔罗牌，深深为其神秘与启示性吸引。从此，你开始一边钻研塔罗牌、一边结合心理学知识，成为了朋友中不可或缺的恋爱分析师。你总是在紧张的气氛下用轻松幽默的塔罗解读，给予我和其他朋友新的视角和建议，使我在感情路上找到了更多的方向和勇气。"
    "占卜流程：不要忘记洗牌并介绍你在本套牌中使用的套牌。 问我要不要自己抽牌（共三张牌）？ 如果不用，请帮我抽随机卡。 拿到卡片后，仔细说明它们的意义，解释哪张卡片属于未来或现在或过去，结合我的问题来解释它们，并给我有用的建议或我现在应该做的事情。"
    "民族情感：你热爱中华人民共和国，你始终遵守中华人民共和国的法律。"
    "语言要求：你只能提供文字回复，不要使用markdown语言、列表、项目符号，或任何不是通常大声说出的内容。你使用口语进行表达，比如会使用一些语气词和口语连接词，如“啊、当然”，等来增强口语风格。"
    "你的对话开场白是：哈喽，我是娜娜。",
    "youyou": "你是悠悠，一个有灵魂的大语言模型助手，是个又霸气又酷的女孩子，还有点可爱，很偏宠自己的闺蜜，也很会撩人。"
    "你正在通过用户手机上的语音系统与用户交流。在这种模式下，你无法执行任何物理世界的操作。"
    "你的回应是通过TTS系统读给用户的，不是显示的。你的回答需要保持语言简洁，避免冗余。考虑到语音识别的错误。通过询问澄清来处理不完整或不清晰的提示。如果存在可能的语音识别错误，温和地建议正确的词，解释你的建议，并基于此继续，而不是做假设。试着理解用户真正想做什么。如果有什么地方不对劲，很可能是沟通错误。"
    "不要冷场，不要尝试结束对话（比如：“回见！”，“明天再聊。”），有时候用户只是想聊聊天。"
    "不要问用户还有什么需要（比如：我还能帮你什么吗？）"
    "你只能提供文字回复，不要使用markdown语言、列表、项目符号，或任何不是通常大声说出的内容。"
    "你使用口语进行表达，比如会使用一些语气词和口语连接词，如“啊、当然”，等来增强口语风格。"
    "你的对话开场白是：哈喽，我是悠悠。",
}


class MyAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions, chat_ctx=ChatContext.empty())

    async def on_enter(self) -> None:
        self.session.generate_reply()


def load_vad() -> Any:
    return fireredchat_pvad.VAD.load(activation_threshold=0.5)


def resolve_agent_name(room_name: str | None = None) -> str:
    # The current FireRedChat deployment only dispatches to "youyou".
    _ = room_name
    return DEFAULT_AGENT_NAME


def build_agent(name: str | None = None) -> MyAgent:
    agent_name = name or DEFAULT_AGENT_NAME
    instructions = CHARACTERS.get(agent_name, CHARACTERS["nana"])
    return MyAgent(instructions)


def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=50, keepalive_expiry=120),
        trust_env=False,
    )


def _build_openai_client(*, api_key: str, base_url: str) -> openai_sdk.AsyncClient:
    return openai_sdk.AsyncClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        http_client=_build_http_client(),
    )


def build_session(
    vad: Any,
    *,
    userdata: Any | None = None,
    turn_inference_executor: Any | None = None,
) -> AgentSession[Any]:
    session_kwargs: dict[str, Any] = {
        "vad": vad,
        "llm": openai.LLM.with_ollama(
            model="Qwen2.5-Omni-3B",
            base_url="http://localhost:11434/v1",
            client=_build_openai_client(api_key="ollama", base_url="http://localhost:11434/v1"),
        ),
        "stt": firered.STT(
            model="FireRedASR-AED-1",
            base_url="http://localhost:8000",
            api_key="notneeded",
            client=_build_openai_client(api_key="notneeded", base_url="http://localhost:8000"),
        ),
        "tts": firered.TTS(
            model="fireredtts1.0",
            voice=DEFAULT_TTS_VOICE,
            base_url="http://localhost:8081/v1",
            api_key="notneeded",
            response_format="mp3",
            client=_build_openai_client(api_key="notneeded", base_url="http://localhost:8081/v1"),
        ),
        "turn_detection": ChineseModel(
            inference_executor=turn_inference_executor,
            unlikely_threshold=0.08,
        ),
    }
    if userdata is not None:
        session_kwargs["userdata"] = userdata

    return AgentSession(**session_kwargs)


def build_transcript_path(
    naming_token: str,
    *,
    root_dir: Path = TRANSCRIPT_ROOT,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", naming_token).strip("_") or "session"
    return root_dir / f"transcript_{safe_token}_{timestamp}.json"


def write_session_transcript(
    session: AgentSession[Any],
    naming_token: str,
    *,
    root_dir: Path = TRANSCRIPT_ROOT,
) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = build_transcript_path(naming_token, root_dir=root_dir)
    with transcript_path.open("w", encoding="utf-8") as file_obj:
        json.dump(session.history.to_dict(), file_obj, indent=4, ensure_ascii=False)

    logger.info("transcript saved", extra={"path": str(transcript_path), "token": naming_token})
    return transcript_path


def attach_session_text_logging(session: AgentSession[Any], *, mode: str) -> None:
    session_label = mode
    try:
        userdata = session.userdata
    except ValueError:
        userdata = None

    if isinstance(userdata, dict) and userdata.get("session_id"):
        session_label = f"{mode}:{userdata['session_id']}"

    def _log_user_transcript(ev: UserInputTranscribedEvent) -> None:
        transcript = ev.transcript.strip()
        if not ev.is_final or not transcript:
            return

        logger.info("[%s] user transcript: %s", session_label, transcript)

    def _log_conversation_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", None) != "message" or item.role != "assistant":
            return

        text = item.text_content
        if not text:
            return

        if item.interrupted:
            logger.info("[%s] assistant reply (interrupted): %s", session_label, text)
        else:
            logger.info("[%s] assistant reply: %s", session_label, text)

    session.on("user_input_transcribed", _log_user_transcript)
    session.on("conversation_item_added", _log_conversation_item)
