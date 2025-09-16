import logging

from dotenv import load_dotenv

from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomIO,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ChatContext
from livekit.plugins import openai, firered, fireredchat_pvad
# from livekit.plugins.dify import compatibility_fix as llm  ## dify连接请额外安装：https://github.com/bigdatatoai/livekit-plugins-dify-workflow/tree/main
from livekit.plugins.fireredchat_turn_detector.base import ChineseModel

from datetime import datetime
import json

logger = logging.getLogger("red-agent")

load_dotenv()

character = {
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
            "你的对话开场白是：哈喽，我是悠悠。"
}


class MyAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(
            instructions=instructions,
            chat_ctx=ChatContext.empty()
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = fireredchat_pvad.VAD.load(activation_threshold=0.5)

async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    voice = "f531"
    scene = ctx.room.name.rsplit("-", 1)[-1]
    logger.info(f"connecting to room {scene}")

    # dispatch agent via room name or livekit protocol, here we only use "youyou"
    name = "youyou"

    ## if Dify is used 
    # used_agent = MyAgent("")
    # used_llm = llm.DifyWorkflowLLM(llm.DifyWorkflowLLMOptions(
    #     api_base="dify-api-base",
    #     api_key="yourkey"))

    instruction = character.get(name, character["nana"])
    used_agent = MyAgent(instruction)
    used_llm = openai.LLM(model="qwen2.5-14b",
                    api_key="notneeded",
                    base_url="http://your_ollama_api"
                    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=used_llm,
        stt=firered.STT(model="FireRedASR-AED-1",
                       base_url="http://your_hosted_asr_server/fireredasr",
                       api_key="notneeded"),
        tts=firered.TTS(
            model="fireredtts1.0",
            voice=voice,
            base_url="http://your_hosted_tts_server/fireredtts1",
            api_key="notneeded",
            response_format="mp3"
            ),
        # use turn detection model
        turn_detection=ChineseModel(unlikely_threshold=0.08),
    )

    #######################
    # setup text records
    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        # This example writes to the temporary directory, but you can save to any location
        filename = f"/workspace/logs/transcript_{ctx.room.name}_{current_date}.json"
        
        with open(filename, 'w+') as f:
            json.dump(session.history.to_dict(), f, indent=4, ensure_ascii=False)
            
        print(f"Transcript for {ctx.room.name} saved to {filename}")

    ctx.add_shutdown_callback(write_transcript)

    #######################
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()

    await session.start(
        agent=used_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    # connect to the room
    await ctx.connect()

    # wait for the first participant to arrive
    participant = await ctx.wait_for_participant()

    # customize behavior based on the participant
    print(f"connected to room {ctx.room.name} with participant {participant.identity}")

    @ctx.room.local_participant.register_rpc_method("new_conversation")
    async def new_conversation(data: rtc.RpcInvocationData):
        session.interrupt()
        session.clear_user_turn()
        await session._agent.update_chat_ctx(ChatContext.empty())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, job_memory_warn_mb=1500))
