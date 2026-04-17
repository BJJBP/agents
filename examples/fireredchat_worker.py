from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    JobContext,
    JobProcess,
    RoomIO,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.cli.log import setup_logging
from livekit.agents.llm import ChatContext

from fireredchat_app import (
    attach_session_text_logging,
    build_agent,
    build_session,
    load_vad,
    resolve_agent_name,
    write_session_transcript,
)
from fireredchat_benchplayer import BenchConfig, run_bench
from fireredchat_benchplayer.config import DEFAULT_BENCH_ROOT

logger = logging.getLogger("red-agent")

load_dotenv()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = load_vad()


async def frontend_entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    scene = ctx.room.name.rsplit("-", 1)[-1]
    logger.info("connecting to room %s", scene)

    agent_name = resolve_agent_name(ctx.room.name)
    session = build_session(ctx.proc.userdata["vad"])
    attach_session_text_logging(session, mode="frontend")
    agent = build_agent(agent_name)

    async def write_transcript() -> None:
        transcript_path = write_session_transcript(session, ctx.room.name)
        print(f"Transcript for {ctx.room.name} saved to {transcript_path}")

    ctx.add_shutdown_callback(write_transcript)

    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    print(f"connected to room {ctx.room.name} with participant {participant.identity}")

    @ctx.room.local_participant.register_rpc_method("new_conversation")
    async def new_conversation(data: rtc.RpcInvocationData) -> None:
        _ = data
        session.interrupt()
        session.clear_user_turn()
        await session.current_agent.update_chat_ctx(ChatContext.empty())


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-mode", choices=("frontend", "bench"), default="frontend")
    parser.add_argument("--bench-root", type=str, default=None)
    parser.add_argument("--bench-case-timeout-s", type=float, default=180.0)
    parser.add_argument("--bench-post-roll-ms", type=int, default=500)
    parser.add_argument("--bench-concurrent-sessions", type=int, default=1)
    return parser.parse_known_args(argv)


def build_bench_config(args: argparse.Namespace) -> BenchConfig:
    bench_root = args.bench_root or str(DEFAULT_BENCH_ROOT)
    return BenchConfig(
        root_dir=Path(bench_root),
        case_timeout_s=args.bench_case_timeout_s,
        post_roll_ms=args.bench_post_roll_ms,
        concurrent_sessions=args.bench_concurrent_sessions,
    )


def main() -> None:
    args, remaining_args = parse_args(sys.argv[1:])
    if args.input_mode == "bench":
        setup_logging("INFO", True, False)
        asyncio.run(run_bench(build_bench_config(args)))
        return

    sys.argv = sys.argv[:1] + remaining_args
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=frontend_entrypoint,
            prewarm_fnc=prewarm,
            job_memory_warn_mb=1500,
            initialize_process_timeout=45.0,
        )
    )


if __name__ == "__main__":
    main()
