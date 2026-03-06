import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _resolve_question(args: argparse.Namespace) -> str:
    if isinstance(args.question, str) and args.question.strip():
        return args.question.strip()
    if args.question_parts:
        joined = " ".join(args.question_parts).strip()
        if joined:
            return joined
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    raise ValueError("Question is empty. Use -q or pass question as positional text.")


def _extract_answer(result: dict[str, Any]) -> str:
    llm_response = result.get("llm_response", {})
    if isinstance(llm_response, dict):
        content = llm_response.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    message = result.get("message", "")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return ""


def _ask_lightrag(question: str, timeout_override: float | None) -> str:
    if timeout_override is not None:
        os.environ["RAG_RETRIEVE_TIMEOUT_SEC"] = str(timeout_override)

    from app.rag.retriever import query_native

    result = query_native(question)
    status = result.get("status")
    answer = _extract_answer(result)
    if status != "success":
        raise RuntimeError(answer or "LightRAG query failed")
    return answer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask via project LightRAG pipeline, using .env settings."
    )
    parser.add_argument("question_parts", nargs="*", help="Question text")
    parser.add_argument("-q", "--question", help="Question text (alternative form)")
    parser.add_argument(
        "--system",
        default=None,
        help="Ignored. System prompt is controlled by LightRAG pipeline.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override RAG_RETRIEVE_TIMEOUT_SEC for this run",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        question = _resolve_question(args)
        if args.system:
            print(
                "Warning: --system is ignored for LightRAG mode.",
                file=sys.stderr,
            )
        answer = _ask_lightrag(question, args.timeout)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        print(answer)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((answer + "\n").encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
