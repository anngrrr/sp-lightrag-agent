import argparse
import json
import os
import sys
from typing import Any
from urllib import error, parse, request

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


def _build_llm_kwargs(timeout_override: float | None) -> tuple[str, dict[str, Any]]:
    model = os.getenv("OLLAMA_LLM_MODEL")
    if not model:
        raise RuntimeError("OLLAMA_LLM_MODEL must be set in .env")

    host = os.getenv("OLLAMA_LLM_HOST")
    api_key = os.getenv("OLLAMA_API_KEY")

    timeout_env = os.getenv("RAG_KEYWORD_LLM_TIMEOUT_SEC", "120")
    timeout = timeout_override if timeout_override is not None else float(timeout_env)

    temperature_env = os.getenv(
        "OLLAMA_LLM_TEMPERATURE", os.getenv("OLLAMA_TEMPERATURE", "0")
    )
    temperature = float(temperature_env)

    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "options": {"temperature": temperature},
    }
    if host:
        kwargs["host"] = host
    if api_key:
        kwargs["api_key"] = api_key

    return model, kwargs


def _build_chat_url(host: str | None) -> str:
    if not host:
        raise RuntimeError("OLLAMA_LLM_HOST must be set in .env")
    parsed = parse.urlparse(host)
    if not parsed.scheme:
        raise RuntimeError(
            "OLLAMA_LLM_HOST must include scheme, example: http://localhost:11434"
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/api/chat"):
        return host
    base = host.rstrip("/")
    return f"{base}/api/chat"


def _ask_llm(
    question: str, system_prompt: str | None, timeout_override: float | None
) -> str:
    model, kwargs = _build_llm_kwargs(timeout_override)
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    api_key = kwargs.pop("api_key", None)
    options = kwargs.pop("options", {})

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        _build_chat_url(host),
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc

    data = json.loads(raw)
    content = data.get("message", {}).get("content", "")
    return str(content).strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask the base LLM directly (without GraphRAG), using .env settings."
    )
    parser.add_argument("question_parts", nargs="*", help="Question text")
    parser.add_argument("-q", "--question", help="Question text (alternative form)")
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override timeout in seconds (default: RAG_KEYWORD_LLM_TIMEOUT_SEC)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        question = _resolve_question(args)
        answer = _ask_llm(question, args.system, args.timeout)
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
