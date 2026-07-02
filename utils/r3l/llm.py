from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional
import io
import base64
from PIL import Image
from colorama import Fore

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from solvers.r3l.config import cfg


def extract_code_block(text: str) -> str:
    if "```" not in text:
        return text
    pattern = r"```(?:[a-z]*)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        raise ValueError(f"cannot find code block in text: {text!r}")
    return match.group(1)


def _reasoning_summary_deltas(chunk: BaseMessage) -> Iterator[str]:
    """Yield reasoning-summary text deltas from a streamed Responses-API chunk. The
    summary streams as content blocks of type 'reasoning' whose `summary` list holds
    {'type': 'summary_text', 'text': <delta>} entries (verified against the live API)."""
    content = chunk.content
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            for s in block.get("summary", []):
                if isinstance(s, dict) and s.get("type") == "summary_text":
                    text = s.get("text", "")
                    if text:
                        yield text


class ChatSession:
    _MODES = {"conversation", "one-shot"}

    @dataclass(slots=True)
    class Turn:
        role: str
        content: str

    def __init__(
        self, 
        model: str, 
        *, 
        mode: str = "one-shot", 
        sys_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        if mode not in self._MODES:
            raise ValueError(mode)
        
        if model.startswith(("gpt", "o1", "o3", "o4")):
            kwargs = {
                "model": model,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
                "api_key": os.environ["OPENAI_API_KEY"],
                # Use the Responses API so `ask` can stream the reasoning summary; the
                # final text content is identical to the old Chat Completions path (verified).
                "use_responses_api": True,
                "output_version": "responses/v1",
            }
            if model.startswith(("gpt-5", "o1", "o3", "o4")) and cfg.llm.reasoning_effort:
                # Responses API takes reasoning as a dict; "summary": "auto" surfaces the
                # streamed reasoning summary (Chat Completions' reasoning_effort exposed none).
                kwargs["reasoning"] = {"effort": cfg.llm.reasoning_effort, "summary": "auto"}
            self._llm = ChatOpenAI(**kwargs)
        else:
            raise ValueError(f"Unsupported model: {model}")

        self._mode = mode
        self._log: list[BaseMessage] = []
        self._last_prompt: HumanMessage | None = None
        self._sys_prompt = sys_prompt
        self._verbose = verbose
        if sys_prompt: 
            assert isinstance(sys_prompt, str)
            sys_msg = SystemMessage(content=sys_prompt)
            self._log.append(sys_msg)


    def _extract_content(self, content: str | list[str | dict]) -> str:
        """Extract text from a langchain message's content, which is either a plain
        string or a list of content blocks ({"type": "text", ...} among others)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return "".join(text_parts)
        return str(content)

    def _complete(
        self,
        messages: list[BaseMessage],
        on_reasoning: Optional[Callable[[str], None]],
    ) -> AIMessage:
        """Run the model over `messages`. With `on_reasoning`, stream and forward each
        reasoning-summary delta to it; otherwise invoke (blocking). The accumulated
        reply's text content is identical either way (verified against the live API)."""
        if on_reasoning is None:
            reply = self._llm.invoke(messages)
            assert isinstance(reply, AIMessage)
            return reply
        acc = None
        for chunk in self._llm.stream(messages):
            for delta in _reasoning_summary_deltas(chunk):
                on_reasoning(delta)
            acc = chunk if acc is None else acc + chunk
        if acc is None:
            raise RuntimeError("LLM stream returned no chunks")
        assert isinstance(acc, AIMessage)
        return acc

    def ask(
        self,
        content: str,
        images: Optional[Iterable[Image.Image]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> str:
        if images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
            for img in images:
                assert isinstance(img, Image.Image)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            prompt = HumanMessage(content=parts)  # type: ignore[arg-type]
        else:
            prompt = HumanMessage(content=content)
        if self._verbose:
            head = "=" * 32 + " Human Message " + "=" * 32
            print(f"{head}\n\n{content}\n")

        self._last_prompt = prompt
        if self._mode == "conversation":
            self._log.append(prompt)
            reply = self._complete(self._log, on_reasoning)
        else:
            assert (
                len(self._log) == 0 or  # no sys prompt
                (len(self._log) == 1 and  # one sys prompt only
                isinstance(self._log[0], SystemMessage))
            )
            reply = self._complete(self._log + [prompt], on_reasoning)
        content = self._extract_content(reply.content)
        if self._mode == "conversation":
            # Store only the final text (drop the reasoning blocks) so the history
            # round-trips cleanly on the next conversation turn.
            self._log.append(AIMessage(content=content))
        if self._verbose:
            head = "=" * 32 + " AI Message " + "=" * 32
            print(f"{Fore.CYAN}{head}\n\n{content}{Fore.RESET}")
        return content


    def retry(self) -> str:
        if not self._last_prompt:
            raise RuntimeError("No message to retry. Call `ask` first.")
        if self._mode == "conversation":
            if not self._log or not isinstance(self._log[-1], AIMessage):
                raise RuntimeError("No response to retry.")
            self._log.pop()
            reply = self._complete(self._log, None)
            content = self._extract_content(reply.content)
            self._log.append(AIMessage(content=content))
            return content
        reply = self._complete(self._log + [self._last_prompt], None)
        return self._extract_content(reply.content)

    def history(self) -> Iterable[Turn]:
        for msg in self._log:
            if isinstance(msg, HumanMessage):
                yield self.Turn("user", self._extract_content(msg.content))
            elif isinstance(msg, AIMessage):
                yield self.Turn("ai", self._extract_content(msg.content))

    def clear(self) -> None:
        self._log.clear()
        if self._sys_prompt:
            self._log.append(SystemMessage(content=self._sys_prompt))
        self._last_prompt = None


# Example Usage
# llm = ChatSession('gpt=4o', mode="conversation", sys_prompt=prompts.constraints['sys'])
# prompt: str = prompts.constraints['coarse']
# template: Template = Template(prompt)
# prompt = template.substitute(
#     design="",
#     asset_list_str="",
# )
# response: str = llm.ask(prompt)