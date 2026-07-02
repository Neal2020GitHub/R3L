# Copyright 2023 Allen Institute for Artificial Intelligence
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from AllenAI Holodeck (https://github.com/allenai/Holodeck;
# Yang et al., CVPR 2024) and Holodeck 2.0 (Bian et al., 2025,
# arXiv:2508.05899). Adapted for the R3L pipeline. See the LICENSE file
# in this directory for the full Apache 2.0 terms.
#
# Modifications (where applicable) are Copyright (c) 2026 Yuqi Wang and
# Zhifeng Gu and licensed under the MIT License (see the repository root
# LICENSE).

import os
from typing import Optional, Sequence, Any, Union

import openai

from objathor.annotation.annotation_utils import compute_llm_cost
from objathor.utils.gpt_utils import (
    DEFAULT_MAX_ATTEMPTS,
    access_gpt_with_retries,
)
from objathor.utils.queries import Message, ComposedMessage, Text

DEFAULT_PROMPT = (
    "You are a helpful assistant who expertly assists users with the questions."
)


class OpenAIWithTracking:
    def __init__(
        self,
        model: str,
        openai_api_key: Optional[str] = None,
        max_tokens=16000,
        temperature=0.7,
        verbose: bool = True,
        **defaults: Any,
    ):
        if openai_api_key is None:
            assert (
                "OPENAI_API_KEY" in os.environ
            ), "Please set the OPENAI_API_KEY environment variable."
            openai_api_key = os.environ["OPENAI_API_KEY"]

        self.openai_api_key = openai_api_key

        self.defaults = {
            "model": model,
            # "max_tokens": max_tokens,  # TODO: change this to fit gpt-5
            "max_completion_tokens": max_tokens,
            # "temperature": temperature,  # TODO: change this to fit gpt-5
            **defaults,
        }
        self.verbose = verbose
        self.queries = []

    def reset(self):
        self.queries = []

    def __call__(self, messages: Union[str, Sequence[Message]], **kwargs: Any) -> str:
        return self.get_answer(messages=messages, **kwargs)

    def get_answer(
        self,
        messages: Union[str, Sequence[Message]],
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        verbose: Optional[bool] = None,
        **chat_completion_cfg: Any,
    ) -> str:
        if verbose is None:
            verbose = self.verbose

        if isinstance(messages, str):
            messages = [
                Text(content=DEFAULT_PROMPT, role="system"),
                Text(content=messages, role="user"),
            ]

        def message_to_content(msg):
            return msg.gpt() if isinstance(msg, ComposedMessage) else [msg.gpt()]

        messages = [  # pyright: ignore[reportAssignmentType]  # reshape Text[] -> provider payload dict[]; element type intentionally changes
            dict(role=msg.role, content=message_to_content(msg)) for msg in messages
        ]

        def chat_completion_create() -> str:
            all_kwargs = dict(
                **self.defaults,
            )
            all_kwargs.update(chat_completion_cfg)
            all_kwargs["messages"] = messages

            completion = openai.OpenAI(
                api_key=self.openai_api_key
            ).chat.completions.create(**all_kwargs)
            res = completion.choices[0].message.content

            pt = completion.usage.prompt_tokens
            ct = completion.usage.completion_tokens
            try:
                cost = compute_llm_cost(
                    input_tokens=pt, output_tokens=ct, model=all_kwargs["model"]
                )
            except:  # TODO: gpt-5
                cost = 0.0

            self.queries.append(
                {
                    "messages": messages,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cost": cost,
                    "chat_kwargs": all_kwargs,
                }
            )
            if verbose:
                pt = completion.usage.prompt_tokens
                ct = completion.usage.completion_tokens
                print(
                    f"Prompt tokens: {pt}."
                    f" Completion tokens: {ct}."
                    f" Approx cost: ${cost:.2g}."
                )

            return res

        return access_gpt_with_retries(
            func=chat_completion_create, max_attempts=max_attempts
        )
