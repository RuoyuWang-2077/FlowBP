"""
UnifiedReward scorer via OpenAI-compatible HTTP API (e.g., sglang).

This scorer supports remote serving. Configure endpoint with either:
  - FLOWBP_UNIFIEDREWARD_BASE_URL (e.g., http://127.0.0.1:17140/v1)
or:
  - FLOWBP_UNIFIEDREWARD_HOST + FLOWBP_UNIFIEDREWARD_PORT

Optional env vars:
  - FLOWBP_UNIFIEDREWARD_BACKEND (openai|local, default: openai)
  - FLOWBP_UNIFIEDREWARD_MODEL_NAME (default: UnifiedReward-7b-v1.5)
  - FLOWBP_UNIFIEDREWARD_API_KEY (default: EMPTY)
  - FLOWBP_UNIFIEDREWARD_TIMEOUT (seconds, default: 120)
  - FLOWBP_UNIFIEDREWARD_LOCAL_MODEL_PATH (for backend=local)
"""

from __future__ import annotations

import base64
import copy
import os
import re
from io import BytesIO

import numpy as np
import torch
from PIL import Image


_SCORE_PATTERN = re.compile(r"Final Score:\s*([1-5](?:\.\d+)?)", re.IGNORECASE)


def _import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "UnifiedReward scorer requires the `openai` package. "
            "Install it via `pip install openai`."
        ) from e
    return OpenAI


def _resolve_base_url() -> str:
    explicit = os.environ.get("FLOWBP_UNIFIEDREWARD_BASE_URL", "").strip()
    if explicit:
        explicit = explicit.rstrip("/")
        return explicit if explicit.endswith("/v1") else f"{explicit}/v1"

    host = os.environ.get("FLOWBP_UNIFIEDREWARD_HOST", "127.0.0.1").strip()
    port = os.environ.get("FLOWBP_UNIFIEDREWARD_PORT", "17140").strip()
    return f"http://{host}:{port}/v1"


def _to_pil_list(images) -> list[Image.Image]:
    if isinstance(images, list) and (len(images) == 0 or isinstance(images[0], Image.Image)):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    if isinstance(images, np.ndarray):
        return [Image.fromarray(img) for img in images]
    raise TypeError(f"Unsupported image type for UnifiedReward scorer: {type(images)}")


def _pil_image_to_data_url(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_question(prompt: str) -> str:
    return (
        "<image>\n"
        "You are given a text caption and a generated image based on that caption. "
        "Your task is to evaluate this image based on two key criteria:\n"
        "1. Alignment with the Caption: Assess how well this image aligns with the provided "
        "caption. Consider the accuracy of depicted objects, their relationships, and "
        "attributes as described in the caption.\n"
        "2. Overall Image Quality: Examine the visual quality of this image, including clarity, "
        "detail preservation, color accuracy, and overall aesthetic appeal.\n"
        "Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
        f"Your task is provided as follows:\nText Caption: [{prompt}]"
    )


def _build_alignment_question(prompt: str) -> str:
    return (
        "<image>\n"
        "You are given a text caption and a generated image based on that caption. "
        "Evaluate only the image-text alignment: how accurately the image depicts "
        "the objects, attributes, counts, spatial relationships, and actions in the caption. "
        "Ignore general image quality unless it prevents judging alignment. "
        "Assign an alignment score from 1 to 5 after 'Final Score:'.\n"
        f"Text Caption: [{prompt}]"
    )


def _build_image_quality_question(prompt: str) -> str:
    return (
        "<image>\n"
        "You are given a text caption and a generated image based on that caption. "
        "Evaluate only the overall image quality: visual fidelity, aesthetics, clarity, "
        "detail preservation, artifacts, composition, and color. Do not penalize caption "
        "mismatch except where it directly affects perceived image quality. "
        "Assign an image-quality score from 1 to 5 after 'Final Score:'.\n"
        f"Text Caption: [{prompt}]"
    )


_QUESTION_BUILDERS = {
    "overall": _build_question,
    "align": _build_alignment_question,
    "iq": _build_image_quality_question,
}


def _parse_score_to_unit_interval(content) -> float:
    text = content if isinstance(content, str) else str(content)
    match = _SCORE_PATTERN.search(text)
    if not match:
        return -10.0
    return float(match.group(1)) / 5.0


class UnifiedRewardScorer(torch.nn.Module):
    def __init__(self, dtype, device, criterion: str = "overall"):
        super().__init__()
        del dtype  # HTTP serving controls precision remotely.

        self.device = torch.device(device)
        self.criterion = str(criterion).lower()
        if self.criterion not in _QUESTION_BUILDERS:
            raise ValueError(
                f"Unsupported UnifiedReward criterion {criterion!r}; "
                f"expected one of {sorted(_QUESTION_BUILDERS)}."
            )
        self.backend = os.environ.get(
            "FLOWBP_UNIFIEDREWARD_BACKEND", "openai"
        ).strip().lower()
        if self.backend not in {"openai", "local"}:
            raise ValueError(
                f"Unsupported FLOWBP_UNIFIEDREWARD_BACKEND={self.backend!r}; "
                "expected 'openai' or 'local'."
            )
        self.base_url = _resolve_base_url()
        self.model_name = os.environ.get(
            "FLOWBP_UNIFIEDREWARD_MODEL_NAME", "CodeGoat24/UnifiedReward-7b-v1.5"
        )
        self.api_key = os.environ.get("FLOWBP_UNIFIEDREWARD_API_KEY", "EMPTY")
        self.timeout = float(os.environ.get("FLOWBP_UNIFIEDREWARD_TIMEOUT", "120"))

        self._client = None
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        if self.backend == "local":
            self._init_local_backend()
        else:
            OpenAI = _import_openai_client()
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )

        self.eval()

    def _init_local_backend(self) -> None:
        try:
            from llava.model.builder import load_pretrained_model
        except ImportError as e:
            raise ImportError(
                "Local UnifiedReward backend requires LLaVA-NeXT. "
                "Install it in an isolated env via "
                "`pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`."
            ) from e

        model_path = os.environ.get(
            "FLOWBP_UNIFIEDREWARD_LOCAL_MODEL_PATH",
            "UnifiedReward-7b-v1.5",
        )
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            model_path,
            None,
            "llava_qwen",
            device_map="auto",
            overwrite_config={"text_config": None},
        )
        self._model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
        return self

    @torch.no_grad()
    def __call__(self, images, prompts):
        if len(prompts) == 0:
            return torch.empty(0, dtype=torch.float32)

        pil_images = _to_pil_list(images)
        if len(pil_images) != len(prompts):
            raise ValueError(
                f"prompts/images length mismatch: {len(prompts)} vs {len(pil_images)}"
            )

        if self.backend == "local":
            return self._call_local(pil_images, prompts)
        return self._call_openai(pil_images, prompts)

    @torch.no_grad()
    def _call_openai(self, pil_images, prompts):
        scores: list[float] = []
        build_question = _QUESTION_BUILDERS[self.criterion]
        for prompt, image in zip(prompts, pil_images):
            question = build_question(str(prompt))
            image_ref = _pil_image_to_data_url(image.resize((512, 512)))
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_ref}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            scores.append(_parse_score_to_unit_interval(content))

        return torch.tensor(scores, dtype=torch.float32)

    @torch.no_grad()
    def _call_local(self, pil_images, prompts):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._image_processor is not None

        scores: list[float] = []
        build_question = _QUESTION_BUILDERS[self.criterion]
        for prompt, image in zip(prompts, pil_images):
            question = build_question(str(prompt))
            if question.startswith("<image>\n"):
                question = question[len("<image>\n") :]
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

            image = image.convert("RGB").resize((512, 512))
            image_tensor = process_images([image], self._image_processor, self._model.config)
            image_tensor = [
                tensor.to(dtype=torch.float16, device=self.device)
                for tensor in image_tensor
            ]
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(
                conv.get_prompt(),
                self._tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
            output = self._model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
            )
            content = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            scores.append(_parse_score_to_unit_interval(content))

        return torch.tensor(scores, dtype=torch.float32)
