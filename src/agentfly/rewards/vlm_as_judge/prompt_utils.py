from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DEFAULT_VLM_PROMPT_TEMPLATE = """You are given a set of visual verification questions and a description of objects and motion observed in an input medium (e.g., image or video).

Your task is to **evaluate each question** based on whether it is **correctly reflected in the visual content**, considering visual cues, shape changes from viewpoint, and possible symbolic representations.


---

 **Visual Reasoning Guidelines**:

1. **Perspective Awareness**:  
   Objects may appear different based on viewpoint. For example:
   - A **cylinder** may look like a **circle (top view)** or a **rectangle/square (side view)**.
   - A **circular path** may appear as a **wave-like curve or straight line** in 2D projection.

2. **Symbolic Representations**:  
   Common simplifications may be used. You should **reasonably infer** their meaning:
   - A series of **dots or circles** may represent **foam markers** or control points.
   - A **rectangle** may represent a **container** (e.g., cylindrical viewed from the side).
   - A **line** may represent a **rubber mat** or constraint boundary.
   - The object and track specifics might do not match directly, if the motion can be interpreted correctly, it is still true.
   - It might use color to represent different objects, such as a green line to represent the flat surface is covered with a felt-like material.
   - The rotation of the object might cannot be judged from the video, but the motion can be interpreted correctly, it is still true.

3. **Container Boundaries**:
   - If **no container is drawn**, you may assume the **video frame itself is the container boundary**.
   - If a **container is visible**, treat it as **transparent** if inner content is visible.
   - If the object is not visible, you should not assume it is in the container.

4. **Focus on Shape & Position**, **not material**:
   - Ignore assumptions about object **material**, **color**, or **texture**.
   - Base your decisions entirely on **observable geometry** (e.g., shape, layout, structure) and **motion** (e.g., direction, trajectory).
   - Use visible movement and positioning to judge truthfulness — even if the object type is unknown.
   - If the described motion is **sliding down a slope**, but the video shows an **upward movement**, the result should be `"False"` — regardless of material or appearance.
   - Make geometric and motion-based reasoning the core of your judgment, even when objects are **partially occluded**.

5. **Occlusion Handling**:
   - If an object is **partially blocked**, assess based on surrounding evidence whether its state or motion can still be inferred.

6. **Avoid excessive uncertainty**:
   - If there is enough visual context and logical structure, make a **confident judgment**.
   - Use "Not sure" only when the evidence is **truly insufficient or ambiguous**.

---

 **Input**:
- Questions: {all_questions}
- Object and motion description: {summarize}

---

 **For each question**, return:
- `"index"`: the question index
- `"question"`: the full question text
- `"analysis"`: your reasoning process and visual inference
- `"result"`: one of `"True"`, `"False"`, or `"Not sure"`
- `"confidence_score"`: an integer from 1 (very uncertain) to 5 (very certain)

---

**Output Format**:
Return a JSON list like this:
[
    {{
        "index": "1",
        "question": "The ball rolls along the circular path.",
        "analysis": "The object follows a closed curve consistent with a circular path from the top view.",
        "result": "True",
        "confidence_score": "5"
    }},
    ...
]
"""


def _format_keywords(
    text: str,
    keywords: Optional[List[str]] = None,
    style: str = "bold",
    case_sensitive: bool = False,
) -> str:
    if not keywords:
        return text

    unique_keywords = [
        k for k in sorted(set(keywords), key=lambda s: len(s or ""), reverse=True) if k
    ]
    if not unique_keywords:
        return text

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile("|".join(re.escape(k) for k in unique_keywords), flags)

    def repl(match: re.Match[str]) -> str:
        start, end = match.start(), match.end()
        if style == "bold":
            prev2 = text[max(0, start - 2):start]
            next2 = text[end:end + 2]
            if prev2 == "**" and next2 == "**":
                return match.group(0)
            return f"**{match.group(0)}**"
        if style == "bracket":
            return f"[{match.group(0)}]"
        if style == "caps":
            return match.group(0).upper()
        return f"**{match.group(0)}**"

    return pattern.sub(repl, text)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def create_vlm_prompt_from_template(
    prompt_template: str,
    variables: Optional[Dict[str, Any]] = None,
    keywords: Optional[List[str]] = None,
    style: str = "bold",
    case_sensitive: bool = False,
) -> str:
    text = prompt_template
    if variables:
        try:
            text = prompt_template.format_map(_SafeDict(variables))
        except Exception:
            text = prompt_template
    return _format_keywords(
        text,
        keywords=keywords,
        style=style,
        case_sensitive=case_sensitive,
    )


def create_vlm_prompt_custom(
    prompt: str,
    keywords: Optional[List[str]] = None,
    style: str = "bold",
    case_sensitive: bool = False,
) -> str:
    return _format_keywords(
        prompt,
        keywords=keywords,
        style=style,
        case_sensitive=case_sensitive,
    )


def create_vlm_prompt(summarize: str, all_questions: str) -> str:
    return create_vlm_prompt_from_template(
        DEFAULT_VLM_PROMPT_TEMPLATE,
        variables={"summarize": summarize, "all_questions": all_questions},
    )
