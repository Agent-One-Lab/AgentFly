import dataclasses
from abc import ABC, abstractmethod
from typing import Callable


@dataclasses.dataclass
class AssistantPolicy:
    content_processor: Callable[[str], str] = None


class AssistantContentProcessor(ABC):
    @abstractmethod
    def __call__(self, assistant_message: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def jinja(self) -> str:
        raise NotImplementedError


class Qwen25AssistantContentProcessor(AssistantContentProcessor):
    def __call__(self, content: str) -> str:
        if content is None or content == "":
            return ""
        else:
            return "\n" + content

    def jinja(self) -> str:
        return """{% if content is none or content == "" %}{% else %}\n\n{{ content }}{% endif %}"""
