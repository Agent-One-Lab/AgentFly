from abc import ABC, abstractmethod
import dataclasses
import datetime
from typing import Callable

@dataclasses.dataclass
class SystemPolicy:
    use_system: bool = True
    use_system_without_system_message: bool = True
    content_processor: Callable[[str], str] = None


class SystemContentProcessor(ABC):
    @abstractmethod
    def __call__(self, system_message: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def jinja(self) -> str:
        raise NotImplementedError

class Llama32DateProcessor(SystemContentProcessor):
    """
    A system content processor that adds date information to system messages.
    
    In Python mode, it dynamically computes the current date.
    In Jinja mode, it provides a template with placeholders that can be processed.
    
    Usage in Jinja templates:
        - The template includes '__CURRENT_DATE__' placeholder
        - Replace '__CURRENT_DATE__' with the actual formatted date during processing
        - Format should be 'dd MMM yyyy' (e.g., '15 Dec 2024')
        - No external context variables required
    """
    def __call__(self, system_message: str) -> str:
        return f"Cutting Knowledge Date: December 2023\nToday Date: {datetime.datetime.now().strftime('%d %b %Y')}\n\n{system_message}"
    
    def jinja(self) -> str:
        # For Jinja templates used by external systems (like vLLM), we need a self-contained approach
        # Since external systems can't provide context variables, we use a placeholder approach
        # The external system should replace __CURRENT_DATE__ with the actual date
        return """Cutting Knowledge Date: December 2023
Today Date: __CURRENT_DATE__

{{ system_message }}"""
        