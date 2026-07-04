from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface (Abstract Base Class) for Large Language Model generation.

    All text generation engines (e.g. Ollama, OpenAI, Claude, Gemini) must implement
    this interface to keep services decoupled from specific SDK invocation structures.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text response using an LLM.

        Args:
            prompt (str): User prompt/instruction.
            system_prompt (str | None): Optional system prompt to instruct model behavior.
            temperature (float): Generation temperature. Defaults to 0.2.

        Returns:
            str: Generated text content response.
        """
        pass
