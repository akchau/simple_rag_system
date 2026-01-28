from src.controllers.base import BaseUseCase
from src.utils.prompt_manager import PromptManager


class RequestUseCase(BaseUseCase[str]):
    
    def execute(self):
        question = self._data
        engine = self._app.engine
        llm_client = self._app.llm_client
        context = engine.retrieve(question)
        prompt = PromptManager(question=question, context=context).result
        answer = llm_client.send_request(prompt)
        print(answer)