from src.controllers.base import BaseUseCase
from src.types_.base_types import UserQuestion


class RequestUseCase(BaseUseCase[UserQuestion]):
    
    def execute(self):
        question = self._data
        engine = self._app.engine
        llm_client = self._app.llm_client
        prompt_manager_class = self._app.prompt_manager_class
        context = engine.retrieve(question)
        prompt = prompt_manager_class(question=question, context=context).result
        answer = llm_client.send_request(prompt)
        print(answer)