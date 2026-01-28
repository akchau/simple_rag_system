from typing import Type
from src.controllers.base import BaseUseCase
from src.controllers.use_cases.create_chunks import LoadIndex
from src.controllers.use_cases.get_request import RequestUseCase
from src.init.init_app import AppContainer


class Controller:
    
    def __init__(self, app: AppContainer):
        self.app = app
    
    def _execute(self, use_case: Type[BaseUseCase], data=None):
        use_case(app=self.app, data=data).execute()
    
    def startup(self):
        self._execute(
            use_case=LoadIndex
        )

    def get_answer(self, question: str):
        self._execute(
            use_case=RequestUseCase,
            data=question
        )