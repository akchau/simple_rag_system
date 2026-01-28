from src.controllers.base import BaseUseCase


class LoadIndex(BaseUseCase[None]):

    def execute(self):
        engine = self._app.engine
        engine.load_or_build_index()