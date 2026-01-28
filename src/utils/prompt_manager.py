# @dataclass
# class IsNeedCreate:
#     need_create: bool
#     create_md: str | None


from enum import Enum
from typing import Type


class PromptTypes(str, Enum):
    LAW = "law"
    GENERAL = "general"


class BasePromptManager:
    prompt_type: PromptTypes = None
    _RAG_PROMPT_TEMPLATE = ""
    
    def __init__(self, question: str, context=None):
        self._prompt = f"Заметки: {context}\n" if context else "" + self._RAG_PROMPT_TEMPLATE.format(question=question)

    @property
    def result(self) -> str:
        """ Промпт для запроса к LLM """
        return self._prompt


class LawPromptManager(BasePromptManager):
    
    prompt_type = PromptTypes.LAW
    
    _RAG_PROMPT_TEMPLATE = """
        ### РОЛЬ: Ты — педантичный российский юрист. Твоя база — ТОЛЬКО предоставленный текст.\n"
        "### ПРАВИЛА:\n"
        "1. ЦИТИРУЙ ДОСЛОВНО. Не меняй юридические формулировки.\n"
        "2. Выделяй **ЖИРНЫМ** ключевые требования (например, **участие специалиста**).\n"
        "3. Пиши кратко и по существу. Если ответа нет, пиши 'ИНФОРМАЦИЯ НЕ НАЙДЕНА'.\n\n"
        "### ВОПРОС: {question}\n\n"
        "### ЮРИДИЧЕСКИЙ ОТВЕТ:"
    """


class PromptManager(BasePromptManager):
    
    prompt_type = PromptTypes.GENERAL
    
    
    _RAG_PROMPT_TEMPLATE = """
    Ты — высококлассный помощник ассистент, отвечающий ТОЛЬКО на основе предоставленных заметок.
    Если информации не достаточно — подготовь ее сам. Ответ подготовь в формате md"

    Вопрос пользователя: {question}

    Формат ответа будет указан ниже в кавычках """ """, внутри кавычек будут вставлены пояснения в (). Они относятся ко всему блоку до (). 
    Ты должен учесть их и не включать в ответ в чистом виде. Разделители - /// необходимо сохранить для парсинга твоего ответа:
    \"""
    
    (Если информация найдена, и главное она относится к вопросу и отвечает на него то блок ниже)
    ПО вашему запросу найдена следующая информация:
    <Тут указана найденная информация из заметок>
    Информация найдена в заметках: <Тут перечисли в каких заметках найдена информация, названия заметок без расширения .md в [[]], например pandas.md запиши как [[pandas]]. Должно быть не более 3-4 заметок>
    
    (Ты должен понять эта информация полностью отвечает на мой запрос и достаточна, если нет, то добавь блок ниже в ///)
    ///
    # {question}
    <Тут указана дополнительная информация из LLM>
    ///
    """

    # def parse(self, answer: str) -> IsNeedCreate:
    #     """
    #     Разбор ответа и поиск новых данных

    #     Args:
    #         answer (str): Ответ LLM

    #     Returns:
    #         IsNeedCreate: Данные ответа
    #     """
    #     note_data = answer.split("///")
    #     is_need_create = len(note_data) == 2
    #     return IsNeedCreate(
    #         need_create=is_need_create,
    #         create_md=note_data[-1] if is_need_create else None
    #     )

class PromptFactory:
    
    prompts_classes: list[Type[BasePromptManager]] = [
        LawPromptManager,
        PromptManager
    ]

    @classmethod
    def get_prompt_class_by_type(cls, target_type: PromptTypes) -> Type[BasePromptManager]:
        for prompt_class in cls.prompts_classes:
            if prompt_class.prompt_type == target_type:
                return prompt_class
        raise RuntimeError(f"Не найден промпт менеджер типа {target_type}")