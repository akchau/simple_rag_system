# @dataclass
# class IsNeedCreate:
#     need_create: bool
#     create_md: str | None


class PromptManager:
    
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
    
    def __init__(self, question: str, context=None):
        self._prompt = f"Заметки: {context}\n" if context else "" + self._RAG_PROMPT_TEMPLATE.format(question=question)

    @property
    def result(self) -> str:
        """ Промпт для запроса к LLM """
        return self._prompt

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