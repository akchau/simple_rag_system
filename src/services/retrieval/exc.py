class BaseRetrievalException(Exception):
    pass


class DocsNotExist(BaseRetrievalException):

    def __init__(self, message: str = "Не загружено не одного файла для индексации"):
        super().__init__(message)