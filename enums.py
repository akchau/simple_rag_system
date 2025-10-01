from enum import Enum


class ModelsEnum(str, Enum):
    """ Доступные модели для Mistral API SDK """
    MEDIUM = "mistral-medium"
    TINY = "mistral-tiny"
    SMALL = "mistral-small"
    LARGE = "mistral-large-latest"