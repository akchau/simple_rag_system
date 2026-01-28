def strip_slash(v: str) -> str:
    """
    Утилита обрезает слэши вначале и конце строки.
    """
    value = v.strip()
    while value.startswith("/") or value.startswith(" "):
        value = value[1:]
    while value.endswith("/") or value.endswith(" "):
        value = value[:-1]
    return value