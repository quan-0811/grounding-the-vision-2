
from typing import Any, Dict, List, Optional

def maybe_int(value: Any) -> Any:
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip()

        if value.isdigit():
            return int(value)

    return value


def first_existing(row: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)

        if value is not None:
            return value

    return None