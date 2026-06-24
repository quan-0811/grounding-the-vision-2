
import os
import re
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


def extract_int_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    value = str(value)
    base = os.path.splitext(os.path.basename(value))[0]
    if base.isdigit():
        return int(base)
    matches = re.findall(r"\d+", base)
    if matches:
        return int(matches[-1])
    raise ValueError(f"Cannot extract integer image id from value: {value}")


def get_image_id(record: Dict[str, Any]) -> int:
    for key in ["image_id", "coco_image_id", "id", "image", "image_path", "file_name"]:
        if key in record:
            return extract_int_id(record[key])
    raise KeyError(f"Cannot find image id field in record keys: {record.keys()}")