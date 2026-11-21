"""Small shared helpers for the analysis / collection plot and animate paths.

``build_exclude_list`` assembles the no-data-plus-extra exclusion list that
``read_array`` / ``render_array`` accept; ``require_one_label_per_frame`` is the
shared "one label per frame" guard for the animate entry points. Both were
inlined in several copies across ``engines.analysis`` and ``collection``.
"""

from __future__ import annotations

from typing import Any


def build_exclude_list(no_data: Any, extra: Any = None) -> list:
    """Assemble the exclusion list: the no-data sentinel (when present) plus an optional extra.

    Args:
        no_data: The no-data sentinel to exclude. ``None`` excludes nothing for
            it (so a path that has no physical sentinel contributes no entry).
        extra: An optional extra value to also exclude -- the caller's
            ``exclude_value``. ``None`` adds nothing.

    Returns:
        list: ``[no_data]`` when ``no_data`` is not ``None``, with ``extra``
            appended when it is not ``None`` -- so an empty list only when both
            are ``None``.
    """
    values = [] if no_data is None else [no_data]
    if extra is not None:
        values.append(extra)
    return values


def require_one_label_per_frame(label_count: int, frame_count: int) -> None:
    """Raise when an animation's supplied label count does not match its frames.

    Args:
        label_count: Number of frame labels supplied.
        frame_count: Number of animation frames.

    Raises:
        ValueError: ``label_count != frame_count``.
    """
    if label_count != frame_count:
        raise ValueError(
            f"animate() got {label_count} frame label(s) for {frame_count} frame(s); "
            f"there must be one label per frame."
        )
