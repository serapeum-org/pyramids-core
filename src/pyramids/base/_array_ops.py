"""Element-wise array operations shared by raster and mesh pipelines.

Every function takes a plain numpy array (plus metadata like a no-data
sentinel) and returns a plain numpy array.  No geometry, no GDAL, no
mesh — the caller (a Dataset engine or a MeshFacades method) owns the
spatial context.

Same design contract as :mod:`pyramids.base._reductions`: pure numpy
kernels that both sides of the class tree can import safely.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

KEEP_NO_DATA = object()
"""``astype_values``' default for ``no_data_value``: keep the source's own sentinel."""


def nodata_mask(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask — ``True`` where *values* is missing.

    For floating types NaN positions are always missing regardless of
    the *nodata* sentinel.  For integer types only exact matches of
    *nodata* count.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None`` for none.

    Returns:
        Boolean array, same shape as *values*.

    Examples:
        - Float array — NaN is always missing:

            >>> import numpy as np
            >>> nodata_mask(np.array([1.0, np.nan, 3.0]))
            array([False,  True, False])

        - Integer array with a sentinel value:

            >>> nodata_mask(np.array([1, -9999, 3]), nodata=-9999)
            array([False,  True, False])

    See Also:
        isnull: Identical behaviour, provided as a named alias.
        notnull: Inverted mask (``True`` where valid).
    """
    is_float = np.issubdtype(values.dtype, np.floating)
    if is_float:
        mask = np.isnan(values)
    else:
        mask = np.zeros(values.shape, dtype=bool)
    if nodata is not None and not (is_float and np.isnan(nodata)):
        mask = mask | (values == nodata)
    return mask


def regap(values: np.ndarray, domain: np.ndarray, sentinel: Any) -> np.ndarray:
    """``values`` with every cell outside ``domain`` set back to its sentinel.

    The kernels that transform cell values leave the gaps out and put them back here, so a
    gap is never clipped, rounded or cast into a measurement. Writing in place, rather than
    through ``np.where``, keeps the array's type: a ``numpy.float64`` sentinel would
    otherwise promote a ``float32`` array under NEP 50.

    ``sentinel`` is a scalar applied to the whole array, or a list/tuple carrying one per
    plane of the LEADING AXIS. The split keys off that **sequence-ness**, not off
    ``values.ndim`` -- a 1-D mesh array has no band axis, and the ndim rule raised
    ``TypeError: 'numpy.float64' object does not support item assignment`` on one, while a
    scalar sentinel on a 3-D raster stack is written across every plane either way.

    The result is **always a fresh array**, never ``values`` itself -- a later in-place write
    through one must not reach the caller's buffer (the mesh facade aliases it otherwise).

    Args:
        values: The transformed cells.
        domain: True where a cell holds data, shaped like ``values``.
        sentinel: The value that marks a gap -- one scalar for the whole array, or a
            list/tuple holding each leading-axis plane's own. ``None``, or a list entry of
            ``None``, leaves that plane's gaps as the operation left them.

    Returns:
        np.ndarray: A fresh copy of the cells, gaps re-marked.

    Examples:
        >>> import numpy as np
        >>> regap(np.array([8.0, 8.0]), np.array([True, False]), -9999.0).tolist()
        [8.0, -9999.0]
    """
    marks = list(sentinel) if isinstance(sentinel, (list, tuple)) else [sentinel]
    out = np.array(values, copy=True)  # never the caller's object (mesh aliasing)
    if not domain.all() and any(one is not None for one in marks):
        if isinstance(sentinel, (list, tuple)):
            for index in range(out.shape[0]):
                mark = _mark_for(marks, index)
                if mark is not None:
                    out[index][~domain[index]] = mark
        elif marks[0] is not None:
            out[~domain] = marks[0]
    return out


def _mark_for(marks: list, index: int) -> Any:
    """The sentinel for one plane of a stack.

    Args:
        marks: The sentinels, one per plane -- or fewer, when the caller passed one for a
            stack, which every plane then shares.
        index: The plane's index.

    Returns:
        Any: That plane's sentinel, or the first when the list is shorter than the stack.
    """
    return marks[index] if index < len(marks) else marks[0]


def _same_gap(one: Any, other: Any) -> bool:
    """Whether two planes declare the same gap marker.

    Compared by value, not by ``repr``: GDAL hands a sentinel back as ``numpy.float64`` while
    a caller passes a Python ``float``, and ``-9999.0`` is the same declaration either way.
    Two NaN markers agree as well, though NaN equals nothing.

    Args:
        one: One plane's sentinel, or ``None``.
        other: Another plane's sentinel, or ``None``.

    Returns:
        bool: ``True`` when they declare the same thing.
    """
    if one is None or other is None:
        same = one is None and other is None
    else:
        left, right = float(one), float(other)
        same = left == right or (math.isnan(left) and math.isnan(right))
    return same


def declared_gap(sentinels: Sequence[Any]) -> Any:
    """What a result should declare as its gaps, given each plane's own sentinel.

    One value when every plane declares the same thing -- the ordinary single-plane and
    uniform-stack cases, where a scalar keeps the metadata simplest -- and the per-plane list
    otherwise. Collapsing a mixed stack onto the first plane's sentinel rewrote the others'
    declarations, and a plane holding another plane's sentinel as a real value then read as
    missing.

    Args:
        sentinels: One sentinel per plane, ``None`` where a plane declares none.

    Returns:
        Any: The scalar the planes agree on, or the list of per-plane sentinels.

    Examples:
        >>> declared_gap([-9999.0, -9999.0])
        -9999.0
    """
    marks = list(sentinels)
    agreed = all(_same_gap(one, marks[0]) for one in marks)
    return marks[0] if agreed and marks else marks


def holds(target: np.dtype, value: Any) -> bool:
    """Whether ``target`` can carry ``value`` as a sentinel at all.

    A float target takes any value inside its range: the sentinel is **snapped** to the type
    on the way in (``-9999.9`` into ``float32`` is declared and written as
    ``-9999.900390625``), and the cells it marks are found by the caller's own domain mask,
    so a value that merely loses precision still marks exactly the cells it should. What must
    not happen is a snapped sentinel landing on a cell that holds data -- ``1e-50`` into
    ``float32`` snaps to ``0.0`` -- and that is
    :func:`_refuse_a_sentinel_the_data_holds`'s question, asked of the cast cells rather than
    guessed from the type.

    An integer target is stricter, because no snapping keeps a gap a gap there: a fraction or
    an out-of-range value wraps into an ordinary number.

    Args:
        target: The target type.
        value: The candidate sentinel.

    Returns:
        bool: ``True`` for a float type and any value inside its range, NaN and the
        infinities included; for an integer type, only a whole number inside its range.

    Examples:
        >>> holds(np.dtype("float16"), -9999.0)
        True
        >>> holds(np.dtype("uint8"), -9999.0)
        False
    """
    number = float(value)
    if np.issubdtype(target, np.floating):
        fits = not np.isfinite(number) or abs(number) <= float(np.finfo(target).max)
    else:
        limits = np.iinfo(target)
        fits = (
            bool(np.isfinite(number))
            and number.is_integer()
            and limits.min <= number <= limits.max
        )
    return fits


def _gap_marker(
    mark: Any, target: np.dtype, domain: np.ndarray, band: int, bands: int
) -> Any:
    """What to write into one plane's gap cells before the cast fills the rest.

    Args:
        mark: The plane's declared sentinel, or ``None``.
        target: The type being cast to.
        domain: True where a cell holds data.
        band: The plane's index.
        bands: How many planes there are.

    Returns:
        Any: The value to fill that plane's gaps with.

    Raises:
        ValueError: The plane has gaps, declares no sentinel, and ``target`` is an integer
            type, which has no NaN to leave them as.
    """
    marker = mark
    if marker is None:
        plane = domain if bands == 1 or domain.ndim == 2 else domain[band]
        if not plane.all():
            if target.kind != "f":
                raise ValueError(
                    f"astype({target.name!r}) would leave {int((~plane).sum())} gap cells "
                    f"of band {band + 1} unmarked: it declares no no-data value and "
                    f"{target.name} has no NaN, so every gap would read as an ordinary "
                    f"number. Pass no_data_value= with one {target.name} holds, or fill "
                    f"the gaps first."
                )
            marker = np.nan
        else:
            marker = 0
    return marker


def _refuse_a_sentinel_the_data_holds(
    cast: np.ndarray,
    domain: np.ndarray,
    marks: Sequence[Any],
    target: np.dtype,
    bands: int,
) -> None:
    """Refuse a cast in which a cell holding data lands on its plane's sentinel.

    The mirror of "a gap stays a gap": data has to stay data. A sentinel a real cell already
    holds -- ``255`` after ``clip(0, 255)``, or ``-9999`` after truncating ``-9999.4`` --
    would read as missing from then on.

    Args:
        cast: The cast cells.
        domain: True where a cell holds data.
        marks: Each plane's sentinel, ``None`` where a plane declares none.
        target: The type cast to.
        bands: How many planes there are.

    Raises:
        ValueError: A plane's data holds its own sentinel.
    """
    for band in range(bands):
        mark = marks[band]
        if mark is None or not np.isfinite(float(mark)):
            continue
        plane = cast if bands == 1 else cast[band]
        kept = domain if bands == 1 else domain[band]
        collisions = int(np.count_nonzero(plane[kept] == target.type(mark)))
        if collisions:
            raise ValueError(
                f"astype({target.name!r}) would mark {collisions} cell(s) of band "
                f"{band + 1} that hold data as missing: they already hold {float(mark)} "
                f"once cast, and that is the value the result declares as its gap. Bound "
                f"or shift them first (clip), or pass no_data_value= with one the data "
                f"never takes."
            )


def fill_dtype(source: Any, fill: Any) -> np.dtype:
    """The dtype a fill needs: ``source``'s own, unless the fill will not fit it.

    The array keeps its own type when it can hold what is written into it, and widens only
    when it cannot: a fractional fill into an integer array, a NaN where no integer means
    "missing", or a finite magnitude no value of that width can represent. So ``uint8`` and
    ``255.0`` stays ``uint8`` while ``float32`` and ``1e300`` widens to ``float64`` -- the
    policy :func:`fillna` and :func:`where_values` apply before a masked write so a fill is
    never silently truncated or overflowed to ``inf``.

    Args:
        source: The array's dtype.
        fill: The value an unselected / filled cell will hold.

    Returns:
        numpy.dtype: The dtype the result needs.

    Examples:
        >>> import numpy as np
        >>> [str(fill_dtype(np.dtype("float32"), v)) for v in (-9999.0, 2.5, np.inf)]
        ['float32', 'float32', 'float32']
        >>> str(fill_dtype(np.dtype("float16"), 70000.0))
        'float64'
        >>> [str(fill_dtype(np.dtype("uint8"), v)) for v in (255.0, np.nan)]
        ['uint8', 'float64']
        >>> str(fill_dtype(np.dtype("int16"), 0.5))
        'float64'

    """
    source = np.dtype(source)
    filler = np.asarray(fill)
    chosen = np.result_type(source, filler)
    if np.issubdtype(source, np.floating):
        holds = bool(np.isnan(filler).all()) or bool(np.isinf(filler).all())
        if holds or bool(np.abs(filler) <= np.finfo(source).max):
            chosen = source
    elif np.issubdtype(source, np.integer) and not np.isnan(filler).any():
        limits = np.iinfo(source)
        value = float(filler)
        if value.is_integer() and limits.min <= value <= limits.max:
            chosen = source
    return chosen


def fillna(
    values: np.ndarray, fill_value, *, domain: np.ndarray, sentinel: Any = None
) -> tuple[np.ndarray, Any]:
    """Give every cell outside ``domain`` the value ``fill_value``.

    ``domain`` is **required** and is computed by the caller (the raster from stored counts
    before CF unpacking, the mesh from the declared sentinel), never guessed here from
    ``values == sentinel`` -- a packed raster's gaps are found on the raw array while the
    values are physical. The result widens via :func:`fill_dtype` BEFORE the fill goes in, so
    ``1e300`` into a ``float32`` array does not store ``inf`` behind a ``RuntimeWarning`` and
    ``2.5`` into ``int16`` does not truncate to ``2`` (D5).

    Args:
        values: The data array.
        fill_value: The replacement value for a gap.
        domain: True where a cell holds data; its gaps are the ``~domain`` cells filled.
        sentinel: The no-data value the result declares -- it now marks nothing, as a filled
            array's does. Returned unchanged.

    Returns:
        tuple: the filled array (fresh), and the sentinel the result declares.

    Examples:
        >>> import numpy as np
        >>> out, sentinel = fillna(
        ...     np.array([1.0, -9999.0]), 0.0, domain=np.array([True, False]), sentinel=-9999.0
        ... )
        >>> out.tolist()
        [1.0, 0.0]
    """
    array = np.asarray(values)
    if bool(np.all(domain)):
        # Nothing is missing, so no cell is written and the fill never lands. Keep the array's
        # own dtype rather than widening for a value that is not used -- otherwise a gap-free
        # ``int16`` filled with ``2.5`` returns ``float64`` despite changing nothing (review
        # M1). A fresh array still, so a caller cannot alias the source.
        return array.copy(), sentinel
    dtype = fill_dtype(array.dtype, fill_value)
    out = np.asarray(np.where(domain, array.astype(dtype, copy=False), fill_value))
    return out, sentinel


def isnull(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask of missing positions.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        Boolean array, same shape as *values*.
    """
    return nodata_mask(values, nodata)


def notnull(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask of non-missing positions.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        Boolean array, same shape as *values*.
    """
    return ~nodata_mask(values, nodata)


def clip_values(
    values: np.ndarray,
    lower: Any,
    upper: Any,
    *,
    domain: np.ndarray,
    sentinel: Any = None,
    operation: str = "clip",
) -> tuple[np.ndarray, Any]:
    """Bound the cells that hold data to ``[lower, upper]``; a gap stays a gap.

    The gaps (the ``~domain`` cells) are left out of the clipping and re-marked with
    ``sentinel`` afterwards. Clipping the stored array instead would lift a ``-9999.0`` gap
    to the lower bound and turn a missing cell into a measurement. The array keeps its own
    type unless a bound cannot fit it, widening then the way :func:`fill_dtype` does.

    Args:
        values: The data array (physical units -- a packed raster unpacks before this).
        lower: Minimum allowed value, or ``None`` for no lower bound.
        upper: Maximum allowed value, or ``None`` for no upper bound.
        domain: True where a cell holds data; its gaps are preserved.
        sentinel: The no-data value a gap is re-marked with, and the one the result declares.
        operation: The caller's own spelling, carried into the error messages so the raster
            says ``clip()`` and the mesh ``clip_values()``.

    Returns:
        tuple: the clipped array (fresh), and the sentinel the result declares.

    Raises:
        ValueError: Neither bound is given, a bound is NaN, or ``lower`` exceeds ``upper``.
        TypeError: A bound is not a real number.

    Examples:
        >>> import numpy as np
        >>> out, sentinel = clip_values(
        ...     np.array([5.0, -9999.0]), 8.0, 22.0,
        ...     domain=np.array([True, False]), sentinel=-9999.0,
        ... )
        >>> out.tolist(), sentinel
        ([8.0, -9999.0], -9999.0)
    """
    if lower is None and upper is None:
        raise ValueError(f"{operation}() needs at least one of min and max.")
    for bound in (lower, upper):
        if bound is not None and (
            not isinstance(bound, Real) or isinstance(bound, (bool, np.bool_))
        ):
            raise TypeError(f"{operation}() needs a number for a bound; got {bound!r}.")
        if bound is not None and np.isnan(float(bound)):
            raise ValueError(
                f"{operation}() cannot bound anything with NaN: every cell would come back "
                f"NaN, and a raster that declares another sentinel does not read those "
                f"as missing. Drop the bound, or use where() to mask."
            )
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(
            f"{operation}() needs min at or below max, got min={lower!r} above max={upper!r}."
        )
    array = np.asarray(values)
    # Cast back after clipping, not only before it: a bound that arrives as a numpy scalar --
    # every percentile, mean or quantile does -- promotes the result under NEP 50, so a
    # float32 array came back float64 and a uint8 array int64.
    dtype = array.dtype
    for bound in (lower, upper):
        if bound is not None:
            dtype = np.promote_types(dtype, fill_dtype(array.dtype, bound))
    clipped = np.clip(array.astype(dtype, copy=False), lower, upper).astype(
        dtype, copy=False
    )
    return regap(clipped, domain, sentinel), sentinel


def round_values(
    values: np.ndarray,
    decimals: int = 0,
    *,
    domain: np.ndarray,
    sentinel: Any = None,
    operation: str = "round",
) -> tuple[np.ndarray, Any]:
    """Round the cells that hold data to ``decimals`` places; a gap stays a gap.

    The gaps are left out and re-marked afterwards: a fractional sentinel such as ``-9999.5``
    rounds to ``-10000.0``, which no longer matches what was declared. An integer array
    rounded to a negative ``decimals`` can leave its own range (numpy wraps ``uint8`` 255 to
    4), so the result widens then rather than corrupting the data silently.

    Args:
        values: The data array.
        decimals: Number of decimal places.
        domain: True where a cell holds data; its gaps are preserved.
        sentinel: The no-data value a gap is re-marked with, and the one the result declares.
        operation: The caller's spelling, for the error message.

    Returns:
        tuple: the rounded array (fresh), and the sentinel the result declares.

    Raises:
        TypeError: ``decimals`` is not an integer.

    Examples:
        >>> import numpy as np
        >>> out, sentinel = round_values(
        ...     np.array([5.0, -9999.5]), 0, domain=np.array([True, False]), sentinel=-9999.5
        ... )
        >>> out.tolist()
        [5.0, -9999.5]
    """
    if isinstance(decimals, bool) or not isinstance(decimals, (int, np.integer)):
        raise TypeError(
            f"{operation}() needs an integer number of decimals, got {decimals!r}."
        )
    array = np.asarray(values)
    rounded = np.round(array, int(decimals))
    if array.dtype.kind in "iu" and int(decimals) < 0:
        wide = np.round(array.astype("float64"), int(decimals))
        limits = np.iinfo(array.dtype)
        inside = wide[domain] if domain.any() else wide
        fits = bool(((inside >= limits.min) & (inside <= limits.max)).all())
        rounded = wide.astype(array.dtype, copy=False) if fits else wide
    return regap(rounded, domain, sentinel), sentinel


def astype_values(
    values: np.ndarray,
    dtype: Any,
    *,
    domain: np.ndarray,
    sentinel: Any = None,
    no_data_value: Any = KEEP_NO_DATA,
    operation: str = "astype",
) -> tuple[np.ndarray, Any]:
    """Cast to a different dtype; a gap stays a gap, re-marked for the new type (D1).

    The cells that hold data are cast as numpy casts them. The gaps are not cast at all: they
    are re-marked with the target's sentinel, snapped to the new type, so a missing cell stays
    missing. A cast that would leave a gap unmarked (an integer target with no NaN and no
    declared sentinel), that cannot carry the sentinel, or in which a cell holding data lands
    on the sentinel once cast, is refused.

    ``sentinel`` carries the per-plane no-data values: a scalar for one plane, or a
    list/tuple for a stack. ``bands`` is read off its sequence-ness, so a 1-D mesh array (one
    plane) and a 3-D raster stack both work.

    Args:
        values: The data array.
        dtype: Target numpy dtype -- any real numeric type (signed/unsigned integer, float).
        domain: True where a cell holds data.
        sentinel: The source's declared sentinel(s): a scalar, a list/tuple per plane, or
            ``None``. Used only when ``no_data_value`` keeps the source's own.
        no_data_value: The sentinel the result declares. :data:`KEEP_NO_DATA` (default) keeps
            the source's own, snapped to the new type; a value overrides it for every plane;
            ``None`` declares none.
        operation: The caller's spelling, for the ``TypeError`` message.

    Returns:
        tuple: the cast array (fresh), and the sentinel the result declares.

    Raises:
        TypeError: ``dtype`` is not a real numeric type.
        ValueError: The sentinel is outside ``dtype``'s range or fractional in an integer
            target; a cell holding data lands on the sentinel once cast; or gaps would be left
            unmarked in an integer target.

    Examples:
        >>> import numpy as np
        >>> out, sentinel = astype_values(
        ...     np.array([5.0, -9999.0]), np.int32,
        ...     domain=np.array([True, False]), sentinel=-9999.0,
        ... )
        >>> out.tolist(), sentinel
        ([5, -9999], -9999.0)
    """
    target = np.dtype(dtype)
    if target.kind not in "iuf":
        raise TypeError(
            f"{operation}() needs a real numeric type GDAL can store — a signed or unsigned "
            f"integer, or a float — and {target.name} is not one."
        )
    array = np.asarray(values)
    bands = len(sentinel) if isinstance(sentinel, (list, tuple)) else 1
    if no_data_value is KEEP_NO_DATA:
        source = list(sentinel) if isinstance(sentinel, (list, tuple)) else [sentinel]
        marks = [source[i] if i < len(source) else None for i in range(bands)]
    else:
        marks = [no_data_value] * bands
    for mark in marks:
        if mark is not None and not holds(target, mark):
            raise ValueError(
                f"{operation}({target.name!r}) cannot mark a gap with {float(mark)}, which "
                f"{target.name} does not hold — cast, it would become an ordinary value "
                f"and every gap would read as data. Pass no_data_value= with one it "
                f"does hold."
            )
    # Snapped to the target before anything is written or declared, so the value the result
    # declares is exactly the value its gap cells hold. A float sentinel loses precision here
    # (`-9999.9` into float32 becomes `-9999.900390625`); what it must not do is land on a
    # cell holding data, which is checked after the cast.
    marks = [None if mark is None else float(target.type(mark)) for mark in marks]
    markers = [
        _gap_marker(mark, target, domain, band, bands)
        for band, mark in enumerate(marks)
    ]
    # Filled, never `np.empty`: the cells outside the domain are not cast, so an
    # uninitialised buffer would ship whatever the allocator held as data.
    cast = np.empty(array.shape, dtype=target)
    for band in range(bands):
        plane = cast if bands == 1 else cast[band]
        plane[...] = markers[band]
    cast[domain] = array[domain].astype(target)
    _refuse_a_sentinel_the_data_holds(cast, domain, marks, target, bands)
    return cast, declared_gap(marks)


def isin_values(
    values: np.ndarray, test_elements: Any, *, domain: np.ndarray
) -> np.ndarray:
    """Boolean mask where ``values`` is in ``test_elements``; a gap is in no set (D1).

    A gap (a ``~domain`` cell) answers ``False`` even when the set holds the gap's own
    sentinel -- a missing cell is missing, not that value. The result declares no sentinel,
    so it is a plain flag, not a ``(values, sentinel)`` pair; the raster stores it as
    ``uint8`` and the mesh as ``bool``, each its own band type.

    Args:
        values: The data array.
        test_elements: One value, or a sequence of them (a list, tuple, set or array).
        domain: True where a cell holds data.

    Returns:
        Boolean array, same shape as ``values``.

    Examples:
        >>> import numpy as np
        >>> isin_values(
        ...     np.array([5.0, -9999.0]), [-9999.0], domain=np.array([True, False])
        ... ).tolist()
        [False, False]
    """
    # A `set` survives `np.asarray` as a 0-d object array holding the set itself, so every
    # comparison is False and the flags come back all zero -- spelled as a list, it compares.
    wanted = (
        list(test_elements)
        if isinstance(test_elements, (set, frozenset))
        else test_elements
    )
    return np.asarray(np.isin(np.asarray(values), np.asarray(wanted)) & domain)


def array_equals(
    a: np.ndarray,
    b: np.ndarray,
    nodata_a=None,
    nodata_b=None,
) -> bool:
    """Value equality — same values at every non-missing position.

    Two arrays are *equal* when they have the same shape, the same set
    of missing positions, and identical values everywhere else.

    Args:
        a: First array.
        b: Second array.
        nodata_a: No-data sentinel for *a*.
        nodata_b: No-data sentinel for *b*.

    Returns:
        ``True`` when the arrays are value-equal.

    Examples:
        - NaN positions must match but are otherwise ignored:

            >>> import numpy as np
            >>> a = np.array([1.0, np.nan, 3.0])
            >>> b = np.array([1.0, np.nan, 3.0])
            >>> array_equals(a, b)
            True

        - Different shapes are never equal:

            >>> array_equals(np.array([1.0]), np.array([1.0, 2.0]))
            False

    See Also:
        array_identical: Strict comparison including dtype and NaN
            bit-patterns.
    """
    result = a.shape == b.shape
    if result:
        mask_a = nodata_mask(a, nodata_a)
        mask_b = nodata_mask(b, nodata_b)
        result = bool(np.array_equal(mask_a, mask_b))
        if result:
            valid = ~mask_a
            if np.any(valid):
                result = bool(np.array_equal(a[valid], b[valid]))
    return result


def array_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict equality — same dtype, same values including NaN.

    Args:
        a: First array.
        b: Second array.

    Returns:
        ``True`` when the arrays are bitwise-identical.

    Examples:
        - Arrays must share dtype to be identical:

            >>> import numpy as np
            >>> a = np.array([1, 2], dtype=np.int32)
            >>> b = np.array([1, 2], dtype=np.int64)
            >>> array_identical(a, b)
            False

    See Also:
        array_equals: Looser comparison that ignores nodata positions.
    """
    result = a.shape == b.shape and a.dtype == b.dtype
    if result:
        result = bool(np.array_equal(a, b, equal_nan=True))
    return result


def where_values(
    values: np.ndarray, cond: Any, other: Any = np.nan, nodata: Any = None
) -> np.ndarray:
    """Keep the cells ``cond`` selects and set the rest to ``other``.

    The array half of the ``where`` capability. A cell is kept only when ``cond`` is true
    **and** the cell actually holds a value: a ``nodata`` / NaN cell is never "selected",
    because there is nothing there to keep, so it goes to ``other`` whatever ``cond`` says.
    That keeps the result single-sentinel — one masking convention rather than ``other`` and a
    surviving fill value side by side.

    Args:
        values: The data array.
        cond: A boolean array broadcastable to ``values``, or anything numpy can treat as one.
        other: What an unselected cell holds. Defaults to NaN, which forces a float result.
        nodata: The declared no-data sentinel to treat as absent, or ``None``.

    Returns:
        np.ndarray: ``values`` where selected, ``other`` elsewhere. Float64 when ``other`` is
        NaN, since an integer array cannot carry it.

    Examples:
        - Keep the cells above a threshold, masking the rest:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._array_ops import where_values
            >>> values = np.array([1.0, 5.0, 9.0])
            >>> where_values(values, values > 4).tolist()
            [nan, 5.0, 9.0]

            ```
        - A no-data cell is never kept, even where the condition is true:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._array_ops import where_values
            >>> where_values(np.array([-999.0, 5.0]), np.array([True, True]), nodata=-999.0).tolist()
            [nan, 5.0]

            ```

    Raises:
        ValueError: ``cond`` cannot be broadcast against ``values`` — a condition shaped for a
            different variable entirely, rather than one axis of this one.

    """
    array = np.asarray(values)
    mask = np.asarray(cond, dtype=bool)
    if mask.shape != array.shape:
        try:
            # Broadcasting is what the docstring promises, and it is the natural thing to
            # write for a temporal variable: a `(n_face,)` condition means "these faces, every
            # step". Only a condition that cannot broadcast at all is an error.
            mask = np.broadcast_to(mask, array.shape)
        except ValueError as error:
            raise ValueError(
                f"where: the condition's shape {mask.shape} cannot be broadcast against the "
                f"values' {array.shape}. Build the condition from this variable (for example "
                f"`ds[name].data > 0`), or from one axis of it."
            ) from error
    keep = mask & ~nodata_mask(array, nodata)
    if other is None or (isinstance(other, float) and np.isnan(other)):
        result = array.astype("float64", copy=True)
        fill: Any = np.nan
    else:
        result = array.copy()
        if not _fits(other, result.dtype):
            result = result.astype("float64", copy=False)
        fill = other
    result[~keep] = fill
    return result


def _fits(value: Any, dtype: np.dtype) -> bool:
    """True when ``value`` can be stored in ``dtype`` without changing it.

    Anything numpy cannot read as a value at all — a ragged nested list raises ``ValueError``,
    an unreadable object ``TypeError`` — answers ``False`` rather than propagating: the
    question asked is "does this fit", and the caller then fails on the assignment itself,
    where numpy's own message names the real problem.

    Args:
        value: The candidate fill value.
        dtype: The dtype it would be stored in.

    Returns:
        bool: True when the value can be stored in ``dtype`` unchanged; False when it cannot,
        when it is an integer outside the dtype's range, or when numpy cannot read it as a
        value at all.
    """
    result = False
    if value is not None:
        # `result_type(None, float64)` is float64, but assigning None stores NaN rather than
        # the value itself, so None does not "fit" in the sense this function answers.
        # NEP 50 makes a *numpy* scalar strong, so `result_type(np.int64(5), int32)` is
        # int64 and a value that fits perfectly was reported as not fitting - which left the
        # range check below unreachable for those. Unwrapping to the Python equivalent puts it
        # back on the weak-promotion path the rest of this function reasons about.
        candidate = value.item() if isinstance(value, np.generic) else value
        try:
            # Compare the VALUE, not its dtype. `np.asarray(0).dtype` is int64, which never
            # casts safely into int32, so every Python int used to force a float64 promotion.
            # NEP 50 weak promotion answers what actually happens: `result_type(0, int32)` is
            # int32.
            promoted: Any = np.result_type(candidate, dtype)
            readable = True
        except (TypeError, ValueError):
            # A separate flag, not a `None` sentinel: `np.dtype("float64") == None` is True,
            # because numpy reads None as the default dtype - so comparing against None would
            # report that anything unreadable fits a float64 array.
            promoted, readable = None, False
        if readable and promoted == dtype:
            if dtype.kind in "iu" and isinstance(candidate, (int, np.integer)):
                # `result_type` reports the kind, not the range: it calls 2**40 an int32. Out
                # of range the value still has to promote, or the assignment would raise
                # OverflowError on a value float64 holds exactly.
                info = np.iinfo(dtype)
                result = bool(info.min <= int(candidate) <= info.max)
            else:
                result = True
    return result


def apply_values(
    values: np.ndarray, func: Any, *, domain: np.ndarray, sentinel: Any = None
) -> tuple[np.ndarray, Any]:
    """Apply ``func`` elementwise to ``values``, keeping the shape; a gap stays a gap.

    ``func`` is handed the whole array, so a numpy ufunc runs vectorised rather than per
    element; anything that preserves the shape is acceptable. A function that changes the
    shape is refused, because the result would no longer line up with the elements it
    describes. The gaps (the ``~domain`` cells) are re-marked with ``sentinel`` after
    ``func``, so ``-9999`` is not doubled to ``-19998`` as if it were a measurement.

    Args:
        values: The data array.
        func: A callable taking and returning an array of the same shape.
        domain: True where a cell holds data; its gaps are preserved.
        sentinel: The no-data value a gap is re-marked with, and the one the result declares.

    Returns:
        tuple: the transformed array (fresh), and the sentinel the result declares.

    Raises:
        ValueError: ``func`` returned an array of a different shape.

    Examples:
        >>> import numpy as np
        >>> out, sentinel = apply_values(
        ...     np.array([5.0, -9999.0]), lambda a: a * 2,
        ...     domain=np.array([True, False]), sentinel=-9999.0,
        ... )
        >>> out.tolist()
        [10.0, -9999.0]
        >>> apply_values(np.array([1.0, 2.0]), lambda a: a[:1], domain=np.array([True, True]))
        Traceback (most recent call last):
            ...
        ValueError: apply: the function changed the array shape from (2,) to (1,); ...
    """
    array = np.asarray(values)
    result = np.asarray(func(array))
    if result.shape != array.shape:
        raise ValueError(
            f"apply: the function changed the array shape from {array.shape} to "
            f"{result.shape}; it must return one value per element so the result still "
            f"lines up with the mesh."
        )
    # `regap` always returns a fresh array, so an identity `func` cannot leave the result
    # aliasing the caller's buffer -- a later in-place write through one must not reach both.
    return regap(result, domain, sentinel), sentinel
