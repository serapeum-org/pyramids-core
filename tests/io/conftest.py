"""Fixtures for the compressed-virtual-file-system reads under ``tests/io``.

Paths to the committed gzip/zip/tar/7z archives (and the member lists inside them) that the
pyramids virtual-file-system reader is exercised against. Scoped to ``tests/io`` because only
these tests consume them; moved here from the root conftest so the root keeps only
cross-cutting bootstrap/determinism fixtures.
"""

from typing import List

import pytest


@pytest.fixture(scope="module")
def one_compressed_file_gzip() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.gz"


@pytest.fixture(scope="module")
def unzip_gzip_file_name() -> str:
    return "tests/data/chirps-v2.0.2009.01.01.tif"


@pytest.fixture(scope="module")
def multiple_compressed_file_gzip() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.gz"


@pytest.fixture(scope="module")
def multiple_compressed_file_gzip_content() -> List[str]:
    return ["1.asc", "2.asc"]


@pytest.fixture(scope="module")
def one_compressed_file_zip() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.zip"


@pytest.fixture(scope="module")
def multiple_compressed_file_zip() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.zip"


@pytest.fixture(scope="module")
def multiple_compressed_file_zip_content() -> List[str]:
    return ["1.asc", "2.asc"]


@pytest.fixture(scope="module")
def multiple_compressed_file_tar() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.tar"


@pytest.fixture(scope="module")
def one_compressed_file_tar() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.tar"
