"""Tests for putting pictures into the RTF output."""
import io
import struct
import zlib
from contextlib import redirect_stdout

import pytest

from src.display import strBuildRtfImage, subPrintImage, pthSaveRtf


def pthMakePng(pthFolder, intWidth=4, intHeight=2):
    def bytChunk(strType, bytData):
        bytBody = strType + bytData
        return struct.pack(">I", len(bytData)) + bytBody + struct.pack(">I", zlib.crc32(bytBody))
    bytRaw = b"".join(b"\x00" + b"\xff\x00\x00" * intWidth for _ in range(intHeight))
    bytPng = (b"\x89PNG\r\n\x1a\n"
              + bytChunk(b"IHDR", struct.pack(">IIBBBBB", intWidth, intHeight, 8, 2, 0, 0, 0))
              + bytChunk(b"IDAT", zlib.compress(bytRaw)) + bytChunk(b"IEND", b""))
    pthFile = pthFolder / "g.png"
    pthFile.write_bytes(bytPng)
    return pthFile


def test_build_rtf_image_keeps_shape(tmp_path):
    strRtf = strBuildRtfImage(pthMakePng(tmp_path), 6.0)
    assert "\\pngblip" in strRtf and "\\picw4\\pich2" in strRtf
    assert "\\picwgoal8640\\pichgoal4320" in strRtf
    assert "\n" not in strRtf


def test_build_rtf_image_rejects_non_png(tmp_path):
    pthFile = tmp_path / "x.png"
    pthFile.write_text("nope")
    with pytest.raises(ValueError):
        strBuildRtfImage(pthFile)


def test_print_image_on_screen_prints_name(tmp_path):
    objOut = io.StringIO()
    with redirect_stdout(objOut):
        subPrintImage("a.png")
    assert "a.png" in objOut.getvalue()
