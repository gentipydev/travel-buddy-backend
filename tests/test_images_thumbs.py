"""Small photo copies: the resize helper and how the image endpoints pair
each full-size photo with its copy in `thumbs/` (falling back to the full
URL when there is none)."""
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.integrations.supabase_storage import thumb_path
from app.routers import images
from scripts.upload_destination_photos import make_thumbnail

pytestmark = pytest.mark.anyio


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 60)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((4000, 2667), (1280, 853)),   # landscape: width-bound
        ((2000, 4000), (960, 1920)),   # tall portrait: height-bound
        ((800, 600), (800, 600)),      # already small: never upscaled
    ],
)
def test_make_thumbnail_size(size, expected):
    out = make_thumbnail(_jpeg(*size))
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"
        assert im.size == expected
        assert "exif" not in im.info


def test_thumb_path():
    assert thumb_path("destinations/berat/berat-1.jpg") == "destinations/berat/thumbs/berat-1.jpg"
    assert thumb_path("berat/onufri.jpg") == "berat/thumbs/onufri.jpg"


@pytest.fixture
def fake_storage(monkeypatch):
    """Storage with a configurable listing per folder; URLs are fake."""
    listings: dict[str, list[str] | Exception] = {}

    async def list_image_names(bucket: str, prefix: str) -> list[str]:
        value = listings.get(f"{bucket}/{prefix}", [])
        if isinstance(value, Exception):
            raise value
        return value

    async def public_url(bucket: str, path: str) -> str:
        return f"https://cdn/{bucket}/{path}"

    monkeypatch.setattr(images, "list_image_names", list_image_names)
    monkeypatch.setattr(images, "public_url", public_url)
    return listings


async def test_photos_pair_each_photo_with_its_thumb(fake_storage):
    fake_storage["place-photos/destinations/berat"] = ["berat-1.jpg", "berat-2.jpg"]
    fake_storage["place-photos/destinations/berat/thumbs"] = ["berat-1.jpg"]

    out = await images.list_destination_photos("berat")

    assert out.photos == [
        "https://cdn/place-photos/destinations/berat/berat-1.jpg",
        "https://cdn/place-photos/destinations/berat/berat-2.jpg",
    ]
    assert out.thumbs == [
        "https://cdn/place-photos/destinations/berat/thumbs/berat-1.jpg",
        # no small copy yet -> full-size URL
        "https://cdn/place-photos/destinations/berat/berat-2.jpg",
    ]


async def test_photos_fall_back_to_full_size_when_thumb_listing_fails(fake_storage):
    fake_storage["place-photos/destinations/korca"] = ["korca-1.jpg"]
    fake_storage["place-photos/destinations/korca/thumbs"] = RuntimeError("storage down")

    out = await images.list_destination_photos("korca")

    assert out.thumbs == out.photos


def _artifact_row(image_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="x", title="t", image_path=image_path, date_text=None, period=None, country=None,
        culture=None, medium=None, dimension=None, classification=None, object_type=None,
        object_begin_year=None, object_end_year=None, icon_name=None,
    )


async def test_artifact_image_small_uses_thumb_when_present(fake_storage):
    fake_storage["artifact-photos/berat/thumbs"] = ["onufri.jpg"]
    with_thumb, without_thumb = _artifact_row("berat/onufri.jpg"), _artifact_row("berat/icon.jpg")

    thumbs = await images._artifact_thumbs([with_thumb, without_thumb])
    a = await images._to_artifact_out(with_thumb, thumbs)
    b = await images._to_artifact_out(without_thumb, thumbs)

    assert a.image == "https://cdn/artifact-photos/berat/onufri.jpg"
    assert a.image_small == a.image_medium == "https://cdn/artifact-photos/berat/thumbs/onufri.jpg"
    assert b.image_small == b.image == "https://cdn/artifact-photos/berat/icon.jpg"
