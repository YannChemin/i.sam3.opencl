# SPDX-License-Identifier: Unlicense
"""Tests for i.sam3.opencl."""

import functools
import http.server
import io
import pickle
import shutil
import sys
import threading
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest
from conftest import RECTANGLES
from grass.tools import ToolError, Tools


def _write_checkpoint(path, shapes):
    """Write a torch.save()-format checkpoint of zero float32 tensors with
    the given {name: shape}, wrapped as {"model": state_dict} like the SAM 2
    releases, without needing torch: stand-ins for the two torch globals
    the pickle references are registered just long enough to pickle."""
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        pass

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    torch_mod.FloatStorage = FloatStorage
    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2

    class Storage:
        def __init__(self, key, numel):
            self.key, self.numel = key, numel

    class Tensor:
        def __init__(self, storage, shape):
            self.storage, self.shape = storage, shape

        def __reduce__(self):
            stride, s = [], 1
            for n in reversed(self.shape):
                stride.insert(0, s)
                s *= n
            return (
                _rebuild_tensor_v2,
                (
                    self.storage,
                    0,
                    tuple(self.shape),
                    tuple(stride),
                    False,
                    OrderedDict(),
                ),
            )

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", FloatStorage, obj.key, "cpu", obj.numel)
            return None

    state = {}
    storages = []
    for i, (name, shape) in enumerate(shapes.items()):
        numel = 1
        for n in shape:
            numel *= n
        storages.append((str(i), numel))
        state[name] = Tensor(Storage(str(i), numel), shape)

    saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = torch_mod, utils_mod
    try:
        buf = io.BytesIO()
        Pickler(buf, protocol=2).dump({"model": state})
    finally:
        for k, v in saved.items():
            if v is None:
                del sys.modules[k]
            else:
                sys.modules[k] = v

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for key, numel in storages:
            z.writestr(f"archive/data/{key}", bytes(4 * numel))


def test_group_must_hold_three_bands(session, tmp_path):
    """A group with other than three raster maps is rejected up front."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pt"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match="exactly 3"):
        tools.i_sam3_opencl(group="two", checkpoint=str(dummy), output="out")


def test_rejects_file_that_is_not_a_checkpoint(session, tmp_path):
    """A file that is not a PyTorch zip archive fails before any OpenCL
    work."""
    tools = Tools(session=session)
    bogus = tmp_path / "bogus.pt"
    bogus.write_bytes(b"\0" * 4096)

    with pytest.raises(ToolError, match="not a zip archive"):
        tools.i_sam3_opencl(group="rgb", checkpoint=str(bogus), output="out")


def test_rejects_zip_without_state_dict(session, tmp_path):
    """A zip archive without a pickled state_dict is not a checkpoint."""
    tools = Tools(session=session)
    archive = tmp_path / "other.pt"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("other/readme.txt", "hello")

    with pytest.raises(ToolError, match="not a PyTorch checkpoint"):
        tools.i_sam3_opencl(group="rgb", checkpoint=str(archive), output="out")


def test_rejects_sam1_checkpoint(session, tmp_path):
    """An original SAM checkpoint is refused with a pointer to
    i.sam.opencl."""
    tools = Tools(session=session)
    ckpt = tmp_path / "sam_vit_b.pth"
    _write_checkpoint(
        ckpt,
        {
            "image_encoder.pos_embed": (1, 64, 64, 8),
            "mask_decoder.iou_token.weight": (1, 256),
        },
    )

    with pytest.raises(ToolError, match=r"SAM\s+\(version\s+1\).*i\.sam\.opencl"):
        tools.i_sam3_opencl(group="rgb", checkpoint=str(ckpt), output="out")


def test_rejects_sam2_checkpoint(session, tmp_path):
    """A SAM 2 checkpoint is refused with a pointer to i.sam2.opencl."""
    tools = Tools(session=session)
    ckpt = tmp_path / "sam2.1_hiera_tiny.pt"
    _write_checkpoint(
        ckpt,
        {
            "image_encoder.trunk.pos_embed": (1, 96, 7, 7),
            "sam_mask_decoder.iou_token.weight": (1, 256),
        },
    )

    with pytest.raises(ToolError, match=r"SAM\s+2\s+checkpoint.*i\.sam2\.opencl"):
        tools.i_sam3_opencl(group="rgb", checkpoint=str(ckpt), output="out")


@pytest.fixture(scope="module")
def segmented(session, checkpoint, device_options):
    """Segment the rectangles once for all result checks."""
    tools = Tools(session=session)
    tools.i_sam3_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output="segments",
        raster="segments",
        points_per_side=8,
        minsize=50,
        **device_options,
    )
    return tools


def _category_at(tools, row, col):
    """Segment category of the cell at 1-based (row, col)."""
    x, y = col - 0.5, 256 - row + 0.5
    value = tools.r_what(map="segments", coordinates=(x, y), format="json").json
    return value[0]["segments"]["value"]


def test_each_rectangle_is_one_segment(segmented):
    """Every rectangle is recovered as its own segment of the right size."""
    categories = set()
    for name, n, s, w, e, _ in RECTANGLES:
        cat = _category_at(segmented, (n + s) // 2, (w + e) // 2)
        assert cat is not None, f"{name} rectangle not segmented"
        categories.add(cat)

        cells = (s - n + 1) * (e - w + 1)
        row = segmented.v_db_select(
            map="segments", where=f"cat = {cat}", format="json"
        ).json["records"][0]
        assert row["cells"] == pytest.approx(cells, rel=0.05), name
        assert row["area"] == pytest.approx(cells, rel=0.05), name
        assert 0 < row["predicted_iou"] <= 1
        assert 0 < row["stability"] <= 1
    assert len(categories) == len(RECTANGLES)


def test_segments_form_a_partition(segmented):
    """Segments do not overlap: the attribute cell counts add up to the
    labelled cells, and every segment has exactly one area."""
    records = segmented.v_db_select(map="segments", format="json").json["records"]
    labelled = segmented.r_univar(map="segments", format="json").json["n"]
    assert sum(r["cells"] for r in records) == labelled

    topo = segmented.v_info(map="segments", flags="t", format="json").json
    assert topo["centroids"] == len(records)
    assert topo["areas"] >= topo["centroids"]


def _tools_with_env(session, **env):
    return Tools(env={**session.env, **env})


def test_prompt_must_be_ascii(session, tmp_path):
    """Prompts needing Unicode cleaning are refused before any OpenCL
    work."""
    tools = Tools(session=session)
    dummy = tmp_path / "sam3.pt"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"not\s+plain\s+ASCII"):
        tools.i_sam3_opencl(
            group="rgb", checkpoint=str(dummy), prompt="for\u00eat", output="out"
        )


def test_text_prompts(session, checkpoint, device_options):
    """Text prompts produce segments labelled with the prompt that found
    them, with a detection score above the threshold."""
    tools = Tools(session=session)
    tools.i_sam3_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output="found",
        prompt="rectangle,square",
        minsize=50,
        **device_options,
    )
    records = tools.v_db_select(map="found", format="json").json["records"]
    assert records, "no instance found"
    for r in records:
        assert r["prompt"] in {"rectangle", "square"}
        assert 0.5 < r["score"] <= 1


def test_default_weights_dir(session, tmp_path):
    """Without checkpoint or weights_dir, the central folder
    $SAM_MODELS_DIR/sam3 is searched."""
    tools = _tools_with_env(session, SAM_MODELS_DIR=str(tmp_path))
    expected = tmp_path / "sam3"

    with pytest.raises(ToolError, match=r"No\s+SAM\s+3\s+checkpoint") as e:
        tools.i_sam3_opencl(group="rgb", output="out", flags="n")
    assert str(expected) in " ".join(str(e.value).split())


def test_checkpoint_and_weights_dir_are_exclusive(session, tmp_path):
    """A checkpoint file and a weights folder cannot both be given."""
    tools = Tools(session=session)
    dummy = tmp_path / "sam3.pt"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"mutually\s+exclusive"):
        tools.i_sam3_opencl(
            group="rgb",
            checkpoint=str(dummy),
            weights_dir=str(tmp_path),
            output="out",
        )


def test_empty_weights_dir(session, tmp_path):
    """A folder without sam3.pt fails before any OpenCL work."""
    tools = Tools(session=session)
    (tmp_path / "readme.txt").write_text("no weights here")
    (tmp_path / "sam_vit_h_4b8939.pth").write_bytes(b"a SAM 1 checkpoint")

    with pytest.raises(ToolError, match=r"No\s+SAM\s+3\s+checkpoint"):
        tools.i_sam3_opencl(
            group="rgb", weights_dir=str(tmp_path), output="out", flags="n"
        )


def test_download_needs_token(session, tmp_path):
    """Without a Hugging Face token, the gated checkpoint cannot be
    downloaded, and the error says how to get access."""
    tools = _tools_with_env(
        session, SAM_MODELS_DIR=str(tmp_path), HF_TOKEN="", HF_HOME=str(tmp_path)
    )

    with pytest.raises(ToolError, match=r"Hugging\s+Face\s+access\s+token"):
        tools.i_sam3_opencl(group="rgb", output="out")
    assert not (tmp_path / "sam3").exists() or not list((tmp_path / "sam3").iterdir())


def test_weights_dir_segments(session, checkpoint, device_options, tmp_path):
    """Segmentation runs with sam3.pt found in a weights folder."""
    tools = Tools(session=session)
    (tmp_path / "sam3.pt").symlink_to(checkpoint)
    tools.i_sam3_opencl(
        group="rgb",
        weights_dir=str(tmp_path),
        output="from_folder",
        points_per_side=8,
        minsize=50,
        **device_options,
    )
    topo = tools.v_info(map="from_folder", flags="t", format="json").json
    assert topo["centroids"] >= len(RECTANGLES)


# Published size of sam3.pt.
SAM3_SIZE = 3450062241


@pytest.fixture
def mirror(tmp_path):
    """A local HTTP server standing in for the download site; yields the
    folder it serves and its URL."""
    root = tmp_path / "mirror"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_download_rejects_wrong_size(session, tmp_path, mirror):
    """A download that does not have the published size is discarded, not
    installed in the weights folder."""
    root, url = mirror
    (root / "sam3.pt").write_bytes(b"truncated")
    weights = tmp_path / "weights"
    tools = _tools_with_env(session, I_SAM3_OPENCL_DOWNLOAD_URL=url)

    with pytest.raises(ToolError, match=rf"instead\s+of\s+{SAM3_SIZE}"):
        tools.i_sam3_opencl(group="rgb", weights_dir=str(weights), output="out")
    assert not list(weights.glob("*.pt*"))


def test_download_fetches_missing_model(session, checkpoint, tmp_path, mirror):
    """sam3.pt missing from the weights folder (created if needed) is
    downloaded there and used."""
    if Path(checkpoint).stat().st_size != SAM3_SIZE:
        pytest.skip("checkpoint is not the published sam3.pt")
    if shutil.disk_usage(tmp_path).free < 2 * SAM3_SIZE:
        pytest.skip(f"not enough free space in {tmp_path} for a copy of sam3.pt")
    root, url = mirror
    (root / "sam3.pt").symlink_to(checkpoint)
    weights = tmp_path / "new" / "weights"
    tools = _tools_with_env(session, I_SAM3_OPENCL_DOWNLOAD_URL=url)

    # A platform that does not exist stops the run right after the
    # checkpoint has been fetched and read.
    with pytest.raises(ToolError, match=r"OpenCL"):
        tools.i_sam3_opencl(
            group="rgb",
            weights_dir=str(weights),
            output="out",
            platform="no_such_platform",
        )
    assert (weights / "sam3.pt").stat().st_size == SAM3_SIZE
    assert not list(weights.glob("*.part"))


@pytest.mark.parametrize("option", ["profile", "precision"])
def test_rejects_invalid_device_option(session, tmp_path, option):
    """profile= and precision= only take their listed values."""
    tools = Tools(session=session)
    with pytest.raises(ToolError):
        tools.i_sam3_opencl(
            group="rgb",
            checkpoint=str(tmp_path / "missing.pt"),
            output="out",
            **{option: "nonsense"},
        )


@pytest.mark.parametrize(
    ("profile", "precision"), [("cpu", "fp32"), ("igpu", "fp32"), ("dgpu", "fp16")]
)
def test_profiles_and_precision_give_the_rectangles(
    session, checkpoint, device_options, profile, precision
):
    """A forced device profile, and half-precision weights, still recover
    every rectangle at the right size."""
    tools = Tools(session=session)
    out = f"seg_{profile}_{precision}"
    tools.i_sam3_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output=out,
        raster=out,
        points_per_side=8,
        minsize=50,
        profile=profile,
        precision=precision,
        **device_options,
    )
    for name, n, s, w, e, _ in RECTANGLES:
        x, y = (w + e) // 2 - 0.5, 256 - (n + s) // 2 + 0.5
        cat = tools.r_what(map=out, coordinates=(x, y), format="json").json[0][out][
            "value"
        ]
        assert cat is not None, f"{name} rectangle not segmented"
        row = tools.v_db_select(map=out, where=f"cat = {cat}", format="json").json[
            "records"
        ][0]
        assert row["cells"] == pytest.approx((s - n + 1) * (e - w + 1), rel=0.05)
