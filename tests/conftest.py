# SPDX-License-Identifier: Unlicense
"""pytest fixtures for i.sam2.opencl.

The segmentation tests need an original SAM 3 checkpoint and an OpenCL
device, neither of which can be assumed on a test machine. They are
skipped unless the checkpoint path is given in I_SAM3_OPENCL_CHECKPOINT.
I_SAM3_OPENCL_PLATFORM optionally selects the OpenCL platform (for Mesa
rusticl, also export RUSTICL_ENABLE=radeonsi or the matching driver).
"""

import os
from pathlib import Path

import grass.script as gs
import pytest
from grass.tools import Tools

# Four flat-coloured rectangles on a grey background: (name, north row,
# south row, west col, east col, (r, g, b)) with r.mapcalc 1-based rows.
RECTANGLES = [
    ("red", 21, 100, 21, 90, (220, 40, 40)),
    ("green", 31, 80, 141, 230, (40, 200, 60)),
    ("blue", 141, 230, 31, 110, (40, 60, 220)),
    ("yellow", 151, 220, 151, 220, (230, 220, 40)),
]
BACKGROUND = (110, 110, 110)
SIZE = 256


def _band_expression(name, band):
    expr = str(BACKGROUND[band])
    for _, n, s, w, e, rgb in RECTANGLES:
        expr = (
            f"if(row() >= {n} && row() <= {s} && col() >= {w} && col() <= {e}, "
            f"{rgb[band]}, {expr})"
        )
    return f"{name} = {expr}"


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session with a 3-band group <rgb> of the rectangles image and a
    2-band group <two>."""
    project = tmp_path_factory.mktemp("i_sam3_opencl") / "project"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(n=SIZE, s=0, e=SIZE, w=0, res=1)
        for band, name in enumerate(("band_r", "band_g", "band_b")):
            tools.r_mapcalc(expression=_band_expression(name, band))
        tools.i_group(group="rgb", input="band_r,band_g,band_b")
        tools.i_group(group="two", input="band_r,band_g")
        yield session


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("I_SAM3_OPENCL_CHECKPOINT")
    if not path or not Path(path).is_file():
        pytest.skip("set I_SAM3_OPENCL_CHECKPOINT to a SAM 3 .pt checkpoint")
    return path


@pytest.fixture(scope="module")
def device_options():
    platform = os.environ.get("I_SAM3_OPENCL_PLATFORM")
    return {"platform": platform} if platform else {}
