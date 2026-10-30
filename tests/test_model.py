import pytest

tf = pytest.importorskip("tensorflow")
from hazardfuse.model import build_hazardfuse  # noqa: E402

# parameter counts of the published run (65 input bands, 21 x 21 patches)
PUBLISHED = {"full": ({}, 98338),
             "w/o band attention": ({"band_attention": False}, 87841),
             "w/o center-pixel branch": ({"center_branch": False, "band_attention": False, "fusion_gate": False}, 53761),
             "w/o fusion gate": ({"fusion_gate": False}, 79810),
             "w/o spatial-embedding branch": ({"spatial_branch": False, "fusion_gate": False}, 26242)}


@pytest.mark.parametrize("name", list(PUBLISHED))
def test_architecture_matches_published(name):
    kw, n = PUBLISHED[name]
    m = build_hazardfuse((21, 21, 65), **kw)
    assert m.output_shape == (None, 1)
    assert m.count_params() == n


def test_even_patch_rejected():
    with pytest.raises(ValueError):
        build_hazardfuse((20, 20, 5))
