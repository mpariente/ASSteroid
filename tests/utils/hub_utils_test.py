import os
from asteroid.utils import hub_utils


HF_EXAMPLE_MODEL_IDENTIFER = "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"
# An actual model hosted on huggingface.co

REVISION_ID_ONE_SPECIFIC_COMMIT = "8ab5ef18ef2eda141dd11a5d037a8bede7804ce4"
# One particular commit (not the top of `main`)


def test_download():
    # We download
    path1 = hub_utils.cached_download("mpariente/ConvTasNet_WHAM!_sepclean")
    assert os.path.isfile(path1)
    # We use cache
    path2 = hub_utils.cached_download("mpariente/ConvTasNet_WHAM!_sepclean")
    assert path1 == path2


def test_hf_download():
    # We download
    path1 = hub_utils.cached_download(HF_EXAMPLE_MODEL_IDENTIFER)
    assert os.path.isfile(path1)
    # We use cache
    path2 = hub_utils.cached_download(HF_EXAMPLE_MODEL_IDENTIFER)
    assert path1 == path2
    # However if specifying a particular commit,
    # file will be different.
    path3 = hub_utils.cached_download(
        f"{HF_EXAMPLE_MODEL_IDENTIFER}@{REVISION_ID_ONE_SPECIFIC_COMMIT}"
    )
    assert path3 != path1
