import pytest
import torch

from asteroid.models import XUMX

sources = [
    ["bass", "drums", "vocals", "other"],
    ["drums"],
]


@pytest.mark.skip(reason="XUMX is not broken in torch 2.x")
@pytest.mark.parametrize("nb_channels", (1, 2))
@pytest.mark.parametrize("sources", sources)
@pytest.mark.parametrize("bidirectional", (True, False))
@pytest.mark.parametrize("spec_power", (1, 2))
@pytest.mark.parametrize("return_time_signals", (True, False))
@pytest.mark.parametrize(
    "data",
    (
        torch.rand(3, 2, 4100, requires_grad=False),
        torch.rand(1, 2, 2 * 4100, requires_grad=False),
    ),
)
def test_forward(nb_channels, sources, bidirectional, spec_power, return_time_signals, data):
    x_umx = XUMX(
        window_length=4096,
        input_mean=None,
        input_scale=None,
        nb_channels=nb_channels,
        hidden_size=10,
        nb_layers=2,
        in_chan=4096,
        n_hop=1024,
        sources=sources,
        max_bin=1000,
        bidirectional=bidirectional,
        sample_rate=44100,
        spec_power=spec_power,
        return_time_signals=return_time_signals,
    )
    x_umx = x_umx.eval()
    with torch.no_grad():
        x_umx(data)


@pytest.mark.skip(reason="XUMX is not broken in torch 2.x")
def test_get_model_args():
    sources_tmp = ["vocals"]
    x_umx = XUMX(sources=sources_tmp, window_length=4096)
    expected = {
        "window_length": 4096,
        "in_chan": 4096,
        "n_hop": 1024,
        "sample_rate": 44100,
        "sources": sources_tmp,
        "hidden_size": 512,
        "nb_channels": 2,
        "input_mean": None,
        "input_scale": None,
        "max_bin": 4096 // 2 + 1,
        "nb_layers": 3,
        "bidirectional": True,
        "spec_power": 1,
        "return_time_signals": False,
    }
    assert x_umx.get_model_args() == expected


@pytest.mark.skip(reason="XUMX is not broken in torch 2.x")
def test_model_loading():
    sources_tmp = ["bass", "drums", "vocals", "other"]
    model = XUMX(
        sources=sources_tmp,
        nb_layers=2,
        hidden_size=12,
    )

    model_conf = model.serialize()

    new_model = XUMX.from_pretrained(model_conf)

    random_input = torch.rand(3, 2, 44100, requires_grad=False)
    model = model.eval()
    new_model = new_model.eval()
    with torch.no_grad():
        output1 = model(random_input)
        output2 = new_model(random_input)
    assert torch.allclose(output1[0], output2[0])
