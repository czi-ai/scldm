import torch

from scldm.nnets import DiT


def _build_dit(pretrained_cfg: dict) -> DiT:
    return DiT(
        n_embed=4,
        n_embed_input=4,
        n_layer=1,
        n_head=1,
        seq_len=2,
        dropout=0.0,
        bias=True,
        norm_layer="layernorm",
        multiple_of=1,
        layernorm_eps=1e-5,
        class_vocab_sizes={"cell_type": 2},
        cfg_dropout_prob=0.1,
        condition_strategy="mutually_exclusive",
        pretrained_class_embeddings=pretrained_cfg,
    )


def test_pretrained_class_embeddings_load_and_freeze(tmp_path) -> None:
    weight = torch.randn(3, 4)
    payload = {
        "state_dict": {"class_embeddings.cell_type.weight": weight},
        "labels": {"cell_type": ["B", "T"]},
    }
    ckpt_path = tmp_path / "embeddings.pt"
    torch.save(payload, ckpt_path)

    model = _build_dit(
        {
            "ckpt_path": str(ckpt_path),
            "freeze": True,
            "strict": True,
        }
    )
    classes2idx = {"cell_type": {"B": 0, "T": 1}}

    model.load_pretrained_class_embeddings_from_config(classes2idx=classes2idx)

    loaded_weight = model.class_embeddings["cell_type"].weight.detach().cpu()
    assert torch.allclose(loaded_weight, weight)
    assert model.class_embeddings["cell_type"].weight.requires_grad is False


def test_pretrained_class_embeddings_label_mismatch_raises(tmp_path) -> None:
    weight = torch.randn(3, 4)
    payload = {
        "state_dict": {"class_embeddings.cell_type.weight": weight},
        "labels": {"cell_type": ["T", "B"]},
    }
    ckpt_path = tmp_path / "embeddings_mismatch.pt"
    torch.save(payload, ckpt_path)

    model = _build_dit(
        {
            "ckpt_path": str(ckpt_path),
            "freeze": False,
            "strict": True,
        }
    )
    classes2idx = {"cell_type": {"B": 0, "T": 1}}

    try:
        model.load_pretrained_class_embeddings_from_config(classes2idx=classes2idx)
    except ValueError as exc:
        assert "Label order mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for label order mismatch.")
