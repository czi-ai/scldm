# Pretrained class embeddings

You can load pretrained class embeddings for the DiT conditioning layers from a `.pt`
file. The loader expects a `state_dict` entry with embedding weights and a `labels`
map to validate label ordering against the label encoder.

Minimal payload:

```python
payload = {
    "state_dict": {
        "class_embeddings.cell_type.weight": weight,  # shape: [num_classes + cfg, n_embed]
    },
    "labels": {
        "cell_type": ["B", "T"],  # ordered by embedding index
    },
}
torch.save(payload, "class_embeds.pt")
```

Configuration example:

```bash
model.module.diffusion_model.pretrained_class_embeddings.ckpt_path=/path/to/class_embeds.pt \
model.module.diffusion_model.pretrained_class_embeddings.freeze=true \
model.module.diffusion_model.pretrained_class_embeddings.strict=true
```

Notes:

- `labels` order must match the label encoder’s index mapping exactly.
- `strict=true` raises a clear error on any mismatch.
