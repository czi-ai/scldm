import json
from pathlib import Path

import numpy as np

from scldm.encoder import VocabularyEncoderSimplified


def test_encoder_loads_metadata_json(tmp_path: Path) -> None:
    metadata = {
        "genes": ["gene_a", "gene_b"],
        "labels": {"cell_type": ["B", "T"]},
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    encoder = VocabularyEncoderSimplified(
        adata_path=None,
        class_vocab_sizes={"cell_type": 2},
        metadata_json=metadata_path,
    )

    np.testing.assert_array_equal(encoder.encode_genes(["gene_a", "gene_b"]), np.array([1, 2]))
    np.testing.assert_array_equal(
        encoder.encode_metadata(["B", "T"], label="cell_type"),
        np.array([0, 1]),
    )
