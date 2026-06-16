"""Data preparation: smoke corpora, normalization, packing, and SFT/DPO datasets."""

from orkhon.data.code_synth import fim_example, make_fim_sft
from orkhon.data.dataset import DPODataset, SFTDataset
from orkhon.data.decontam import build_ngram_index, decontaminate, is_contaminated
from orkhon.data.dedupe import dedupe_exact, dedupe_minhash
from orkhon.data.download import download_tinystories
from orkhon.data.hf_text import (FINEWEB_EDU, WIKIPEDIA_TR, download_fineweb_edu, download_wikipedia_tr, stream_hf_text)
from orkhon.data.instruct_synth import make_story_instructions
from orkhon.data.normalize import iter_documents, normalize_text
from orkhon.data.pack import PackedDataset
from orkhon.data.synth import (
    make_all,
    make_smoke_corpus,
    make_smoke_dpo,
    make_smoke_sft,
)
from orkhon.data.shard import ShardedPackedDataset, prepare_pretrain_sharded
from orkhon.data.tokenize import prepare_pretrain

__all__ = [
    "prepare_pretrain",
    "prepare_pretrain_sharded",
    "ShardedPackedDataset",
    "download_tinystories",
    "stream_hf_text",
    "download_fineweb_edu",
    "download_wikipedia_tr",
    "WIKIPEDIA_TR",
    "FINEWEB_EDU",
    "make_story_instructions",
    "make_fim_sft",
    "fim_example",
    "PackedDataset",
    "SFTDataset",
    "DPODataset",
    "iter_documents",
    "normalize_text",
    "make_smoke_corpus",
    "make_smoke_sft",
    "make_smoke_dpo",
    "make_all",
]
