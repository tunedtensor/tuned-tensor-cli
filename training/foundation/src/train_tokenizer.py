from __future__ import annotations

from common import ensure_private_directory, load_config, make_private_file, write_json
from data import (
    corpus_documents_hash,
    corpus_hash,
    corpus_manifest,
    example_pairs,
    instruction_corpus_documents,
    iter_corpus_documents,
    smoke_corpus,
    train_tokenizer,
)


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    output_dir = ensure_private_directory(config["output_dir"])
    examples = example_pairs(config.get("examples") or [])
    max_chars = int(config.get("max_chars") or 20_000)
    corpus_path = config.get("corpus_path")
    if corpus_path:
        manifest_before = corpus_manifest(corpus_path)
        system_prompt = str(config.get("system_prompt") or "")

        def documents():
            return instruction_corpus_documents(
                system_prompt,
                iter_corpus_documents(corpus_path, max_chars),
            )

        corpus_sha256, corpus_chars = corpus_documents_hash(documents())
        tokenizer = train_tokenizer(
            documents(),
            int(config.get("vocab_size") or 256),
        )
        if corpus_manifest(corpus_path) != manifest_before:
            raise RuntimeError("Foundation corpus changed while the tokenizer was being trained.")
    else:
        corpus = smoke_corpus(examples, max_chars, extra=str(config.get("system_prompt") or ""))
        corpus_sha256 = corpus_hash(corpus)
        corpus_chars = len(corpus)
        tokenizer = train_tokenizer(corpus, int(config.get("vocab_size") or 256))
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    make_private_file(tokenizer_path)
    write_json(output_dir / "metrics.json", {
        "ok": True,
        "vocab_size": tokenizer.get_vocab_size(),
        "corpus_chars": corpus_chars,
        "corpus_sha256": corpus_sha256,
        "corpus_path": str(corpus_path) if corpus_path else None,
        "corpus_manifest": manifest_before if corpus_path else None,
        "tokenizer_path": str(tokenizer_path),
    })


if __name__ == "__main__":
    main()
