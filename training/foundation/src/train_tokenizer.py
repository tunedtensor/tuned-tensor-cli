from __future__ import annotations

from common import ensure_private_directory, load_config, make_private_file, write_json
from data import corpus_hash, example_pairs, smoke_corpus, train_tokenizer


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    output_dir = ensure_private_directory(config["output_dir"])
    examples = example_pairs(config.get("examples") or [])
    corpus = smoke_corpus(examples, int(config.get("max_chars") or 20_000), extra=str(config.get("system_prompt") or ""))
    tokenizer = train_tokenizer(corpus, int(config.get("vocab_size") or 256))
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    make_private_file(tokenizer_path)
    write_json(output_dir / "metrics.json", {
        "ok": True,
        "vocab_size": tokenizer.get_vocab_size(),
        "corpus_chars": len(corpus),
        "corpus_sha256": corpus_hash(corpus),
        "tokenizer_path": str(tokenizer_path),
    })


if __name__ == "__main__":
    main()
