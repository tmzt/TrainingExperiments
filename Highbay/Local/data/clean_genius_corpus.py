#!/usr/bin/env python3
"""Produce the cleaned Genius training corpus from the raw files in source/.

The inputs are Gemini-generated training data, so their shape carries
generation artifacts that a hand-authored schema would not. This script is the
one place those are fixed, so the notebook and any Rust loader can start from a
file that already agrees with itself.

WHAT IT FIXES, all of which were measured rather than assumed:

  1. TWO ON-DISK FORMATS. mobile_data_prompts.jsonl is real JSONL carrying a
     UTF-8 BOM; prompts.jsonl is a pretty-printed JSON ARRAY despite its name.
     Reading the latter line-by-line does not error - it yields
     schema_mutations fragments that look like records - so the loader tries
     the array form first. Output is true JSONL, UTF-8, no BOM.

  2. UPDATE_RECORD's payload is written two ways. It is a field->value map in
     23/23 records, but the generator emitted an object 8 times and a
     STRINGIFIED object 15 times. Same meaning, two spellings, no rule a model
     can learn. The map wins; the strings are parsed back.

  3. DUPLICATE user_inputs across the two files are one example, not two.

WHAT IT DELIBERATELY LEAVES ALONE:

  * conditions[].value is str x44 / int x4 and that union is REAL -
    `total_amount > 500` sits beside string comparisons.
  * Table and column names, ui_prompt wording, and the choice of columns. Those
    are the content; this script only touches how it was spelled.

Sources, script and outputs all live in this directory, so a clone of this
repo alone is enough to regenerate:

    source/mobile_data_prompts.jsonl   the raw generated corpora, verbatim -
    source/prompts.jsonl               BOM, array-not-JSONL and all
    clean_genius_corpus.py             this
    genius_corpus_clean.jsonl          derived
    genius_chatml.jsonl                derived

Usage:
    python3 clean_genius_corpus.py                 # source/ beside this file
    python3 clean_genius_corpus.py --src PATH      # read the corpora elsewhere
    python3 clean_genius_corpus.py --out DIR       # default: beside this file
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ["mobile_data_prompts.jsonl", "prompts.jsonl"]

# Ordered guesses, first hit wins. `source/` is the answer for a standalone
# clone and is therefore first; the rest are fallbacks for someone who has the
# Highbay tree or a Colab runtime with the files staged on Drive.
def _src_candidates():
    out = [HERE / "source"]
    # <highbay>/deps/TrainingExperiments/Highbay/Local/data is five levels down,
    # but a shallow clone has fewer parents than that - indexing blindly raises
    # IndexError before the search ever runs, which is how this was found.
    if len(HERE.parents) > 4:
        out.append(HERE.parents[4] / "data/genius")
    out.append(Path.cwd() / "data/genius")
    out.append(Path("/content/drive/MyDrive/Training Data"))
    return out


SRC_CANDIDATES = _src_candidates()


def resolve_src(explicit):
    """The directory holding the source corpora, or a clear error saying where
    it looked. A wrong guess here would silently produce a partial corpus."""
    if explicit:
        src = Path(explicit).expanduser().resolve()
        if not src.is_dir():
            raise SystemExit(f"--src {src} is not a directory")
        return src
    for candidate in SRC_CANDIDATES:
        if all((candidate / name).is_file() for name in SOURCES):
            return candidate.resolve()
    looked = "\n  ".join(str(c) for c in SRC_CANDIDATES)
    raise SystemExit(
        f"could not find {SOURCES} in any of:\n  {looked}\n"
        "Pass --src with the directory that holds them.")

INTENT_TYPES = {"PIPELINE", "SCHEMA_SUGGESTION"}
TRIGGER_TYPES = {"ON_CREATE", "ON_UPDATE", "ON_DELETE", "SCHEDULED"}
ACTION_TYPES = {"SEND_NOTIFICATION", "UPDATE_RECORD", "CALCULATE"}
COLUMN_TYPES = {"NUMBER", "DATE", "STRING", "BOOLEAN", "RELATION"}
MUTATION_KEYS = {
    "CREATE_TABLE": {"action", "name"},
    "ADD_COLUMN": {"action", "table", "column_name", "type"},
}

SYSTEM = (
    "You convert a user's plain-language automation request into a strict JSON "
    "AST. Reply with JSON only - no prose, no code fences."
)


def load_corpus(text):
    """A JSON array or true JSONL. Array first - see the module doc."""
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def canonical(ast):
    """One string form per AST: key order and spacing normalized, so a
    comparison measures the AST rather than its formatting."""
    return json.dumps(ast, sort_keys=True, separators=(",", ":"))


def normalize_payload(ast):
    """Parse a stringified UPDATE_RECORD payload back into the map it means.

    Returns a new AST; never mutates its argument. Note the DIRECTION: the map
    is the meaning and the string is the artifact, so the map wins. Normalizing
    the other way - stringifying the objects - promotes the artifact to ground
    truth.
    """
    action = (ast.get("pipeline_ast") or {}).get("action")
    if not action or not isinstance(action.get("payload"), str):
        return ast
    try:
        decoded = json.loads(action["payload"])
    except json.JSONDecodeError:
        return ast                      # a plain message, as intended
    if not isinstance(decoded, dict):
        return ast                      # a bare string or number, likewise
    action = dict(action, payload=decoded)
    return dict(ast, pipeline_ast=dict(ast["pipeline_ast"], action=action))


def validate(ast):
    """Problems with this AST; empty means well-formed."""
    problems = []
    if not isinstance(ast, dict):
        return ["not an object"]
    intent = ast.get("intent_type")
    if intent not in INTENT_TYPES:
        problems.append(f"intent_type={intent!r}")
    if intent == "PIPELINE":
        pipeline = ast.get("pipeline_ast")
        if not isinstance(pipeline, dict):
            problems.append("PIPELINE without pipeline_ast")
        else:
            trigger, action = pipeline.get("trigger"), pipeline.get("action")
            if not isinstance(trigger, dict):
                problems.append("trigger missing")
            elif trigger.get("type") not in TRIGGER_TYPES:
                problems.append(f"trigger.type={trigger.get('type')!r}")
            if not isinstance(action, dict):
                problems.append("action missing")
            elif action.get("type") not in ACTION_TYPES:
                problems.append(f"action.type={action.get('type')!r}")
    if intent == "SCHEMA_SUGGESTION":
        mutations = ast.get("schema_mutations")
        if not isinstance(mutations, list) or not mutations:
            problems.append("SCHEMA_SUGGESTION without schema_mutations")
        else:
            for i, mutation in enumerate(mutations):
                want = MUTATION_KEYS.get(mutation.get("action"))
                if want is None:
                    problems.append(f"mutation[{i}].action={mutation.get('action')!r}")
                elif set(mutation) != want:
                    # Exact key set BOTH ways: an EXTRA key is the Arrow
                    # struct-unification backfill, which this also catches.
                    problems.append(
                        f"mutation[{i}] keys {sorted(set(mutation))} != {sorted(want)}")
                elif (mutation["action"] == "ADD_COLUMN"
                        and mutation.get("type") not in COLUMN_TYPES):
                    problems.append(f"mutation[{i}].type={mutation.get('type')!r}")
    return problems


def to_chatml(record):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": record["user_input"]},
        {"role": "assistant", "content": canonical(record["output_ast"])},
    ]}


def main():
    ap = argparse.ArgumentParser(description="Build the cleaned Genius corpus.")
    ap.add_argument("--src", default=None,
                    help="directory holding %s" % ", ".join(SOURCES))
    ap.add_argument("--out", default=str(HERE),
                    help="where to write (default: beside this script)")
    args = ap.parse_args()
    src_dir = resolve_src(args.src)
    print(f"  source: {src_dir}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, provenance = [], []
    for name in SOURCES:
        path = src_dir / name
        # utf-8-sig: mobile_data_prompts.jsonl has a BOM, and with plain utf-8
        # the FIRST record and only the first fails to parse.
        got = load_corpus(path.read_text(encoding="utf-8-sig"))
        provenance.append((name, len(got)))
        print(f"  {name}: {len(got)} records")
        records += got

    seen, deduped = set(), []
    for record in records:
        if record["user_input"] not in seen:
            seen.add(record["user_input"])
            deduped.append(record)
    dupes = len(records) - len(deduped)
    print(f"  merged {len(records)}, dropped {dupes} duplicate user_inputs")
    records = deduped

    def payload_types():
        counts = collections.Counter()
        for record in records:
            action = (record["output_ast"].get("pipeline_ast") or {}).get("action") or {}
            if action.get("type") == "UPDATE_RECORD":
                counts[type(action.get("payload")).__name__] += 1
        return dict(counts)

    before = payload_types()
    records = [{"user_input": r["user_input"],
                "output_ast": normalize_payload(r["output_ast"])} for r in records]
    after = payload_types()
    print(f"  UPDATE_RECORD payload: {before} -> {after}")

    bad = [(i, p) for i, p in
           ((i, validate(r["output_ast"])) for i, r in enumerate(records)) if p]
    print(f"  validate: {len(records) - len(bad)}/{len(records)} well-formed")
    for i, problems in bad[:10]:
        print(f"    record {i}: {problems}")
    if bad:
        raise SystemExit("refusing to write a corpus that does not validate")

    corpus_path = out_dir / "genius_corpus_clean.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:   # utf-8, no BOM
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    chatml_path = out_dir / "genius_chatml.jsonl"
    with chatml_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(to_chatml(record), ensure_ascii=False) + "\n")

    # Read both back the way a consumer would, and prove nothing moved.
    reread = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert len(reread) == len(records), "corpus did not round-trip"
    assert all(canonical(a["output_ast"]) == canonical(b["output_ast"])
               for a, b in zip(reread, records)), "an AST changed on write"
    chat = [json.loads(l) for l in chatml_path.read_text(encoding="utf-8").splitlines()]
    assert all(canonical(json.loads(c["messages"][2]["content"]))
               == canonical(r["output_ast"])
               for c, r in zip(chat, records)), "an AST changed through ChatML"

    intents = collections.Counter(r["output_ast"]["intent_type"] for r in records)
    print(f"\n  {corpus_path.name}: {len(records)} records {dict(intents)}")
    print(f"  {chatml_path.name}: {len(chat)} conversations")
    print(f"  written to {out_dir}")
    return provenance, len(records), before, after


if __name__ == "__main__":
    main()
