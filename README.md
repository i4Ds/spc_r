# Swiss Parliament Corpus – Correction Toolkit

This repository accompanies the [Swiss Parliament Corpus (SPC-R) dataset](https://huggingface.co/datasets/i4ds/spc_r) and the paper [Swiss Parliaments Corpus Re-Imagined (SPC_R): Enhanced Transcription with RAG-based Correction and Predicted BLEU](https://arxiv.org/abs/2506.07726). It provides the scripts used to generate RAG-augmented correction prompts, submit OpenAI Batch jobs, and consolidate the returned corrections and quality judgements. The raw data is not shared in this repository, only the code.

## Idea
The idea is to correct a weakly labeled transcription of a parliamentary discussion with an LLM, which receives as context semantically relevant chunks from a manually generated summary of it.

## Prerequisites

- Python 3.10 or newer.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.
- An OpenAI API key with access to the `gpt-4o` family and the Batch API. Export it before running any command:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

Install the dependencies via `uv` once:
```bash
uv sync
```
`uv` will create an isolated environment and install the packages defined in `pyproject.toml`. Use `uv run ...` for every command below so the managed environment is reused.

## Repository Overview

| Script | Purpose |
| --- | --- |
| `correct_transcriptions.py` | Generates correction or judgement prompts, optionally writes OpenAI Batch JSONL jobs, or runs the correction + judgement pipeline synchronously. |
| `submit_correction_job.py` / `submit_judgement_job.py` | Upload batch JSONL files to OpenAI and start 24h batch runs. |
| `retrieve_correction_results.py` / `retrieve_judgement_results.py` | Poll batch jobs until completion and download the answer JSONL files. |
| `add_correction_to_file.py` | Merge correction batch answers back into the original transcription JSON files. |
| `add_judgement_to_file.py` | Merge judgement batch answers back into the already corrected JSON files. |
| `batch_utils.py` | Thin OpenAI Batch helper functions shared by the CLI utilities. |
| `utils.py` | PDF extraction, embedding cache management, and prompt construction helpers. |

## Typical Workflow

The workflow operates in two phases: correction and judgement. Both can run synchronously (direct API calls) or asynchronously via batches. The batch route scales better and matches the experiments in the paper.

UPDATE: We would strongly recommend to work with [Supervised fine-tuned Models](https://platform.openai.com/docs/guides/supervised-fine-tuning), especially for the quality scoring (Step 5).

### 1. Generate correction batch jobs

```bash
uv run correct_transcriptions.py \
  --folder data/spc_r/kanton_be_grosser_rat/2018_06 \
  --pdf data/spc_r/kanton_be_grosser_rat/2018_06/tagblatt.pdf \
  --batch --step correction \
  --correction_model gpt-4o \
  --temperature 0.1
```

- `--folder` processes every JSON file inside the folder (excluding already corrected ones).
- The script extracts text from the provided summary PDF, chunks it, builds (or loads) cached embeddings in `embeddings/`, and writes `*_batch.jsonl` next to each JSON file.
- For a single file use `--file /path/to/transcript.json` instead of `--folder`.

### 2. Submit correction batches

```bash
uv run submit_correction_job.py --jsonl_folder data/spc_r/kanton_be_grosser_rat/2018_06
```
- uploads every `*_batch.jsonl` to OpenAI and starts a batch.
- writes `<file>.batch_id` next to the JSONL so you can poll later.

### 3. Poll and download correction answers

```bash
uv run retrieve_correction_results.py --batch_job_id_folder data/spc_r/kanton_be_grosser_rat/2018_06
```
- waits for each recorded batch id to complete and writes `*_batch_answer.jsonl` files next to the submissions.

### 4. Merge corrections back into the transcripts

```bash
uv run add_correction_to_file.py --folder_original_files data/spc_r/kanton_be_grosser_rat/2018_06
```
- reads the original JSON, matches by `custom_id`, and writes `*_corrected.json` containing `segments[<id>]["corrected_text"]`.
- these corrected files are the inputs for the judgement stage.

### 5. Judgement phase (quality scoring)

Repeat the same four steps with `--step judgement` and the judgement scripts:

1. Generate JSONL requests for already corrected files:
   ```bash
   uv run correct_transcriptions.py \
     --folder data/spc_r/kanton_be_grosser_rat/2018_06 \
     --pdf data/spc_r/kanton_be_grosser_rat/2018_06/tagblatt.pdf \
     --batch --step judgement \
     --judge_model gpt-4o-mini \
     --temperature 0.1
   ```
2. Submit with `uv run submit_judgement_job.py ...` (note the default glob `*_corrected_batch.jsonl`).
3. Retrieve results with `uv run retrieve_judgement_results.py ...`.
4. Merge answers into `*_corrected_judged.json` via `uv run add_judgement_to_file.py --folder_original_files ...`.

The resulting JSON files contain both `corrected_text` and `judgement` metadata for each segment, mirroring the released dataset.

### Synchronous processing (optional)

For small experiments you can bypass batches by omitting `--batch`:
```bash
uv run correct_transcriptions.py \
  --file data/spc_r/.../20180606_03_test.json \
  --pdf data/spc_r/.../tagblatt.pdf \
  --correction_model gpt-4o --judge_model gpt-4o-mini --temperature 0.1
```
The script will sequentially call the Chat Completions API for each segment, first for corrections, then for judgements, and write `<file>_corrected.json` once finished.

## File Placement Cheatsheet

- JSON transcripts and their derived files stay together. Batch scripts derive filenames automatically:
  - `*.json` → correction input
  - `*_batch.jsonl` → correction batch requests
  - `*_batch_answer.jsonl` → correction batch responses
  - `*_corrected.json` → transcript augmented with `corrected_text`
  - `*_corrected_batch.jsonl` → judgement batch requests
  - `*_corrected_batch_answer.jsonl` → judgement batch responses
  - `*_corrected_judged.json` → final output with scores
- Embeddings are cached in `embeddings/` (created on demand). You can delete the folder at any time to rebuild embeddings when the summary text changes.

## Troubleshooting

- **Missing dependencies**: re-run `uv sync`. `uv` keeps the lock file up to date whenever you change dependencies.
- **Rate limits / retries**: the batch helpers automatically back off using exponential waits when OpenAI returns rate-limit errors.
- **Incorrect file names**: ensure the patterns follow the dataset convention. `custom_id` is constructed from `<basename>.json__<segment_id>__<step>` so the merge utilities can map answers back to segments.
- **PDF OCR quirks**: if the toolkit saves `tagblatt.txt` next to the PDF you can edit it manually; the next run reuses the cached text.

## License

The code is released under the terms of the repository's `LICENSE`. Dataset licensing and responsible-use statements are documented on the [Hugging Face dataset card](https://huggingface.co/datasets/i4ds/spc_r).
