#!/usr/bin/env python3
import argparse
import glob
import json
import os

import backoff
import numpy as np
from openai import APIError, APITimeoutError, OpenAI, RateLimitError  # type: ignore
from tqdm import tqdm

from utils import (
    extract_text_from_pdf,
    save_text_to_file,
    chunk_text,
    create_vector_store,
    prepare_messages,
)


client = OpenAI()


# --- API CALLS: CORRECTION & JUDGEMENT (Synchronous Mode) ---
@backoff.on_exception(
    backoff.expo, (RateLimitError, APIError, APITimeoutError), max_time=60
)
def chat_completion(messages, model, temperature):
    response = client.chat.completions.create(
        model=model, temperature=temperature, messages=messages
    )
    return response.choices[0].message.content


@backoff.on_exception(
    backoff.expo, (RateLimitError, APIError, APITimeoutError), max_time=60
)
def chat_completetion_logprobs_to_normalized_probs(
    messages, model, temperature, top_logprobs=None, logprobs=None
):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )
    return logprobs_to_normalized_probs(
        response.choices[0].logprobs.content[0].top_logprobs
    )


def logprobs_to_normalized_probs(top_logprobs):
    tokenprobs = {
        int(x.token): np.exp(x.logprob) for x in top_logprobs if x.token.isdigit()
    }
    probsum = np.sum(list(tokenprobs.values()))
    return {k: v / probsum for k, v in tokenprobs.items()}


# --- SEGMENT-PROCESSING FUNCTIONS (Synchronous Mode) ---
def process_segment_correction(
    seg_id, seg_data, embedding_model, faiss_index, text_chunks, model, temperature
):
    messages, relevant_chunks = prepare_messages(
        seg_data, "correction", embedding_model, faiss_index, text_chunks
    )
    corrected_text = chat_completion(messages, model, temperature)
    seg_data["corrected_text"] = corrected_text
    seg_data["relevant_chunks"] = "\n".join(relevant_chunks)
    return seg_id, seg_data


def process_segment_judgement(
    seg_id, seg_data, embedding_model, faiss_index, text_chunks, model, temperature
):
    messages, _ = prepare_messages(
        seg_data, "judgement", embedding_model, faiss_index, text_chunks
    )
    comment = chat_completetion_logprobs_to_normalized_probs(
        messages, model, temperature, top_logprobs=3, logprobs=True
    )
    seg_data["rating"] = comment
    return seg_id, seg_data


def process_segments_sequential(process_func, segments, **kwargs):
    updated_segments = {}
    for seg_id, seg_data in tqdm(
        segments.items(), desc=process_func.__name__, total=len(segments)
    ):
        updated_id, updated_data = process_func(seg_id, seg_data, **kwargs)
        updated_segments[updated_id] = updated_data
    return updated_segments


# --- BATCH JOB GENERATION FUNCTIONS ---
def generate_batch_job(
    json_file,
    embedding_model,
    faiss_index,
    text_chunks,
    model,
    temperature,
    step,
    output_batch_file,
):
    """
    Generate a JSONL file with one job per segment.
    Each job is a request payload for the Batch API (/v1/chat/completions).
    """
    with open(output_batch_file, "w", encoding="utf-8") as f_out:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", {})
        for seg_id, seg_data in segments.items():
            messages, _ = prepare_messages(
                seg_data, step, embedding_model, faiss_index, text_chunks
            )
            custom_id = f"{os.path.basename(json_file)}__{seg_id}__{step}"
            job = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                },
            }
            if step == "judgement":
                job["body"]["logprobs"] = True
                job["body"]["top_logprobs"] = 3
            f_out.write(json.dumps(job) + "\n")
    print(f"Batch job file generated: {output_batch_file}")


# --- JSON & SRT HANDLING (Synchronous Mode) ---
def process_json_file(
    json_file_path,
    embedding_model,
    faiss_index,
    text_chunks,
    correction_model,
    judge_model,
    temperature,
):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", {})
    # Correction step.
    segments = process_segments_sequential(
        process_segment_correction,
        segments,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        text_chunks=text_chunks,
        model=correction_model,
        temperature=temperature,
    )
    # Judgement step.
    segments = process_segments_sequential(
        process_segment_judgement,
        segments,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        text_chunks=text_chunks,
        model=judge_model,
        temperature=temperature,
    )
    data["segments"] = segments
    base, _ = os.path.splitext(json_file_path)
    out_json = base + "_corrected.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Processed and saved: {out_json}")


# --- MAIN ENTRY POINT ---
def main():
    parser = argparse.ArgumentParser(
        description="Transcription correction pipeline using summary PDF and JSON transcripts."
    )
    parser.add_argument(
        "--folder",
        help="Folder containing JSON files with transcription segments.",
    )
    parser.add_argument(
        "--file",
        help="Single JSON file with transcription segments.",
    )
    parser.add_argument(
        "--pdf", required=True, help="PDF file from which the summary is extracted."
    )
    parser.add_argument(
        "--correction_model",
        default="gpt-4o",
        help="OpenAI model to use for correction completions.",
    )
    parser.add_argument(
        "--judge_model",
        default="gpt-4o-mini",
        help="OpenAI model to use for judgement completions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the OpenAI API calls.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate batch jobs instead of processing synchronously.",
    )
    parser.add_argument(
        "--step",
        choices=["correction", "judgement"],
        default="correction",
        help="Step for which to generate batch jobs (default: correction).",
    )
    args = parser.parse_args()

    if args.file and args.folder:
        print("Please provide either --file or --folder, not both.")
        return
    if not args.file and not args.folder:
        print("Please provide either --file or --folder.")
        return

    # Extract summary text from PDF.
    print("Extracting summary from PDF...")
    summary_txt_path = os.path.splitext(args.pdf)[0] + ".txt"
    if os.path.exists(summary_txt_path):
        with open(summary_txt_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
    else:
        summary_text = extract_text_from_pdf(args.pdf)
        save_text_to_file(summary_text, summary_txt_path)
    print(f"Summary saved to {summary_txt_path}")

    text_chunks = chunk_text(summary_text, chunk_size=600, overlap=450)
    embedding_model, faiss_index = create_vector_store(text_chunks)

    if args.file:
        json_files = [args.file]
    if args.folder:
        json_files = glob.glob(os.path.join(args.folder, "*.json"))
    if not json_files:
        print("Keine JSON-Dateien im angegebenen Ordner gefunden.")
        return

    if args.batch:
        # For judgement step, process only already corrected files.
        if args.step == "judgement":
            json_files = [f for f in json_files if f.endswith("corrected.json")]
        # For correction step, skip files that are already corrected.
        if args.step == "correction":
            json_files = [
                f
                for f in json_files
                if not f.endswith("corrected.json") and not f.endswith("judged.json")
            ]
        else:
            raise ValueError("Invalid step specified. Use 'correction' or 'judgement'.")
        for json_file in json_files:
            output_batch_file = json_file.replace(".json", "_batch.jsonl")
            generate_batch_job(
                json_file,
                embedding_model,
                faiss_index,
                text_chunks,
                args.correction_model if args.step == "correction" else args.judge_model,
                args.temperature,
                args.step,
                output_batch_file,
            )
        print(
            "Batch file generated. Exiting the script. Please submit the batch file via the OpenAI Batch API."
        )
        return

    # Synchronous processing:
    for json_file in json_files:
        if json_file.endswith("corrected.json") and not args.file:
            print(f"Überspringe bereits korrigierte Datei: {json_file}")
            continue
        print(f"Verarbeite {json_file}...")
        process_json_file(
            json_file,
            embedding_model,
            faiss_index,
            text_chunks,
            args.correction_model,
            args.judge_model,
            args.temperature,
        )


if __name__ == "__main__":
    main()
