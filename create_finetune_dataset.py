#!/usr/bin/env python3
import os
from utils import (
    prepare_finetune_data, 
    save_finetune_data,
    extract_text_from_pdf,
    chunk_text,
    create_vector_store
)

# Input paths
json_file = "/mnt/nas05/data01/sg_data/swiss_parliament/kanton_be_grosser_rat/crawl_20200107/orig/2017_03/20170608_02_corrected_judged.json"
pdf_file = os.path.join(os.path.dirname(json_file), "tagblatt.pdf")

# Output directory for fine-tuning files
output_dir = "finetune_data"
os.makedirs(output_dir, exist_ok=True)

# Extract and chunk text from PDF
print("Extracting text from PDF...")
text = extract_text_from_pdf(pdf_file)
text_chunks = chunk_text(text)

# Create vector store
print("Creating vector store...")
embedding_model, faiss_index = create_vector_store(text_chunks)

# Prepare and save the fine-tuning data
print("Preparing fine-tuning data...")
finetune_data = prepare_finetune_data(json_file, embedding_model, faiss_index, text_chunks)
output_path = os.path.join(output_dir, "transcription_quality_finetune.jsonl")
save_finetune_data(finetune_data, output_path)

print(f"Created fine-tuning dataset with {len(finetune_data)} examples")
print(f"Dataset saved to: {output_path}")
