# Swiss-German Subtitle Correction

This project uses OpenAI's API to correct automatically generated Swiss-German subtitles based on a manually generated German summary.

## Features

- Creates an OpenAI assistant for subtitle correction
- Uses a vector store for efficient retrieval of relevant summary chunks
- Processes subtitle files (.srt) to correct Swiss-German to standard German
- Utilizes Retrieval-Augmented Generation (RAG) for context-aware corrections

## Requirements

- Python 3.x
- OpenAI API key
- Required Python packages: `openai`, `pysubs2`, `tqdm`

## Setup

1. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Install required packages:
   ```
   pip install openai pysubs2 tqdm
