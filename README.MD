# PodScribe
Transcribe podcast episodes using OpenAI whisper.cpp. Once fetched and transcribed, the transcripts can be processed into a ChromaDB vector DB for semantic search. Optionally ask questions about the podcast using OpenAI GPT or Mistral. Implements a simple retrieval augmented generation pipeline without using LangChain or other frameworks as a learning exercise. 

## Installation

Download whisper.cpp into a subdirectory (called whisper.cpp) and compile for your platform. Including GPU or CoreML support depending on what you're running on will speed things up dramatically. Download the models you plan to use using the scripts in the models subdirectory of whisper.cpp and configure those in the config.py file. 

## Configuration

1. Create a venv and activate
2. Install requirements
```bash
pip install -r ./requirements.txt
```
3. Edit the config file to change the podcast directory and add the RSS feed for the podcast

## fetchtoTscript.py

Provided the configuration is done, running will download the available podcast episodes, transcribe them and create a metadata file of all episodes, their titles and episode numbers. Metadata and filenames are generated from the RSS XML so should work across different podcasts. 

Episodes are converted to wav format using ffmpeg, then transcribed. The transcription can be configured using the ```tscript_mode``` variable which is in the config file. Documentation of that and the ```whisper_model``` variable are inline in the config. 

## podtoVector.py

This will take the transcripts, chunk the text according to the ```max_tokens``` size set in the configuration and generate embeddings which are stored in a ChromaDB instance. The embeddings model can be configured in config.py by changing ```embeds_model```. 

TODO: FIX INCREMENTAL UPDATES OF CHROMADB - CURRENTLY INSERTS DUPELICATES

## podSemSearch.py

Allows for simple tests of semantic search to be performed against the ChromaDB instance. Edit the search term directly in the script.

## pod[Mistral|GPT]question.py

Implements a simple RAG based Q&A using the ChromaDB embeddings and metadata. Allows for testing of different models. Requires an API key to be entered at run-time.