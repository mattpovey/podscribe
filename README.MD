# PodScribe
Transcribe podcast episodes using OpenAI whisper.cpp. Once fetched and transcribed, the transcripts can be processed into a ChromaDB vector DB for semantic search. Optionally ask questions about the podcast using OpenAI GPT or Mistral. Implements a simple retrieval augmented generation pipeline without using LangChain or other frameworks as a learning exercise. 

## Installation

Download whisper.cpp into a subdirectory (called whisper.cpp) and compile for your platform. Including GPU or CoreML support depending on what you're running on will speed things up dramatically. 

## Configuration

1. Create a venv and activate
2. Install requirements
```bash
pip install -r ./requirements.txt
```
3. Edit the config file to change the podcast directory and add the RSS feed for the podcast

## fetchtoTscript.py

Provided the configuration is done, running will download the available podcast episodes, transcribe them and create a metadata file of all episodes, their titles and episode numbers. Metadata and filenames are generated from the RSS XML so should work across different podcasts. 

4th October 2023

podtoVector does not currently do updates correctly. As things stand, the raw and chunks csv files as well as the chroma.db directory must be deleted before updating. A full re-index is necessary

fetchtoTscript.py works correctly and incrementally

whoosh_index.py updates the index files correctly although for some reason they have shrunk considerably.

Need to integrate the chromadb search functionaligy from podSemSearch.py into the podSearch script. The required functions have been copied over to lib_podsearch.py.
The code from podSemSearch.py needs to be integrated now. 