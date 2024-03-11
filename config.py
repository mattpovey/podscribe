# PODCAST
pod_prefix="[PODCAST_TL_DIR]"
feed_url="[FEED_URL]"

# There are 3 transcription modes, diarized, single and both. Diarized uses the 
# small-tdrz model to generate speaker turns. Single, uses the whisper_model model
# to transcribe the episode as a single speaker. Both, creates two transcripts
tscript_mode="single"
# whisper_model can take any whisper model as its parameter. It does not affect the 
# diarization model which is always small-tdrz.ggml. Models are downloaded with the 
# ggml-download scripts in the models subdirecory of whisper.cpp
whisper_model="medium.en"

# EMBEDDINGS
# Different lengths of max_tokens give different results in searches. The goal is to capture as much 
# meaning in each chunk without having too many different ideas in each chunk.
# TODO: Send preceding and following chunks to LLMs 
max_tokens=240
# max_tokens=120

# LLM system selection
mistral_model="mistral-medium"
oai_model="gpt-4"

# OpenAI parameters
# embeds_model="text-embeddings-ada-002"

# ChromaDB Built In embedding model
embeds_model="all-MiniLM-L6-v2"

# Mistral
# TBD

# Test model to test OpenAI API Key
test_model="gpt3.5-turbo" # used to check validity of API key

# Used in calculating the cost of embeddings. Not relevant if using local model
embeds_model_cost = 0 # per 1K tokens

# DIRECTORIES
pod_dir=f'{pod_prefix}/audio'
tscript_dir=f'{pod_prefix}/transcripts'
csv_dir=f'{pod_prefix}/csv'
save_dir=f'{pod_prefix}/audio'
episode_dir=f'{pod_prefix}/audio'
wav_dir=f'{pod_prefix}/wav'
csv_dir=f'{pod_prefix}/csv'
transcription_dir=f'{pod_prefix}/transcripts'

# FILENAMES
metadata_csv=f'{csv_dir}/episodeMetadata.csv'
raw_csv=f'{csv_dir}/raw.csv'
chunks_csv=f'{csv_dir}/chunks_{str(max_tokens)}T.csv'
embeds_csv=f'{csv_dir}/embeds{str(max_tokens)}T.csv'
final_csv=f'{csv_dir}/embeds_url_{str(max_tokens)}T.csv'

# CHROMA
chromadb_name=f'{pod_prefix}/chroma.db'
chroma_collection=f'{pod_prefix}_{max_tokens}T_Collection'
collection_metadata={"hnsw:space": "cosine", "model.max_seq_length": max_tokens}




