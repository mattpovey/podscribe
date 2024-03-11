from libPodSemSearch import gpt_proc, check_key
from libPodSemSearch import setup_chromadb, query_chromadb

from config import \
    pod_prefix, \
    max_tokens, \
    oai_model, \
    chromadb_name, \
    chroma_collection, \
    collection_metadata

# pod_prefix="rihPodcast"
# max_tokens=240

# # Setup ChromaDB Variables
# chromadb_name = f'{pod_prefix}/chroma.db'
# chroma_collection = f'{pod_prefix}_{max_tokens}T_Collection'
# collection_metadata = {"hnsw:space": "cosine", "model.max_seq_length": max_tokens}

# The number of tokens represents the length of the quotes in chromedb and so affects the total 
# number of tokens in the query. It also determines the name of the collection.
if max_tokens == 240:
    num_results = 16
elif max_tokens == 120:
    num_results = 32
chromadb, collection = setup_chromadb(chromadb_name, chroma_collection, collection_metadata)
# print(chromadb.get_version())

# Load the API key
check_key()

# question = f'Can you explain the significance of the word, \"sacral\" to the podcast?'
# question = f'Can you explain the running joke about, \"the wrong shoes\"?'
# question = f'Explain the significance of the phrase, \"the wrong shoes\".'
# question = f'Who was responsible for building a search engine for the podcast?'
# question = f'What phrases similar to, \"on that bombshell\" have been used on the podcast?'
# question = f'What role has Belgium played in world events?'
# question = f'What was the role of England in the development of the modern irish state?'

# question = f'The hosts make a lot of impressions of historical figures. What is their favorite impression?'
# question = f'Which does the podcast have to say about Winston Churchill\'s record in India?'
# question = f'Which historical figures have the podcast hosts done impressions of?'
question = f'What do the podcast hosts think of Amsterdam?'
# question = f'What do the hosts think is the best way to teach history to children?'

# question = f'whose picture is in winston churchill\'s bedroom in Chartwell?'

# question = f'Do the hosts believe in historical determinism?'
# question = f'How do the hosts account for the supernatural in history?'
# question = f'What do the hosts think is the best way to teach history to children?'

# question = f'What does the podcast think about the relationship between england and france?'
# print(question)

# sys_role_content_query = f'You are an assistant which answers questions about a popular history podcast asked by fans of the podcast. You have access to a tool which lets you search transcripts of the podcast. You can use the tool to search the podcast transcripts for snippets of episodes that will help you to answer the question. Please supply up to 3 queries to help you answer the question.'
# sys_role_content_query = f'You are a podcaster responding to listener questions about your podcast. The listener has provided search results. Please respond to the question considering only the search results. The question is: {question}'
sys_role_content_query = f'You are responding to listener questions about a history focused podcast series. You have access to a tool which lets you search transcripts of the podcast. You can use the tool to search the podcast transcripts for snippets of episodes that will help you to answer the question. Please supply up to 3 queries to help you answer the question. When creating queries, consider that you are searching transcripts of podcast episodes so there is no need to mention the words podcast or episode.'
user_role_content_query = f'A fan of the podcast has asked, \"{question}\". Please tell me 3 simple queries, such as, "fall of the roman empire", that you would use to answer this query. Please supply each question on a new line with no additional text.'

user_role_content_query = f'\"{question}\". Please tell me 3 simple queries that you would use to answer the question. For example, to answer the question, "Why did the Roman Empire fall?, you might use the queries, "fall of the roman empire", "eastern roman empire" and, "end of the roman empire". Please supply each question on a new line with no additional commentary, text or punctuation. Queries should not be questions but formatted as semantic search queries.'

# queries = gpt_proc(sys_role_content_query, user_role_content_query, 'gpt-3.5-turbo')
queries = gpt_proc(sys_role_content_query, user_role_content_query, oai_model)

queries = queries.split('\n')
queries_cleaned = []
for i in queries:
    print(i)
    # check whether the query begins with a number
    if i[0].isdigit():
        # remove the number and the following space
        # print(i[3:])
        # Add i to the list queries_cleaned
        queries_cleaned.append(i[3:])
    # Remove any empty lines
    elif i == '':
        pass

print(queries_cleaned)

cq_results = []

# Query ChromaDB for each query
# Add the results to cq_results
for i in queries_cleaned:
    cq_results.append(query_chromadb(collection, str(i), num_results))

# Extract the episode number, title and quote from the cq_results into a new list
res_eps = []
for query in cq_results:
    # print(query)
    for result in query:
        # Add result to res_eps
        res_eps.append(f'{query[result][5]}')

sys_role_content_answer = f'You are responding to listener questions about a podcast. Please respond to the listener question considering only the content provided and do not use the names of the hosts. Write your answers in the third person. Your answer should focus solely on the answer to the question: "{question}".'
user_role_content_answer = f'{res_eps}'

# answer = gpt_proc(sys_role_content_answer, user_role_content_answer, 'gpt-3.5-turbo-16k')
answer = gpt_proc(sys_role_content_answer, user_role_content_answer, oai_model)
print(answer)

