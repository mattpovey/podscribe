from libPodSemSearch import gpt_proc, check_key
from libPodSemSearch import setup_chromadb, query_chromadb

from config import \
    pod_prefix, \
    max_tokens, \
    mistral_model, \
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

# question = f'Can you explain the significance of the word, \"sacral\" to the podcast?'
# question = f'Can you explain the running joke about, \"the wrong shoes\"?'
question = f'Explain the significance of the phrase, \"the wrong shoes\".'
# question = f'Who was responsible for building a search engine for the podcast?'
# question = f'What phrases similar to, \"on that bombshell\" have been used on the podcast?'
# question = f'What role has Belgium played in world events?'
# question = f'What was the role of England in the development of the modern irish state?'
# question = f'The hosts make a lot of impressions of historical figures. What is their favorite impression?'
# question = f'Which does the podcast have to say about Winston Churchill\'s attitude to India?'
# question = f'Which historical figures have the podcast hosts done impressions of?'
# question = f'What do the podcast hosts think of Amsterdam?'
# question = f'What do the hosts think is the best way to teach history to children?'
# question = f'whose picture is in winston churchill\'s bedroom in Chartwell?'
# question = f'Do the hosts believe in historical determinism?'
# question = f'How do the hosts account for the supernatural in history?'
# question = f'What do the hosts think is the best way to teach history to children?'
# question = f'What was the reason for the Reichstag fire?'
# question = f'Who do the hosts believe was England\'s greatest Prime Minister?'
# print(question)

# sys_role_content_query = f'You are an assistant which answers questions about a popular history podcast asked by fans of the podcast. You have access to a tool which lets you search transcripts of the podcast. You can use the tool to search the podcast transcripts for snippets of episodes that will help you to answer the question. Please supply up to 3 queries to help you answer the question.'
# sys_role_content_query = f'You are a podcaster responding to listener questions about your podcast. The listener has provided search results. Please respond to the question considering only the search results. The question is: {question}'
#sys_role_content_query = f"""You are responding to listener questions about a history focused podcast series. You have access to a tool which lets you search transcripts of the podcast. You can use the tool to search the podcast transcripts for snippets of episodes that will help you to answer the question. Please supply up to 3 queries to help you answer the question. When creating queries, consider that you are searching transcripts of podcast episodes so there is no need to mention the words podcast or episode."""
sys_role_content_query = f"""You are responding to listener questions about a history focused podcast series. 
You have access to a tool which lets you search transcripts of the podcast. 
You can use the tool to search the podcast transcripts for snippets of episodes that will help you to answer the question. 
Supply 3 queries to help you answer the question. The queries should focus on the subject only.
Queries should not be posed as questions but formatted as semantic search queries.
For example, to answer the question, "Why did the Roman Empire fall?, you might use the queries:
fall of the roman empire
eastern roman empire
end of the roman empire
Please supply each question on a new line.
It is very important that you provide only 3 queries. 
Do not include any other commentary, notes, text or punctuation, regardless of the subject. 
"""

user_role_content_query = question

# queries = gpt_proc(sys_role_content_query, user_role_content_query, 'gpt-3.5-turbo')
queries = gpt_proc(sys_role_content_query, user_role_content_query, mistral_model)

queries = queries.split('\n')
print(queries)
#queries_cleaned = []
print("Cleaning the queries...")
# for i in queries:
#     print(i)
#     # check whether the query begins with a number
#     if i[0].isdigit():
#         # remove the number and the following space
#         # print(i[3:])
#         # Add i to the list queries_cleaned
#         queries_cleaned.append(i[3:])
#     # Remove any empty lines
#     elif i == '':
#         pass
#     else:
#         # Add i to the list queries_cleaned
#         queries_cleaned.append(i)


# print("The cleaned queries are as follows: ", queries_cleaned)
cq_results = []

# Query ChromaDB for each query
# Add the results to cq_results
for i in queries:
    cq_results.append(query_chromadb(collection, str(i), num_results))

# print("The Chroma Query returned: ", cq_results)
# input("Press Enter to continue...")

# Extract the episode number, title and quote from the cq_results into a new list
res_eps = []
for query in cq_results:
    # print(query)
    for result in query:
        # Add result to res_eps
        res_eps.append(f'Episode: {query[result][3]}, Quote: {query[result][5]}')

# user_role_content_query = f"""A fan of the podcast has asked, "{question}". Please tell me 3 simple queries, such as, "fall of the roman empire", that you would use to answer this query. Please supply each question on a new line with no additional text."""
user_role_content_answer = f""""{question}". Please tell me 3 simple queries that you would use to answer the question."""

# The actual query prompts using the semantic results
sys_role_content_answer = f"""You are responding to listener questions about a history focused podcast series. 
The user has used a tool to search the transcripts of over 500 episodes of the podcast.
The listener will ask you a question and provide the search results.
Please respond to the listener question considering only the search results.
Do not include your general knowledge in the answer and do not comment beyond answering the question.
Try to include the names of the hosts in the answer and what they said in the answer.
Each search result includes the title of the episode from which the episode is drawn.
Ensure that you refer to the episode titles in your answer so that the listener can find the relevant episodes
to support your answer. 
"""
user_role_content_answer = f"""The question is: "{question}": The search results are: {res_eps}"""

# sys_role_content_answer = f'You are responding to listener questions about a podcast. Please respond to the listener question considering only the content provided and do not use the names of the hosts. Write your answers in the third person. Your answer should focus solely on the answer to the question: "{question}".'
# user_role_content_answer = f'{res_eps}'

# answer = gpt_proc(sys_role_content_answer, user_role_content_answer, 'gpt-3.5-turbo-16k'
answer = gpt_proc(sys_role_content_answer, user_role_content_answer, mistral_model)
print(answer)

