from create_index import *

LIMIT = 50

def get_inverted_index(file: str):
    """ Get inverted index from json file """
    with open(file, 'r') as j:
        json_data = json.load(j)
    return json_data

def create_scored_documents(query: str, index: dict):
    """ Store retrieved documents with scores """
    doc_scores = {}
    token_weights = {}
    index_idf = index['idf']
    index_tf = index['tf']
    query_tf = get_tf(query)
    for token in query_tf.keys():
        token_idf = index_idf.get(token, 0.0)
        token_tf = query_tf[token]
        token_weights[token] = token_tf * token_idf
        token_occurences = index_tf.get(token, {})
        for doc in token_occurences.keys():
            doc_tf = token_occurences[doc]
            if doc not in doc_scores.keys():
                doc_scores[doc] = 0.0
            doc_scores[doc] += token_weights[token] * token_idf * doc_tf
    return doc_scores, token_weights

def compute_cossim(doc_scores: dict, token_weights: dict, index: dict):
    """ Compute cossim of retrieved documents and write to file """
    final_scores = {}
    index_len = index['vec_len']
    query_L = sqrt(sum(pow(weight, 2) for weight in token_weights.values()))
    for doc in doc_scores.keys():
        score = doc_scores[doc]
        doc_len = index_len[doc]
        final_score = score / (query_L * doc_len)
        final_scores[doc] = final_score
    sorted_docs = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    text_file = open("ranked_query_docs.txt", "w")
    text_file.writelines("\n".join([i[0] for i in sorted_docs][:LIMIT]))
    text_file.close()

