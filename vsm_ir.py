import sys
from retrieval import *

if sys.argv[1] == "create_index":
    xml_path = sys.argv[2]
    build_index(xml_path)

elif sys.argv[1] == "query":
    index_path = sys.argv[2]
    query = sys.argv[3]
    index = get_inverted_index(index_path)
    retrieved_doc_scores, token_weights = create_scored_documents(query=query, index=index)
    compute_cossim(doc_scores=retrieved_doc_scores, token_weights=token_weights, index=index)
