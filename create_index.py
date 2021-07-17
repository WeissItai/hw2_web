
import os
import json
from math import log, sqrt
import xml.etree.ElementTree as ET
from process_language import tokenize
from collections import defaultdict, Counter


# Extract relevant text for each document in given file.
# Maps doc_id to text
def parse_xml(fname):

    data = {}

    tree = ET.parse(fname)
    root = tree.getroot()
    for record in root.findall("./RECORD"):
        doc_id = int(record.find("./RECORDNUM").text)

        # add title
        title = record.find("./TITLE").text
        data[doc_id] = title

        # add extract
        extract = getattr(record.find("./ABSTRACT"), 'text', '')
        data[doc_id] += '\n' + extract

        # add abstract
        abstract = getattr(record.find("./EXTRACT"), 'text', '')
        data[doc_id] += '\n' + abstract

    # p = pprint.PrettyPrinter()
    # p.pprint(data)
    return data


# Create index and write to json file
def create_corpus(dir_path):

    corpus = {}

    for filename in os.listdir(dir_path):
        if filename.endswith(".xml"):
            data = parse_xml(os.path.join(dir_path, filename))
            corpus.update(data)

    return corpus


def get_tf(text):

    tokens = tokenize(text)
    c = Counter()
    c.update(tokens)
    return c


def build_index(dir_path):

    tf = defaultdict(dict)
    idf = defaultdict(lambda: 0.0)
    lengths = defaultdict(lambda: 0.0)

    num_docs = 0

    corpus = create_corpus(dir_path)

    for doc_id, text in corpus.items():

        # Compute TF
        c = get_tf(text)
        max_word = c[max(c, key=c.get)]

        for term, cnt in c.items():
            tf[term][doc_id] = cnt / max_word
            idf[term] += 1

        num_docs += 1

    # Compute IDF
    for term in tf.keys():
        idf[term] = log(num_docs / idf[term])

    # Compute vector lengths
    for token, scores in tf.items():
        for doc_id, score in scores.items():
            lengths[doc_id] += (score * idf[token]) ** 2

    for doc_id in lengths.keys():
        lengths[doc_id] = sqrt(lengths[doc_id])

    data = {
        'vec_len': lengths,
        'tf': tf,
        'idf': idf
    }

    with open('vsm_inverted_index.json', 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    build_index("C:\Dev\hw2_web\cfc-xml_corrected")