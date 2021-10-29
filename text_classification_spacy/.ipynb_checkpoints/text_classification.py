import spacy
# tqdm is a great progress bar for python
# tqdm.auto automatically selects a text based progress
# for the console
# and html based output in jupyter notebooks
from tqdm.auto import tqdm
# DocBin is spacys new way to store Docs in a
# binary format for training later
from spacy.tokens import DocBin
# We want to classify movie reviews as positive or negative
# http://ai.stanford.edu/~amaas/data/sentiment/
from ml_datasets import imdb
# load movie reviews as a tuple (text, label)
train_data, valid_data = imdb()
# load a medium sized english language model in spacy
nlp = spacy.load(“en_core_web_md”)

def make_docs(data):
    """
    this will take a list of texts and labels
    and transform them in spacy documents

    data: list(tuple(text, label))

    returns: List(spacy.Doc.doc)
    """

    docs = []
    # nlp.pipe([texts]) is way faster than running
    # nlp(text) for each text
    # as_tuples allows us to pass in a tuple,
    # the first one is treated as text
    # the second one will get returned as it is.

    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total=len(data)):
        # we need to set the (text)cat(egory) for each document
        doc.cats["positive"] = label

        # put them into a nice list
        docs.append(doc)

    return docs


