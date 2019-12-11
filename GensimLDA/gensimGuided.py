import gensim
import pandas as pd
from bengali_stemmer.rafikamal2014 import RafiStemmer
import docx
import string

FILE_NAME = r"Datasets/thesis_dataset_kalerkantho_new.csv"
STOP_WORD_LIST_DIR = r"Stopwords/StopWords.docx"

def prepare_bag_of_words(processed_docs, dictionary):
    return [dictionary.doc2bow(doc) for doc in processed_docs]

def read_doc_as_pandasDF(filename):

    data = pd.read_csv(filename, error_bad_lines=False)
    data_text = data[['content']]

    data_text['index'] = data_text.index
    documents = data_text

    return documents

def load_stop_word():

    global STOP_WORD_LIST_DIR

    stop_directory = STOP_WORD_LIST_DIR

    doc = docx.Document(stop_directory)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)

    bengali_stop_words = fullText[0].split()
    bengali_stop_words = frozenset(bengali_stop_words)
    #print(bengali_stop_words)

    return bengali_stop_words

def preprocess_documents(doc):

    preprocessed_list_of_docs = []
    stemmer = RafiStemmer()

    stop_words = load_stop_word()
    preprocessed_docs = []

    doc_token = []

    if isinstance(doc, str):
        for token in punctuation_remover(doc).split():
            if token not in stop_words and len(token) >= 3:
                if len(stemmer.stem_word(token)) >= 2:
                    doc_token.append(stemmer.stem_word(token))


    return doc_token

def punctuation_remover(text):
    BENGALI_PUNCTUATION = string.punctuation + "—।’‘"
    BENGALI_NUMERALS = "০১২৩৪৫৬৭৮৯"
    return text.translate(str.maketrans(' ', ' ', BENGALI_PUNCTUATION+BENGALI_NUMERALS))

if __name__ == '__main__':
    pd_document = read_doc_as_pandasDF(FILE_NAME)
    smaller_documents = pd_document

    #smaller_documents = pd_document[:20]

    # processed_docs after stop word removal && length

    #print(smaller_documents['content'][2])
    processed_docs = smaller_documents['content'].map(preprocess_documents)

    print("Printing Processed Docs\n")
    print(processed_docs)

    print('\n Preparing and showing dictionary \n')

    # Adjust no_below and no_above
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)

    print(dictionary)

    bow_corpus = prepare_bag_of_words(processed_docs, dictionary)

    print(" bow corpus number 5 entry:", end=" ")
    print(bow_corpus[5])

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary, passes=2, workers=3)

    print("LDA Model:")

    for idx in range(3):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

    print("=" * 20)