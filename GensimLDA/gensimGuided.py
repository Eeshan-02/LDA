import gensim
import pandas as pd
from bengali_stemmer.rafikamal2014 import RafiStemmer
import docx
import string
import numpy as np

FILE_NAME = r"Datasets/thesis_dataset_prothomalo.csv"
STOP_WORD_LIST_DIR = r"Stopwords/StopWords.docx"


def boost_words_in_eta(eta, dictionary, boost_words):
    for i, word_list in enumerate(boost_words):
        boost_word_id_list = []
        print("{} is in {}".format(i, word_list))


        uniq, count = np.unique(eta, return_counts=True)
        print(dict(zip(uniq, count)))

        #print([list(dictionary.keys())[list(dictionary.values()).index(each_word)] for each_word in word_list])

        for each_word in word_list:

            try:
                word_id = list(dictionary.keys())[list(dictionary.values()).index(each_word)]
                print(word_id)
                boost_word_id_list.append(word_id)

            except ValueError:
                print("{} Not found in Dictionary ".format(each_word))

        for word in boost_word_id_list:
            eta[i][word] = 0.75

    uniq, count = np.unique(eta, return_counts=True)
    print(dict(zip(uniq, count)))

    return eta

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

    NUM_TOPICS = 3

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
    dictionary.filter_extremes(no_below=.10, no_above=0.5, keep_n=100000)

    print(dictionary)

    bow_corpus = prepare_bag_of_words(processed_docs, dictionary)

    print(" bow corpus number 5 entry:", end=" ")
    print(bow_corpus[5])

    eta = np.ones((NUM_TOPICS, len(dictionary))) * 0.1
    print("Shape of eta is {}".format(eta.shape))
    print("type of dictionary {}".format(type(dictionary)))
    print(eta)

    boost_words =[
        ['অধিনায়ক', 'মেসি', 'ক্রিকেট', 'জাতীয়', 'রান', 'উইকেট'],
        ['মূলধন', 'বিনিয়োগ', 'শেয়ার', 'আইন', 'টাকা', 'শতাংশ', 'ব্যাংক', 'কো', 'পণ্য']

    ]
    boosed_eta = boost_words_in_eta(eta, dictionary, boost_words)


    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=2, workers=3)

    print("LDA Model:")

    for idx in range(NUM_TOPICS):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

    print("=" * 20)

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=2, workers=3, eta=eta)

    print("LDA Model:")

    for idx in range(NUM_TOPICS):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

    print("=" * 20)

