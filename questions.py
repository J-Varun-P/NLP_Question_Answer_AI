import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary = dict()
    for x in os.listdir('corpus'):
        file_path = os.path.join('corpus', x)
        f = open(file_path, 'r', encoding="utf8")
        dictionary[x] = f.read()
        f.close()
    return dictionary
    #raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    n = len(words)
    for i in range(n):
        words[i] = words[i].lower()
    i = 0
    x = string.punctuation
    y = nltk.corpus.stopwords.words("english")
    while i < n:
        if words[i] in x:
            words.remove(words[i])
            i -= 1
            n -= 1
        elif words[i] in y:
            words.remove(words[i])
            i -= 1
            n -= 1
        i += 1
    return words
    #raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    n1 = len(documents)
    for i in documents:
        for j in documents[i]:
            n2 = 0
            for x in documents:
                if x != i:
                    if j in documents[x]:
                        n2 += 1
            n2 += 1
            idfs[j] = math.log(n1/n2)
    return idfs
    #raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top = dict()
    for x in files:
        sum = 0
        for y in query:
            n1 = 0
            for i in files[x]:
                if i == y:
                    n1 += 1
            sum = sum + (n1 * idfs[y])
        top[x] = sum
    top = sorted(top.items(), key=lambda item: item[1], reverse=True)
    top_n = []
    i = 0
    for x in top:
        if i == n:
            break
        top_n.append(x[0])
        i += 1
    return top_n
    #raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top = dict()
    for x in sentences:
        sum = 0
        temp = []
        for temp1 in query:
            temp.append(temp1)
        for y in sentences[x]:
            if y in temp:
                sum += idfs[y]
                temp.remove(y)
        top[x] = sum
    top = sorted(top.items(), key=lambda item: item[1], reverse=True)
    top_n = []
    i = 0
    for x in top:
        if i == n:
            break
        top_n.append(x[0])
        i += 1
    if len(top) > 1:
        clash = False
        if top[n][1] == top[n-1][1]:
            clash = True
        if clash == False:
            return top_n
        else:
            temp = dict()
            top_n2 = []
            for x in top:
                if x[1] != top[n-1][1]:
                    top_n2.append(x[0])
                elif x[1] == top[n-1][1]:
                    break
            m1 = n - len(top_n2)
            for x in top:
                if x[1] == top[n-1][1]:
                    z = x[0].split()
                    n1 = len(z)
                    n2 = 0
                    for y in z:
                        if y.lower() in query:
                            n2 += 1
                    temp[x[0]] = n2 / n1
                elif x[1] < top[n-1][1]:
                    break
            temp = sorted(temp.items(), key=lambda item: item[1], reverse=True)
            i = 0
            for x in temp:
                if i == m1:
                    break
                top_n2.append(x[0])
                i += 1
            return top_n2
    else:
        return top_n
    #raise NotImplementedError


if __name__ == "__main__":
    main()
