from math import sqrt
import random
import os
import json
import codecs

import argparse
# n-gram length should be 4 as mentioned in the paper
nl = 4
# according to the paper larger feature sets yeilded greater accuracy
# therefore we take feature list value as 30000
feat_list = 100000
# if the score is greater than threshold (t)the text was written by the author else
# else none
t = 0.5
# according to the paper, algorithm must repeat kl times. kl= 100 initially
kl = 100
# minimum training length to eliminate class imbalance case
min_length = 500
# these variables are declared because they are used in the json file of the dataset
encoding = ""
language = ""
path = ""
# path for unknown text
unknown_path = ""
# authors list extracted from json file
authors = []
# unknown text list extracted from json file
unknowns = []
# list of trianing files for each candidate
training_files_list = {}

# finds similar features between fl and random string s
def add_similar_features(s, fl):
    new_vector = {}
    converted = convert_to_vector(s)
    for w in fl:
        if w in converted:
            new_vector[w] = converted[w]
    return new_vector

# converts a string to vector
def convert_to_vector(s):
    store = {}
    words = s.split()
    for w in words:
        if len(w) <= nl:
            add(store, w)
        else:
            #seperate each word with respect to n-gram length
            for i in range(len(w) - nl + 1):
                add(store, w[i:i + nl])
    return store
 # check if word is present, if present add 1 else 1
def add(store, w):
    if w in store:
        store[w] += 1
    else:
        store[w] = 1
# training the corpus
def c_training(s):
    print("started training...")
    vector = convert_to_vector(s)
    print("started selecting features...")
    # select most frequent features
    fl = sorted(vector, key=vector.get, reverse=True)[:min(len(vector), feat_list)]
    print("done")
    return fl

# cosine similarity is calculated for the two vectors vec1
def cosinesimilarity(s1, s2, fl):
    vec1 = add_similar_features(s1, fl)
    vec2 = add_similar_features(s2, fl)
    sp = float(0)
    length_vec1 = 0
    length_vec2 = 0
    for w in vec1:
        length_vec1 += vec1[w] ** 2
    for w in vec2:
        length_vec2 += vec2[w] ** 2
    length_vec1 = sqrt(length_vec1)
    length_vec2 = sqrt(length_vec2)
    for ngram in vec1:
        if ngram in vec2:
            sp += vec1[ngram] * vec2[ngram]
    return sp / (length_vec1 * length_vec2)

# random string is generated in the known text for similairty measurement
def generate_random_string(s, length):
    words = s.split()
    r = random.randint(0, len(words) - length)
    return "".join(words[r:r + length])

# process json file and assign it to the declared variables
def process_json_file(corpus):
    global path, unknown_path, authors, unknowns, encoding, language
    path += corpus
    # in the path search for the file name meta-file.json
    file = open(os.path.join(path, "meta-file.json"), "r")
    # load the file
    insidejson = json.load(file)
    file.close()

    unknown_path += os.path.join(path, insidejson["folder"])
    encoding += insidejson["encoding"]
    language += insidejson["language"]
    authors += [author["author-name"]
                   for author in insidejson["candidate-authors"]]
    unknowns += [text["unknown-text"] for text in insidejson["unknown-texts"]]

    # process training text so that its text can be analysed
def load_training_text(cand, fname):
    dfile = codecs.open(os.path.join(path, cand, fname), "r", "utf-8")
    s = dfile.read()
    dfile.close()
    return s
    # process unknown text so it can be analysed
def load_unknown_text(f):
    unknown_file = codecs.open(os.path.join(unknown_path, f), "r", "utf-8")
    utxt = unknown_file.read()
    unknown_file.close()
    return utxt

def main():
    global authors, unknowns
    location = "C:\\Users\\vicky\\Desktop\\pan12-authorship-attribution-test-dataset-problem-c-2015-10-20"
    authors = authors
    unknowns = unknowns
    # loads the json file for processing
    process_json_file(location)
    # updates training file list, os.walk helps in finding the subdirectories
    for a in authors:
        training_files_list[a] = []
        for subdir, dirs, files in os.walk(os.path.join(location, a)):
            for doc in files:
                training_files_list[a].append(doc)

    # the works of each author are stored
    works = {}
    # corpus is created for storing all the workds
    corpus = ""
    # texts with less than minimum length are deleted
    deletes = []
    # for each author, training text is loaded and added to works , else they are added to corpus.
    # delete those authors whose work is less than that of minimum length.
    for a in authors:
        works[a] = ""
        for file in training_files_list[a]:
            works[a] += load_training_text(a, file)
        print("all texts from " + a + " were read")
        if len(works[a].split()) < min_length:
            del works[a]
            deletes.append(a)
        else:
            corpus += works[a]
    # refreshing the list of authors by removing those works of authors whose length is less
    # than the minimum length
    newcands = []
    for a in authors:
        if a not in deletes:
            newcands.append(a)
    authors = newcands
    words = [len(works[a].split()) for a in works]
    min_words = min(words)

    # create a feature list by training the corpus
    fl = c_training(corpus)
    # store authors of unknown texts
    u_author = []
    scores = []

    for f in unknowns:
        print("testing " + f)
        # process unknown text files
        unknown_text = load_unknown_text(f)
        unknown_length = len(unknown_text.split())
        # initialise winning score
        winning_score = [0] * len(authors)
        # restrict the scope of the text to be analysed
        text_length = min(unknown_length, min_words)
        # converts characters to string and joins them till text length
        unknown_string = "".join(unknown_text.split()[:text_length])
        # repeat the algorithm kl times
        for i in range(kl):
            rfl = random.sample(fl, len(fl) // 3) #randomly choose feature according algorithm step 1 (a)
            sims = []
            for a in authors:
                astring = generate_random_string(works[a], text_length)
                sims.append(cosinesimilarity(astring, unknown_string, rfl))
            winning_score[sims.index(max(sims))] += 1
        # step 3 of algorithm
        final_score = max(winning_score) / float(kl)
        if final_score >= t:
           u_author.append(authors[winning_score.index(max(winning_score))])
           scores.append(final_score)
        else:
           u_author.append("None")
           scores.append(final_score)
    # store the result in a .txt file.
    f = open('results.txt', 'w')
    for i in range(len(unknowns)):
        f.write('Uunknown Text: {}\n'.format(unknowns[i]))
        f.write('Author: {}\n'.format(u_author[i]))
        f.write('Score: {}\n\n'.format(str(scores[i])))

    f.close()

main()