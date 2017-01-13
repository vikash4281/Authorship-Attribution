this code is an implementation of Kopell et al. (2011). kopell suggested an extension to his work koppel (2007).
the algorithm for this code is
given a snippet of length L1, known-texts of length L2 for each of C candidates
1.repeat kl times
    a. randomly choose some fraction k2 from the feature set.
    b. find top match using cosine similarity

2. for each author a, proportion of times a is top match.
3. if greater than threshold, its the author. else none.

this code uses test data set from pan.webis.de.
these data sets contain a file named meta-json. this file contains the names of all the text files in the
directory, the language of the texts, encoding in the  json format.

detailed explanation is given in the code. for running the code, please store the path of the meta-json
file in the location part first in the main function in the format of a string. use double slashes inplace of single slash.

for example:
location = C:\\Users\\vicky\\Desktop\\authorship-attribution-test-dataset-problem-c-2015-10-20

and you can run without passing any arguments.

the result will be created in a text format in a file named results.txt
