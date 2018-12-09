import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from nltk import (word_tokenize, pos_tag)
from nltk.corpus import sentiwordnet as swn
from nltk.metrics import edit_distance
# import hunspell
import re
from tqdm import tqdm
import argparse
import sys
import config as cf

# def parse2014AspectTerm(filepath):
#     """
#     Since no good way of collecting the aspect term words from the raw xml data,
#     this function is using loop to facilitate collecting the terms manually.
#     """
#     aspectTerm_dict = {
#                         'food': [],
#                         'service': [],
#                         'price': [],
#                         'ambience': [],
#                         'anecdotes/miscellaneous': []
#                       }
#     tree = et.parse(filepath)
#     root = tree.getroot()
#     sentences = root.findall('sentence')

def parse2014(filepath, args):
    """
    parse 2014 raw data in xml format
    only tested for restaurant data
    """
    data = pd.DataFrame(columns = ['id', 'text', 'aspect', 'polarity'])
    tree = et.parse(filepath)
    root = tree.getroot()
    sentences = root.findall('sentence');
    i = 0
    for sentence in tqdm(sentences):
        id = sentence.attrib.get('id')
        text = sentence.find('text').text
        # TODO categorize term words/phrases into aspect terms
        # aspectTerms = child.find('aspectTerms')
        # if aspectTerms != None:
        #     for term in aspectTerms.findall('aspectTerm'):
        #         terms.append(term.attrib.get('term'))
        for category in sentence.find('aspectCategories').findall('aspectCategory'):
            if category.attrib.get('polarity') != 'conflict':
                data.loc[i] = [id, text, category.attrib.get('category'), category.attrib.get('polarity')]
                i = i + 1
    writeCSV(data, cf.ROOT_PATH + cf.DATA_PATH + '%s_%s_%s_raw.csv' % (args.domain, args.aim, args.year))
    # revised 3/28/18 to add call to writeCOR
    writeCOR(data, cf.ROOT_PATH + cf.DATA_PATH + '%s_%s_%s_raw.cor' % (args.domain, args.aim, args.year))
    return data

def writeCSV(dataframe, filepath):
    dataframe.to_csv(filepath, index = False)

def writeCOR(dataframe, filepath):
    numex = len(dataframe.index)
    with open(filepath, 'w') as f:
        for i in range(numex):
            #
            if dataframe.loc[i][3] == 'positive':
                f.write(dataframe.loc[i][1] + '\n')
                f.write(dataframe.loc[i][2] + '\n')
                f.write('1' + '\n')
            elif dataframe.loc[i][3] == 'negative':
                f.write(dataframe.loc[i][1] + '\n')
                f.write(dataframe.loc[i][2] + '\n')
                f.write('-1' + '\n')
            elif dataframe.loc[i][3] == 'neutral':
                f.write(dataframe.loc[i][1] + '\n')
                f.write(dataframe.loc[i][2] + '\n')
                f.write('0' + '\n')
    #
    f.close()
    # end of writeCor()

def tokenize(data):
    wordData = []
    for s in data:
        wordData.append([w for w in word_tokenize(s.lower())])
    return wordData

def cleanup(wordData):
    dictionary = embeddingDict(embeddingPath)
    wordData = cleanOp(wordData, re.compile(r'-'), dictionary, correctDashWord)
    wordData = cleanOp(wordData, re.compile(r'-'), dictionary, cleanDashWord)
    wordData = cleanOp(wordData, re.compile(r':'), dictionary, parseTime)
    wordData = cleanOp(wordData, re.compile('\+'), dictionary, parsePlus)
    wordData = cleanOp(wordData, re.compile(r'\d+'), dictionary, parseNumber)
    # Revised 3/29/18 to move spell check to separate method
    # wordData = cleanOp(wordData, re.compile(r''), dictionary, correctSpell)
    return wordData

def spellcheck(wordData):
    dictionary = embeddingDict(embeddingPath)
    wordData = cleanOp(wordData, re.compile(r''), dictionary, correctSpell)
    return wordData

def cleanOp(wordData, regex, dictionary, op):
    for i, sentence in enumerate(wordData):
        if bool(regex.search(sentence)):
            newSentence = ''
            for word in word_tokenize(sentence.lower()):
                if bool(regex.search(word)) and word not in dictionary:
                    word = op(word)
                newSentence = newSentence + ' ' + word
            wordData[i] = newSentence[1:]   # revised 3/29/18 to avoid space at start of sentence
    return wordData

def parseTime(word):
    time_re = re.compile(r'^(([01]?\d|2[0-3]):([0-5]\d)|24:00)(pm|am)?$')
    if not bool(time_re.match(word)):
        return word
    else:
        dawn_re = re.compile(r'0?[234]:(\d{2})(am)?$')
        earlyMorning_re = re.compile(r'0?[56]:(\d{2})(am)?$')
        morning_re = re.compile(r'((0?[789])|(10)):(\d{2})(am)?$')
        noon_re = re.compile(r'((11):(\d{2})(am)?)|(((0?[01])|(12)):(\d{2})pm)$')
        afternoon_re = re.compile(r'((0?[2345]):(\d{2})pm)|((1[4567]):(\d{2}))$')
        evening_re = re.compile(r'((0?[678]):(\d{2})pm)|(((1[89])|20):(\d{2}))$')
        night_re = re.compile(r'(((0?9)|10):(\d{2})pm)|((2[12]):(\d{2}))$')
        midnight_re = re.compile(r'(((0?[01])|12):(\d{2})am)|(0?[01]:(\d{2}))|(11:(\d{2})pm)|(2[34]:(\d{2}))$')
        if bool(noon_re.match(word)):
            return 'noon'
        elif bool(evening_re.match(word)):
            return 'evening'
        elif bool(morning_re.match(word)):
            return 'morning'
        elif bool(earlyMorning_re.match(word)):
            return 'early morning'
        elif bool(night_re.match(word)):
            return 'night'
        elif bool(midnight_re.match(word)):
            return 'midnight'
        elif bool(dawb_re.match(word)):
            return 'dawn'
        else:
            return word

def parsePlus(word):
    return re.sub('\+', ' +', word)

def parseNumber(word):
    if bool(re.search(r'\d+', word)):
        return word
    else:
        search = re.search(r'\d+', word)
        pos = search.start()
        num = search.group()
        return word[:pos] + ' %s ' % num + parseNumber(word[pos+len(num):])

# def translateSymbol(word):

def checkSpell(word):
    global hobj
    return hobj.spell(word)

def correctSpell(word):
    global hobj
    suggestions = hobj.suggest(word)
    if len(suggestions) != 0:
        distance = [edit_distance(word, s) for s in suggestions]
        return suggestions[distance.index(min(distance))]
    else:
        return word

def createTempVocabulary(wordData, args):
    words = sorted(set([word for l in wordData for word in l.split(' ')]))
    global embeddingPath
    vocabulary = filterWordEmbedding(words, embeddingPath, args)
    return vocabulary

def splitDashWord(word):
    if '-' not in word:
        return [word]
    else:
        return word.split('-')

def cleanDashWord(word):
    return ''.join([s + ' ' for s in word.split('-')])

def correctDashWord(word):
    splittedWords = word.split('-')
    for i, word in enumerate(splittedWords):
        if not checkSpell(word):
            splittedWords[i] = correctSpell(word)
    return ''.join([s + '-' for s in splittedWords])[:-1]

def joinWord(words):
    return ''.join([s + ' ' for s in words])[:-1]

def embeddingDict(embeddingPath):
    dictionary = []
    with open(embeddingPath) as f:
        for line in tqdm(f):
            values = line.split()
            word = joinWord(values[:-300])
            dictionary.append(word)
    f.close()
    return dictionary

def filterWordEmbedding(words, embeddingPath, args):
    vocabulary = []
    filteredEmbeddingDict = []
    words = [word.lower() for word in words]
    with open(embeddingPath) as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            # word = word.decode('utf-8')     # added to remove Unicode warning
            # try-except added to debug Unicode warning
            # to see the word that triggers warning, from command line: python -W error::UnicodeWarning preprocess.py
            try:
                if word in words:
                    vocabulary.append(word)
                    filteredEmbeddingDict.append(line)
            except:
                print("stopping in filterWordEmbedding")
                # print("line: ", line)
                # print("values: ", values)
                print("word: ", word)
                # print("words: ", words)
                # exit()
    f.close()
    unknownWords = [word for word in words if word not in vocabulary]
    with open(dataPath + '%s_filtered_%s.txt' % (cf.WORD2VEC_FILE[0:-4], args.aim), 'w+') as f:
        for line in filteredEmbeddingDict:
            f.write(line)
    with open('unknown.txt', 'w+') as f:
        for i, word in enumerate(unknownWords):
            f.write(word + '\n')

def createVocabulary(trainDictPath, testDictPath, gloveDictPath):
    dictionary = []
    with open(trainDictPath) as f:
        for line in f:
            dictionary.append(line)
    f.close()
    with open(testDictPath) as f:
        for line in f:
            dictionary.append(line)
    f.close()
    with open(gloveDictPath) as f:
        miscFlag = True
        anecFlag = True
        for line in f:
            if not (miscFlag or anecFlag):
                break
            word = line.split()[0]
            if miscFlag and word == 'miscellaneous':
                dictionary.append(line)
                miscFlag = False
            if anecFlag and word == 'anecdotes':
                dictionary.append(line)
                anecFlag = False
    f.close()
    dictionary = set(dictionary)
    dictionaryNP = np.zeros((len(dictionary) + 1, 300))
    with open(dataPath + '%s_filtered.txt' % cf.WORD2VEC_FILE[0:-4], 'w+') as f:
        for i, line in enumerate(dictionary):
            values = line.split()
            try:
                dictionaryNP[i] = np.asarray(values[-300:], dtype='float32')
            except ValueError:
                print(joinWord(values[:-300]))
            f.write(line)
    f.close()
    dictionaryNP[-1] = np.random.normal(0, 0.01, [1,300])
    np.save(dataPath + 'glove', dictionaryNP)

def sampleData():
    """
    To randomly sample a small fraction from the processed train and test data,
    which will be used for testing the models
    """
    trainDataPath = dataPath + cf.TRAIN_FILE
    testDataPath = dataPath + cf.TEST_FILE
    train = pd.read_csv(trainDataPath)
    test = pd.read_csv(testDataPath)
    trainSample = train.sample(frac = 0.3, replace = False)
    testSample = test.sample(frac = 0.3, replace = False)
    writeCSV(trainSample, dataPath + 'rest_train_sample.csv')
    writeCSV(testSample, dataPath + 'rest_test_sample.csv')

def encodeAllData():
    """
    encode the process data into index array in the filtered glove dictionary,
    which will be used by the
    """
    def encodeData(filePath, type):
        data = pd.read_csv(filePath)
        texts = data['text']
        sentences = [word_tokenize(text) for text in texts]
        textIndex = []
        encoding = pd.DataFrame(columns = ['id', 'text_encode', 'aspect', 'polarity'])
        i = 0
        # for counting the length of the longest sentence
        max = 0
        for i, words in enumerate(sentences):
            sentenceIndex = []
            for word in words:
                try:
                    idx = index.index(word)
                except ValueError:
                    idx = 4859
                sentenceIndex.append(idx)
            if max < len(sentenceIndex):
                max = len(sentenceIndex)
            textIndex.append(sentenceIndex)
        print(max)
        np.save(dataPath + '%s' % type, np.array(textIndex))

    dictionary = {}
    with open(dataPath + '%s_filtered.txt' % cf.WORD2VEC_FILE[0:-4], 'r') as f:
        # print(header)
        for line in f:
            values = line.split()
            word = joinWord(values[:-300])
            vector = np.array(values[-300:], dtype='float32')
            dictionary[word] = vector
    f.close()
    index = list(dictionary.keys())
    trainDataPath = dataPath + cf.TRAIN_FILE
    testDataPath = dataPath + cf.TEST_FILE
    trainSampleDataPath = dataPath + 'rest_train_sample.csv'
    testSampleDataPath = dataPath + 'rest_test_sample.csv'
    encodeData(trainDataPath, 'train')
    encodeData(testDataPath, 'test')
    encodeData(trainSampleDataPath, 'train_sample')
    encodeData(testSampleDataPath, 'test_sample')


if __name__ == '__main__':
    argv = sys.argv[1:]                                                     # Slice off the first element of argv (which would just be the name of the program)

    ################################################
    ##  BEGIN SETTING UP DEFAULT HYPERPARAMETERS  ##
    ################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2014')                 # Name will be written to the results file  # Seed used for 'random'  module, which will shuffle training data
    parser.add_argument('--domain', type=str, default='rest')              #
    parser.add_argument('--embedding', type=str, default='glove')                  #
    parser.add_argument('--aim', type=str, default='trial')
    parser.add_argument('--full_run', type=int, choices=[0, 1], default=0)      #

    args, _ = parser.parse_known_args(argv)

    cf.configure(args.year, args.domain, args.embedding, args.aim)
    dataPath = cf.ROOT_PATH + cf.DATA_PATH
    embeddingPath = dataPath + cf.WORD2VEC_FILE
    # loading dictionaries for hunspell is a bit wierd, you have to put the dictionaries
    # in a root-derivative folder path e.g. a folder ~/some-other-path is not allowed
    hobj = hunspell.HunSpell(cf.HUNSPELL_PATH + cf.HUNSPELL_DICT[0],
                             cf.HUNSPELL_PATH + cf.HUNSPELL_DICT[1])
    parser = cf.PARSER[args.year]
    if args.full_run == 0:
        rawDataPath = dataPath + cf.DATA_FILE
        data = parser(rawDataPath, args)
        data['text'] = cleanup(data['text'])
        # revised 3/29/18 to create interim data without corrected spellings
        writeCOR(data, dataPath + '%s_%s_%s_clean_no_spellck.cor' % (args.domain, args.aim, args.year))
        data['text'] = spellcheck(data['text'])
        tempVocabulary = createTempVocabulary(data['text'], args)
        writeCSV(data, dataPath + '%s_%s_%s_processed.csv' % (args.domain, args.aim, args.year))
        # revised 3/28/18 to add call to writeCOR
        writeCOR(data, dataPath + '%s_%s_%s_processed.cor' % (args.domain, args.aim, args.year))
        createVocabulary(dataPath + '%s_filtered_train.txt' % cf.WORD2VEC_FILE[0:-4], dataPath + '%s_filtered_test.txt' % cf.WORD2VEC_FILE[0:-4], embeddingPath)
    else:
        #process train data
        args.aim = 'train'
        cf.configure(args.year, args.domain, args.embedding, args.aim)
        trainDataPath = dataPath + cf.DATA_FILE
        trainData = parser(trainDataPath, args)
        trainData['text'] = cleanup(trainData['text'])
        # revised 3/29/18 to create interim data without corrected spellings
        writeCOR(trainData, dataPath + 'rest_train_2014_clean_no_spellck.cor')
        trainData['text'] = spellcheck(trainData['text'])
        trainVocabulary = createTempVocabulary(trainData['text'], args)
        writeCSV(trainData, dataPath + 'rest_train_2014_processed.csv')
        # revised 3/28/18 to add call to writeCOR
        writeCOR(trainData, dataPath + 'rest_train_2014_processed.cor')
        #
        # process test data
        args.aim = 'test'
        cf.configure(args.year, args.domain, args.embedding, args.aim)
        testDataPath = dataPath + cf.DATA_FILE
        testData = parser(testDataPath, args)
        testData['text'] = cleanup(testData['text'])
        # revised 3/29/18 to create interim data without corrected spellings
        writeCOR(testData, dataPath + 'rest_test_2014_clean_no_spellck.cor')
        testData['text'] = spellcheck(testData['text'])
        testVocabulary = createTempVocabulary(testData['text'], args)
        writeCSV(testData, dataPath + 'rest_test_2014_processed.csv')
        # revised 3/28/18 to add call to writeCOR
        writeCOR(testData, dataPath + 'rest_test_2014_processed.cor')

        # export the final embedding dictionary by combining the dict from train and test data
        createVocabulary(dataPath + '%s_filtered_train.txt' % cf.WORD2VEC_FILE[0:-4], dataPath + '%s_filtered_test.txt' % cf.WORD2VEC_FILE[0:-4], embeddingPath)

        # sampling from the processed train and test data
        sampleData()

        # encode all data
        encodeAllData()
