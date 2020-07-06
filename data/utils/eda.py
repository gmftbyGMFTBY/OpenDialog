# Easy data augmentation techniques on Chinese
# Jason Wei and Kai Zou
# Fixed by GMFTBY, 2019.8.3
# https://github.com/gmftbyGMFTBY/EDA-NLP-Chinese

import random
from random import shuffle
import re
import ipdb
import jieba
import synonyms
from tqdm import tqdm

def gen_eda(sentence, alpha=0.1, num_aug=9):
    '''
    :alpha: the percent of words in each sentence to be changed, details can be found in [here](https://arxiv.org/abs/1901.11196)
    :num_aug: number of augmented sentences per original sentence
    '''
    aug_sentences = eda(
            sentence, 
            alpha_sr=alpha, 
            alpha_ri=alpha, 
            alpha_rs=alpha, 
            p_rd=alpha, 
            num_aug=num_aug)
    return aug_sentences

# ========== HELPER FUNCTIONS ========== #
def load_stop_words(path):
    stop_words = []
    with open(path) as f:
        for i in f.readlines():
            if i.strip():
                stop_words.append(i.strip())
    stop_words.append('')
    return stop_words

def cleanup(line):
    line = line.replace('"', "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.strip()
    return line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from Synonyms
########################################################################

def synonym_replacement(words, n, stop_words):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synon = synonyms.nearby(random_word)[0]
        if len(synon) >= 1:
            synonym = random.choice(list(synon))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) <= 1:
        return words
    
    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
            
    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    
    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    if len(new_words) == 0:
        new_words.append('.')
        new_words.append('.')
    elif len(new_words) == 1:
        new_words.append('.')
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synon = []
    counter = 0
    if len(new_words) == 0:
        new_words.append('.')
        new_words.append('.')
    elif len(new_words) == 1:
        new_words.append('.')
    while len(synon) < 1:
        random_word = random.choice(new_words)
        synonym = synonyms.nearby(random_word)[0]
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonym[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=5):
    sentence = cleanup(sentence)
    # jieba
    words = list(jieba.cut(sentence))
    words = [word for word in words if word is not '']
    num_words = len(words)
    
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    for _ in range(num_new_per_technique):
        stop_words = load_stop_words('stopwords.txt')
        a_words = synonym_replacement(words, n_sr, stop_words)
        augmented_sentences.append(''.join(a_words))
        
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(''.join(a_words))
        
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))
        
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))
        
    augmented_sentences = [cleanup(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)
    
    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
    
    return augmented_sentences
