import os
import sys
import pickle

import nltk
import tqdm

def preprocess(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if possible_answer is None:
        possible_answer = {}

    with open(os.path.join(root, f'CLEVR_{split}_questions.json')) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer))

    with open(f'data/{split}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = sys.argv[1]

    word_dic, answer_dic = preprocess(root, 'train')
    preprocess(root, 'val', word_dic, answer_dic)