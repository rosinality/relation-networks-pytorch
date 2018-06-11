import os
import sys
import json
import pickle

import nltk
import tqdm
from torchvision import transforms
from PIL import Image


def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', f'CLEVR_{split}_questions.json')) as f:
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

        result.append(
            (
                question['image_filename'],
                question_token,
                answer,
                question['question_family_index'],
            )
        )

    with open(f'data/{split}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


resize = transforms.Resize([128, 128])


def process_image(path, output_dir):
    images = os.listdir(path)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for imgfile in tqdm.tqdm(images):
        img = Image.open(os.path.join(path, imgfile)).convert('RGB')
        img = resize(img)
        img.save(os.path.join(output_dir, imgfile))


if __name__ == '__main__':
    root = sys.argv[1]

    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open('data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

    process_image(
        os.path.join(sys.argv[1], 'images/train'),
        os.path.join(sys.argv[1], 'images/train_preprocessed'),
    )
    process_image(
        os.path.join(sys.argv[1], 'images/val'),
        os.path.join(sys.argv[1], 'images/val_preprocessed'),
    )

