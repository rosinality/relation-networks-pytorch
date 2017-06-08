import os
import sys
import json
import pickle

import nltk
import tqdm
from torchvision import transforms

from transforms import Scale

transform = transforms.Compose([
    Scale([128, 128]),
    transforms.Pad(4)
])

def process_image(root, split, image_file):
    img = Image.open(os.path.join(root, 'images',
                                split, image_file)).convert('RGB')
    img = transform(img)
    new_path = os.path.join(root, 'images', f'{split}_processed', image_file)
    img.save(new_path)

def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions',
                        f'CLEVR_{split}_questions.json')) as f:
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

    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open('data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

    train_images = os.listdir(os.path.join(root, images, 'train'))

    for image in train_images:
        process_image(root, 'train', image)