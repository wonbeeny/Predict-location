import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import matplotlib.pyplot as plt
from IPython.display import Image
from tqdm import tqdm, tqdm_notebook
from tqdm import tqdm, tqdm_notebook
from keras.models import load_model


def make_model(data, embed_size, LSTM_size, epoch, model_name = None):
    route = data.fr_danger_to_occur_route
    
    p_id_all_route = []
    for i in range(0, len(route)):
        globals()['p_id_route_{}'.format(i)] = route[i].split("->")
        p_id_all_route.extend(route[i].split("->"))
        
    p_id_all_route_unique = list(set(p_id_all_route))
    p_id_all_route_unique.sort()
    
    word_dict = {w: i for i, w in enumerate(p_id_all_route_unique)}
    
    for i in range(0, len(route)):
        globals()['test_{}'.format(i)] = []

    for j in range(0, len(route)):
        for i in range(0, len(globals()['p_id_route_{}'.format(j)])):
            globals()['test_{}'.format(j)].append(word_dict.get(globals()['p_id_route_{}'.format(j)][i]))
            
    for i in range(0, len(route)):
        for j in range(0, len(globals()['test_{}'.format(i)])-1 ):
            globals()['sequences_{}'.format(i)] = []

    vocab_size = len(word_dict)
            
    for i in range(0, len(route)):
        for j in range(1, len(globals()['test_{}'.format(i)])):
            sequence = globals()['test_{}'.format(i)][j-1:j+1]
            globals()['sequences_{}'.format(i)].append(sequence)
            
    for i in tqdm(range(0, len(route))):
        for j in range(0, len(globals()['sequences_{}'.format(i)])):
            globals()['sequences_{}'.format(i)] = np.array(globals()['sequences_{}'.format(i)])
            globals()['X_{}'.format(i)], globals()['y_{}'.format(i)] = globals()['sequences_{}'.format(i)][:,0],globals()['sequences_{}'.format(i)][:,1]
            globals()['y_{}'.format(i)] = to_categorical(globals()['y_{}'.format(i)], num_classes=vocab_size)

    for i in tqdm_notebook(range(0,1)):
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train(files=corpus, vocab_size=vocab_size, special_tokens=special_tokens)
        
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=1))
    model.add(LSTM(LSTM_size))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    for i in tqdm_notebook(range(0, len(route))):
        model.fit(globals()['X_{}'.format(i)], globals()['y_{}'.format(i)], epochs = epoch, verbose = 0)

    print('Model 완성 !!')
    
    if model_name == str:
        model.save(model_name)
    else:
        model.save('LSTM_model')
        
def pred_route(data, model, route_text, n_pred):
    if type(model) == str:
        model = load_model(model)

        route = data.fr_danger_to_occur_route

        p_id_all_route = []
        for i in range(0, len(route)):
            globals()['p_id_route_{}'.format(i)] = route[i].split("->")
            p_id_all_route.extend(route[i].split("->"))

        p_id_all_route_unique = list(set(p_id_all_route))
        p_id_all_route_unique.sort()

        word_dict = {w: i for i, w in enumerate(p_id_all_route_unique)}

        route_all = [route_text]

        for _ in range(n_pred):
            encoded = [word_dict[route_text]]
            encoded = np.array(encoded)
            yhat = model.predict_classes(encoded, verbose=0)

            for route, index in word_dict.items():
                if index == yhat:
                    route_all.append(route)
                    route_text = route
                    break

        return route_all
    
    else :
        print('model의 type은 str입니다.')
    
    