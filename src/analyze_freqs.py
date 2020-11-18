"""
Code for analyzing word frequencies.
Author: Anthony Sicilia
"""

import torch
from dataset import LocalizedCOCO
from tqdm import tqdm
import nltk
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # # commented out to save time after running once and saving data 
    # trainset = LocalizedCOCO('coco_localized/train', 
    #     'coco_localized/vocab.pickle')
    
    # vocab = {v : k for k,v in trainset.vocab.items()}

    # train_loader = torch.utils.data.DataLoader(trainset,
    #     batch_size=32, 
    #     shuffle=True, 
    #     num_workers=6)
    
    # words = dict()
    
    # for imgs, caps, caplens, traces in tqdm(train_loader):
    #     for arr in caps:
    #         text = []
    #         for word in [vocab[w.item()] for w in arr]:
    #             if word not in {'<start>', '<end>', '<pad>'}:
    #                 text.append(word)
    #         for j, (word, pos) in enumerate(nltk.pos_tag(text)):
    #             if pos == 'NN':
    #                 if word not in words:
    #                     words[word] = []
    #                 words[word].append(j)
    # pickle.dump(words, open('coco_localized/counts.pickle', 'wb'))
    words = pickle.load(open('coco_localized/counts.pickle', 'rb'))
    for k,v in words.items():
        if len(v) > 50:
            words[k] = sum([1 for a in v if a < 10]) / len(v)
        else:
            words[k] = -1
    words = {k : v for k,v in sorted(words.items(), key=lambda x: -x[1])}
    words = {k : v for k,v in list(words.items())[:50]}
    df = pd.DataFrame({'word' : [k for k in words], 
        'perecent of appearances' : [words[k] for k in words]})
    fig, ax = plt.subplots(figsize=(6,10))
    sns.barplot(x='perecent of appearances', y='word', data=df, ax=ax)
    plt.title('Percent of total appearances within the first 10 words\n (tagged NN, appearing more than 50 times)')
    plt.tight_layout()
    # plt.xscale('log')
    plt.savefig('counts')
