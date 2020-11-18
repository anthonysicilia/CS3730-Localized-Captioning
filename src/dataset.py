from torchvision import transforms
from PIL import Image
import torch
import pickle
import itertools
import pickle
import numpy as np
import os

class LocalizedCOCO(torch.utils.data.Dataset):

    def __init__(self, pickle_dir, vocab_pickle, caption_grouping=None):
        self.pickle_paths = [
            os.path.join(pickle_dir, f)
            for f in os.listdir(pickle_dir)
            if '.pickle' in f]
        self.caption_map = pickle.load(open(caption_grouping, 'rb')) \
            if caption_grouping is not None else None
        self.vocab =  pickle.load(open(vocab_pickle, 'rb'))
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.pickle_paths)

    
    def encode(self, arr):
        return [self.vocab[w] if w in self.vocab 
                else self.vocab['<ukn>']
                for w in arr ]
    
    def get_length(self, arr):
        i = 0
        w = arr[i]
        while w != '<end>':
            i +=1
            w = arr[i]
        return i + 1

    def __getitem__(self, index):

        data = pickle.load(open(self.pickle_paths[index], 'rb'))
        caption_len = torch.LongTensor([self.get_length(data['caption'])])
        caption = torch.LongTensor(self.encode(data['caption']))
        img = Image.open(data['img'])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = torch.FloatTensor(self.transform(img))
        traces = torch.FloatTensor(np.array([(x,y) for x,y in zip(data['x'], data['y'])]))

        if self.caption_map is not None:
            key = self.pickle_paths[index].split('/')[-1]
            root = '/'.join(self.pickle_paths[index].split('/')[:-1])
            all_captions = [self.encode(pickle.load(open(root + '/' + p, 'rb'))['caption'])
                for p in self.caption_map[key]]
            return img, caption, caption_len, traces, all_captions
        else:
            return img, caption, caption_len, traces
            
