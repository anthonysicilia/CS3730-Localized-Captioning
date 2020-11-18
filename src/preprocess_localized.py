from tqdm import tqdm
from PIL import Image
from math import ceil
import numpy as np
import itertools
import pickle
import string
import json
import os

# 'coco/mscoco_train2017/000000000009.jpg'

KEYWORDS = {'<start>', '<pad>', '<end>', '<ukn>'}

def process_caption(timed_caption, max_len=35):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    caption = []
    end_times = []
    for ci in timed_caption:
        utterances = ci['utterance'].translate(table).lower().split()
        caption += utterances
        start = ci['start_time']
        end = ci['end_time']
        if len(utterances) > 0:
            step = (end - start) / len(utterances)
            cutpoint = start
            while len(end_times) < len(caption):
                cutpoint = start + step
                end_times += [cutpoint]

    # chop if needed and add ending
    caption = caption[:max_len] + ['<end>']
    end_times = end_times[:max_len] + [max(end_times)]
    # pad until long enough
    while len(caption) < max_len + 1:
        caption += ['<pad>']
    while len(end_times) < len(caption):
        end_times += [max(end_times)]
    # add beginning
    caption = ['<start>'] + caption
    end_times = [0.0] + end_times
    assert len(caption) == max_len + 2
    assert len(end_times) == max_len + 2
    return caption, end_times

def temporal_average(arr, times, end_times):
    i = 0
    averaged = [[]]
    for a,t in zip(arr, times):
        if t <= end_times[i]:
            averaged[i].append(a)
        else:
            i +=1
            if i < len(end_times):
                averaged.append([a])
            else:
                break
    while len(averaged) < len(end_times):
        averaged.append([-1])
    for i in range(len(averaged)):
        if len(averaged[i]) > 0:
            averaged[i] = sum(averaged[i]) / len(averaged[i])
        else:
            averaged[i] = -1
    return averaged
    
def process_trace(traces, caption, end_times):
    traces = list(itertools.chain.from_iterable(traces))
    x = np.array([min(max(trace['x'], 0), 1) for trace in traces])
    y = np.array([min(max(trace['y'], 0), 1) for trace in traces])
    t = np.array([trace['t'] for trace in traces])
    x = temporal_average(x, t, end_times)
    y = temporal_average(y, t, end_times)

    assert len(x) == len(y)
    assert len(x) == len(caption)

    return x,y
    

def read_and_write_to(jsonl_f, write_dir, words=None, traces=True):
    read_dir = '/'.join(jsonl_f.split('/')[:-1])
    for line in tqdm(open(jsonl_f, 'r').readlines()):
        save_data = dict()
        data = json.loads(line)
        save_path = os.path.join(write_dir, 
            data['image_id'] + '-' + str(data['annotator_id']) + '.pickle')
        img_dir = os.path.join(read_dir, data['dataset_id'])
        img_path = os.path.join(img_dir, data['image_id'].zfill(12)) + '.jpg'
        # check file path
        _ = Image.open(img_path)
        caption, end_times = process_caption(data['timed_caption'])
        if words is not None:
            for w in caption:
                if not w in KEYWORDS:
                    if w not in words:
                        words[w] = 0
                    words[w] += 1
        if len(caption) == 0:
            with open('log.txt', 'a') as out:
                out.write(save_path)
            continue
        if traces:
            x,y = process_trace(data['traces'], caption, end_times)
            if x is None or y is None:
                with open('log.txt', 'a') as out:
                    out.write(save_path)
                continue
        save_data['img'] = img_path
        save_data['caption'] = caption
        if traces:
            save_data['x'] = x
            save_data['y'] = y
        pickle.dump(save_data, open(save_path, 'wb'))

def generate_caption_grouping(dir_):
    paths = os.listdir(dir_)
    # data['image_id'] + '-' + str(data['annotator_id'])
    caption_grouping = dict()
    for pi in paths:
        caption_grouping[pi] = []
        image_id = pi.split('-')[0]
        for pj in paths:
            if pj.split('-')[0] == image_id:
                caption_grouping[pi].append(pj)
    return caption_grouping

if __name__ == '__main__':

    # train_paths = [
    #     '../../../shared_data/coco17/coco_train_localized_narratives-00000-of-00004.jsonl',
    #     '../../../shared_data/coco17/coco_train_localized_narratives-00001-of-00004.jsonl',
    #     '../../../shared_data/coco17/coco_train_localized_narratives-00002-of-00004.jsonl',
    #     '../../../shared_data/coco17/coco_train_localized_narratives-00003-of-00004.jsonl'
    # ]

    # words = dict()

    # for tp in train_paths:
    #     print(tp)
    #     read_and_write_to(tp, 'coco_localized/train', words=words, traces=True)
    
    # words = [w for w in words.keys() if words[w] > 10]
    # words = {k: v for v, k in enumerate(words)}
    # for w in KEYWORDS:
    #     words[w] = len(words)
    
    # pickle.dump(words, open('coco_localized/vocab.pickle', 'wb'))

    val_path = '../../../shared_data/coco17/coco_val_localized_narratives.jsonl'

    read_and_write_to(val_path, 'coco_localized/val', words=None, traces=True)
    
    grouping = generate_caption_grouping('coco_localized/val')
    pickle.dump(grouping, open('coco_localized/caption_grouping.pickle', 'wb'))
