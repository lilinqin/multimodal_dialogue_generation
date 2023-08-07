
from torch.utils.data import Dataset, DataLoader
import PIL
import torch
from torchvision import datasets, models, transforms
from transformers import AutoTokenizer, BertTokenizer, BartTokenizer
import argparse
from tqdm import tqdm
import json
import copy
import os
from pysubs2 import SSAFile
import numpy as np

datasets_dir = '/home/lqli/video_dialog_generation/data'

type2season = {'train':list(range(1,9)), 'eval':[9], 'test':[10]}
seasons_length = [24,24,25,24,24,25,24,24,23,17]

class Utterance:
    def __init__(self, text, video=None, audio=None):
        self.text = text
        self.video = video
        self.audio = audio

def pad(sentences, max_len, num):
    for sentence in sentences:
        sentence += [num] * (max_len - len(sentence))
    return sentences

def get_utteraces_from_subtile(season,episode): # extract the subtitle time fragment from subtitle file
    read_path = os.path.join(datasets_dir,'subtitle','season'+str(season),'applied','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)+'.srt')
    subs = SSAFile.load(read_path,encoding='ISO-8859-1',format='srt')
    return subs

def get_data(seasons, max_turns=4, frame="friendsVideoSegFeature"):
    datas = []
    print(frame)
    for ses in seasons:
        for epi in range(seasons_length[ses-1]):
            epi += 1
            subs = get_utteraces_from_subtile(ses,epi)
            his = []
            for segid, sub in enumerate(subs):
                segid = str(segid+1).zfill(3)
                text = sub.text.replace(r'\N',' ').replace(r'- ', ' ')
                text = text.replace(r'{\i0}', '').replace(r'{\i1}', '')
                text = ' '.join(text.split())
                video = os.path.join(datasets_dir, frame, 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                # video = os.path.join(datasets_dir, 'friendsVideoSegFeaturer21d', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                audio = os.path.join(datasets_dir, 'friendsAudioSegFeature', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')

                utterance = Utterance(text, video, audio)
                if segid != '001':
                    datas.append((his[:], utterance))
   
                his.append(utterance)
                if len(his) > max_turns * 2 + 1:
                    his = his[1:]

    return datas


def get_utteraces(seasons, max_turns=4, frame="friendsVideoSegFeature"):
    datas = []
    print(frame)
    for ses in seasons:
        for epi in range(seasons_length[ses-1]):
            epi += 1
            subs = get_utteraces_from_subtile(ses,epi)
            his = None
            for segid, sub in enumerate(subs):
                segid = str(segid+1).zfill(3)
                text = sub.text.replace(r'\N',' ').replace(r'- ', ' ')
                text = text.replace(r'{\i0}', '').replace(r'{\i1}', '')
                text = ' '.join(text.split())
                video = os.path.join(datasets_dir, frame, 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                # video = os.path.join(datasets_dir, 'friendsVideoSegFeaturer21d', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                audio = os.path.join(datasets_dir, 'friendsAudioSegFeature', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                utterance = Utterance(text, video, audio)
                if segid == '001':
                    his = utterance
                else:
                    datas.append((his, utterance))
                    his = utterance
                
    return datas


class DialogDataset(Dataset):
    def __init__(self, args, data_type='train'):
        self.tokenizer = args.tokenizer
        self.mode = args.mode
        self.max_turns = args.max_turns
        self.device = args.device

        seasons = type2season[data_type]
        self.frame = args.frame
        self.datas = get_data(seasons, args.max_turns, self.frame)

        self.max_len = args.max_len
        self.target_max_len = args.target_max_len


    def __getitem__(self, index):
        data = self.datas[index]
        his = data[0]
        ans = data[1]

        input_ids, attention_masks, role_type_ids, video_features, audio_features, labels, video_mask, audio_mask, text_idx, text_ids = self.build_input_from_segments(his, ans) 

        # print(audio_idx, len(audio_features))
        # print(input_ids,text_idx)
        return input_ids, attention_masks, role_type_ids, video_features, audio_features, labels, video_mask, audio_mask, text_idx, text_ids

    
    def build_input_from_segments(self, his, ans):
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        input_ids = []
        attention_masks = []
        role_type_ids = [] # 0, 
        video_features = []
        audio_features = []
        video_mask = []
        audio_mask = []
        text_ids = []
        text_idx = []
        his_len = len(his)
        role_type = 0

        for i in range(his_len):
            idx = his_len - i - 1
            text = his[idx].text
            video = his[idx].video
            audio = his[idx].audio

            
            if 'Audio' in self.mode:
                audio = np.load(audio).tolist()
                audio_features.append(audio)
                audio_mask.append([1]*len(audio))

            if 'Video' in self.mode:
                # video = np.load(video).tolist()
                video = np.load(video)
                # print(video.shape)
                video = video.tolist()

                video_features.append(video)
                video_mask.append([1]*len(video))
                                
            if 'Text' in self.mode:
                temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                text_ids.append(temp_input_ids.copy())
                temp_input_ids = temp_input_ids + [sep_id]

                if len(temp_input_ids) + len(input_ids) > self.max_len - 1:
                    break
                else:
                    role_type_ids = [role_type] * len(temp_input_ids) + role_type_ids
                    input_ids = temp_input_ids + input_ids
                    attention_masks = [1] * len(temp_input_ids) + attention_masks


            role_type = (role_type + 1) % 2
        
        role_type_ids = role_type_ids[0:1] + role_type_ids
        input_ids = [cls_id] + input_ids
        attention_masks = [1] + attention_masks

        audio_features.reverse()
        video_features.reverse()
        text_ids.reverse()
        video_mask.reverse()
        audio_mask.reverse()
        
        text_idx = [0] + [i for i, v in enumerate(input_ids) if v == sep_id]
        text_idx = [[text_idx[i]+1, text_idx[i+1]] for i in range(len(text_idx)-1)]

        labels = self.tokenizer.encode(ans.text, add_special_tokens=True, max_length=self.target_max_len, truncation=True)

        return input_ids, attention_masks, role_type_ids, video_features, audio_features, labels, video_mask, audio_mask, text_idx, text_ids


    def collate_fn(self, batch):
        input_ids, attention_masks, role_type_ids, video_features, audio_features, labels, video_mask, audio_mask, text_idx, text_ids = list(zip(*batch))
        # print(video_idx)
        pad_id = self.tokenizer.pad_token_id

        max_len = max((len(x) for x in input_ids))
        input_ids = pad(input_ids, max_len, pad_id)
        attention_masks = pad(attention_masks, max_len, 0)
        role_type_ids = pad(role_type_ids, max_len, 0)

        max_label_len = max((len(x) for x in labels))
        labels = pad(labels, max_label_len, pad_id)

        max_text_len = max((len(x) for x in text_idx))
        text_idx = pad(text_idx, max_text_len, [-1, -1])

        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_masks = torch.Tensor(attention_masks).to(self.device)
        role_type_ids = torch.LongTensor(role_type_ids).to(self.device)
        
        video_temp = []
        for utts in video_features:
            video_temp.extend(utts)
        
        audio_temp = []
        for utts in audio_features:
            audio_temp.extend(utts)
        
        video_mask_temp = []
        for utts in video_mask:
            video_mask_temp.extend(utts)
        
        audio_mask_temp = []
        for utts in audio_mask:
            audio_mask_temp.extend(utts)
        
        text_ids_temp = []
        for utts in text_ids:
            text_ids_temp.extend(utts)
        
        max_len = max((len(x) for x in text_ids_temp))
        text_ids = pad(text_ids_temp, max_len, 0)
        text_ids = torch.LongTensor(text_ids).to(self.device)
        
        if video_temp:
            max_len = max((len(x) for x in video_temp))
            video_temp = pad(video_temp, max_len, [0]*2048)
            video_mask_temp = pad(video_mask_temp, max_len, 0)

            video_features = torch.Tensor(video_temp).to(self.device)
            video_mask = torch.Tensor(video_mask_temp).to(self.device)
            # print(video_features.shape)
            # print(video_mask.shape)
        else:
            video_features = None
            video_mask = None
        

        if audio_temp:
            max_len = max((len(x) for x in audio_temp))
            audio_temp = pad(audio_temp, max_len, [0]*128)
            audio_mask_temp = pad(audio_mask_temp, max_len, 0)

            audio_features = torch.Tensor(audio_temp).to(self.device)
            audio_mask = torch.Tensor(audio_mask_temp).to(self.device)
        else:
            audio_features = None
            audio_mask = None

        
        labels = torch.LongTensor(labels).to(self.device)
        

        output = {
            'input_ids':input_ids,
            'attention_mask':attention_masks,
            'role_type':role_type_ids,
            'video_features':video_features,
            'video_mask':video_mask,
            'audio_features':audio_features,
            'audio_mask':audio_mask,
            'text_idx': text_idx,
            'text_ids': text_ids,
            'labels':labels
        }
        
        return output

    def __len__(self):
        # return 2
        # return 10000
        return len(self.datas)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--target_max_len', default=256, type=int)
    parser.add_argument('--mode', default='TextVideoAudio', type=str)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    parser.add_argument('--max_turns', default=2, type=int, help='max_turns')

    args = parser.parse_args()

    args.tokenizer = BartTokenizer.from_pretrained('/home/lqli/project/video_dialog_generation/pretrain/bart-base')
    dataset = DialogDataset(args, 'train')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    # print(len(dataloader))
    # print(len())
    for x in tqdm(dataloader):
        # print(x['video_features'].shape)
        # print(x['video_idx'])
        break
        1
        
    print('ok')