import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import json
from nltk.tokenize import wordpunct_tokenize
import string

root_path = '/data/zhangruwen/MM_Controversy_Detection_Released-main/mcd/dataset/'



class Oursdataset(Dataset):
    def __init__(self, base_path, path_vid):
        self.video_feas = h5py.File(
            os.path.join(base_path, "ours_video_feature_clip_vit_l_fixframe.h5"), "r"
        )["video_feas"]
        self.title_feas = h5py.File(
            os.path.join(base_path, "ours_desc_feature_clip.h5"), "r"
        )
        self.comment_feas = h5py.File(
            os.path.join(base_path, "ours_comments_feature_clip.h5"), "r"
        )
        self.author_feas = h5py.File(
            os.path.join(base_path, "ours_author_feature_clip.h5"), "r"
            # os.path.join(base_path, "ours_author_feature_clip_heart.h5"), "r"
        )
        self.asr_feas = h5py.File(
            os.path.join(base_path, "ours_asr_feature_clip.h5"), "r"
        )
        self.topics_feas = h5py.File(
            os.path.join(base_path, "ours_topics_feature_clip2.h5"), "r"
        )

        self.data = []
        with open(root_path+"ours/final_all_data_cleaned_v4.json", "r", encoding="utf-8-sig") as f:
            self.hotnum_complete = json.load(f)
        with open(os.path.join(base_path, "data-split/", path_vid), "r") as fr:
            self.vid = fr.readlines()
        self.vid = [i.replace("\n", "") for i in self.vid]
        self.data = []
        for i in self.hotnum_complete:
            if i["video_id"] in self.vid:
                self.data.append(i)
        # self.hotnum = []
        # for i in self.hotnum_complete:
        #     # self.hotnum.append(i['hot_num'])
        #     self.hotnum.append(round(i['hot_num'], 3))
        self.hotnum = {}
        for i in self.hotnum_complete:
            self.hotnum[i["video_id"]] = round(i['hot_num'], 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = str(item["video_id"])
        # 热度
        # hotNum = self.hotnum[idx]
        hotNum = item["hot_num"]
        # label
        label = torch.tensor(int(item["controversy"]))
        # video
        video_fea = self.video_feas[vid][:]
        video_fea = torch.FloatTensor(video_fea)
        # title
        title_fea = self.title_feas[vid][:]
        title_fea = torch.FloatTensor(title_fea)
        # comments
        if vid in self.comment_feas.keys():
            comments_fea = self.comment_feas[vid][:]
            comments_fea = torch.FloatTensor(comments_fea)
            comment_lens = len(comments_fea[:50])
        else:
            comments_fea = torch.FloatTensor(768)
            comment_lens = 1
        # author
        author_fea = self.author_feas[vid][:]
        author_fea = torch.FloatTensor(author_fea)
        # asr
        asr_fea = self.asr_feas[vid][:]
        asr_fea = torch.FloatTensor(asr_fea)
        # topics
        topics_fea = self.topics_feas[vid][:]
        topics_fea = torch.FloatTensor(topics_fea)
        return {
            "vid": vid,
            "label": label,
            "video_fea": video_fea,
            "title_fea": title_fea,
            "comments_fea": comments_fea,
            "comment_lens": comment_lens,
            "author_fea": author_fea,
            "asr_fea": asr_fea,
            "topics_fea": topics_fea,
            "hotNum": hotNum,
        }




def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        if len(video.shape) == 1:
            video = torch.zeros((1, 768))
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat(
                (
                    video,
                    torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float),
                ),
                dim=0,
            )
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def MCD_collate_fn(batch):
    num_comments = 50
    num_frames = 10  # 80
    vids = [int(item["vid"]) for item in batch]
    labels = [item["label"] for item in batch]
    video_feas = [item["video_fea"] for item in batch]
    video_feas, video_masks = pad_frame_sequence(num_frames, video_feas)
    asr_feas = [item["asr_fea"] for item in batch]
    topics_fea = [item["topics_fea"] for item in batch]
    title_feas = [item["title_fea"] for item in batch]
    comment_lens = [len(item["comments_fea"][:50]) for item in batch]
    comment_feas = [item["comments_fea"] for item in batch]
    comment_feas, comment_masks = pad_frame_sequence(num_comments, comment_feas)
    author_feas = [item["author_fea"] for item in batch]
    hotNums = [item["hotNum"] for item in batch]

    return {
        "vid": torch.tensor(vids),
        "label": torch.stack(labels),
        "video_feas": video_feas,
        "video_masks": video_masks,
        "title_feas": torch.stack(title_feas),
        "comment_feas": comment_feas,
        "comment_lens": torch.IntTensor(comment_lens),
        "author_feas": torch.stack(author_feas),
        "asr_feas": torch.stack(asr_feas),
        "topics_fea": torch.stack(topics_fea),
        "hotNums": torch.FloatTensor(hotNums),
    }