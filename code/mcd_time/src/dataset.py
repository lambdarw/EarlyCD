import os
from datetime import datetime

import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import json

root_path = '/data/zhangruwen/MM_Controversy_Detection_Released-main/mcd/'

class MCDDataset(Dataset):
    def __init__(self, base_path, path_vid):
        self.video_feas = h5py.File(
            os.path.join(base_path, "video_feature_clip_chinese_vit_h.h5"), "r"
        )["video_feas"]
        self.title_feas = h5py.File(
            os.path.join(base_path, "title_feature_clip.h5"), "r"
        )
        self.comment_feas = h5py.File(
            os.path.join(base_path, "comment_feature_clip.h5"), "r"
        )
        self.author_feas = h5py.File(
            os.path.join(base_path, "author_feature_clip.h5"), "r"
        )
        self.asr_feas = h5py.File(
            os.path.join(base_path, "asr_feature_clip.h5"), "r"
        )
        self.topics_feas = h5py.File(
            os.path.join(base_path, "topics_feature_clip.h5"), "r"
        )
        self.data = []
        self.senticNet = pickle.load(
            open(root_path+"dataset/mmcd/senticnet_word.pkl", "rb")
        )
        self.comment_trans = pickle.load(
            open(root_path+"dataset/mmcd/translate/translate_comments.pkl", "rb")
        )
        for k in self.comment_trans.keys():
            self.comment_trans[k] = [
                re.sub(
                    "[0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/\\\[\]'\"\u4e00-\u9fa5]",
                    "",
                    i,
                )
                .strip()
                .lower()
                for i in self.comment_trans[k]
            ]
        # with open(root_path+"dataset/metadata_clean.json", "r", encoding="utf-8-sig") as f:
        #     self.data_complete = f.readlines()
        with open(root_path+"dataset/mmcd/metadata_clean2_cut.json", "r", encoding="utf-8-sig") as f:
            self.hotnum_complete = json.load(f)
        with open(os.path.join(base_path, "data-split/", path_vid), "r") as fr:
            self.vid = fr.readlines()
        self.vid = [i.replace("\n", "") for i in self.vid]

        self.data = []
        for i in self.hotnum_complete:
            if i["video_id"] in self.vid:
                self.data.append(i)
        # 热度数据
        self.hotnum = {}
        for i in self.hotnum_complete:
            if i["video_id"] in self.vid:
                self.hotnum[i["video_id"]] = round(i['hot_num'], 3)
        # 视频发布时间数据
        self.video_time = {}
        for i in self.hotnum_complete:
            if i["video_id"] in self.vid:
                self.video_time[i["video_id"]] = i["publish_time"]
        # 评论发布时间数据, 时间间隔（小时）
        self.comment_times = {}
        for item in self.hotnum_complete:
            if item["video_id"] in self.vid:
                self.comment_times[item["video_id"]] = []
                for com in item['comments'].values():
                    self.comment_times[item["video_id"]].\
                        append(time_span(com["comments_time"], item["publish_time"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = str(item["video_id"])
        # 热度
        # hotNum = self.hotnum[idx]
        hotNum = item["hot_num"]
        # label
        label = torch.tensor(item["controversy"])
        # video
        video_fea = self.video_feas[vid][:]
        video_fea = torch.FloatTensor(video_fea)
        # title
        title_fea = self.title_feas[vid][:]
        title_fea = torch.FloatTensor(title_fea)
        # comments
        if vid in self.comment_feas.keys():
            comments_fea_all = self.comment_feas[vid][:]

            if len(comments_fea_all) > 0:
                comments_fea = torch.FloatTensor(comments_fea_all)
                comment_times = self.comment_times.get(vid, [])  # 安全获取  list of comments timestamp deltas
                comment_lens = len(comments_fea_all)
            else:
                comments_fea = torch.zeros(1, 768)  # 没有评论时用padding
                comment_times = []
                comment_lens = 0
        else:
            comments_fea = torch.zeros(1, 768)  # 没有评论时用padding
            comment_times = []
            comment_lens = 0
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
            "comment_times": comment_times,
            "comment_lens": comment_lens,
            "author_fea": author_fea,
            "asr_fea": asr_fea,
            "topics_fea": topics_fea,
            "hotNum": hotNum,
        }

# 统计时间间距
def time_span(comment_time, create_time):
    comment_time = datetime.fromtimestamp(comment_time)
    publish_time = datetime.fromtimestamp(create_time)
    time_difference = comment_time - publish_time
    span_hours = time_difference.total_seconds() / 3600  # ✅ 保持浮点数精度
    return span_hours  # 返回精确的小时数

def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        if len(video.shape) == 1:
            video = torch.zeros((1, 1024))
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

def pad_time_sequence(seq_len, time_list):
    """对时间序列进行padding"""
    result = []
    masks = []

    for times in time_list:
        ori_len = len(times)

        if ori_len >= seq_len:
            # 截断到seq_len
            padded_times = times[:seq_len]
            mask = torch.ones(seq_len, dtype=torch.int)
        else:
            # padding到seq_len
            padded_times = times + [0] * (seq_len - ori_len)  # 用0填充
            mask = torch.cat([torch.ones(ori_len), torch.zeros(seq_len - ori_len)])

        result.append(torch.IntTensor(padded_times))
        masks.append(mask)

    return torch.stack(result), torch.stack(masks)

def MCD_collate_fn(batch):
    num_comments = 40
    num_frames = 5
    vids = [int(item["vid"]) for item in batch]
    labels = [item["label"] for item in batch]
    video_feas = [item["video_fea"] for item in batch]
    video_feas, video_masks = pad_frame_sequence(num_frames, video_feas)
    asr_feas = [item["asr_fea"] for item in batch]
    topics_fea = [item["topics_fea"] for item in batch]
    title_feas = [item["title_fea"] for item in batch]
    author_feas = [item["author_fea"] for item in batch]
    hotNums = [item["hotNum"] for item in batch]

    comment_lens = [item["comment_lens"] for item in batch]
    comment_feas = [item["comments_fea"] for item in batch]
    comment_times = [item["comment_times"] for item in batch]
    comment_feas, comment_masks = pad_frame_sequence(num_comments, comment_feas)
    comment_times, _ = pad_time_sequence(num_comments, comment_times)

    return {
        "vid": torch.tensor(vids),
        "label": torch.stack(labels),
        "video_feas": video_feas,
        "video_masks": video_masks,
        "title_feas": torch.stack(title_feas),
        "comment_feas": comment_feas,
        "comment_times": torch.IntTensor(comment_times),
        "comment_lens": torch.IntTensor(comment_lens),
        "author_feas": torch.stack(author_feas),
        "asr_feas": torch.stack(asr_feas),
        "topics_fea": torch.stack(topics_fea),
        "hotNums": torch.FloatTensor(hotNums),
    }
