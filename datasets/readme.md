# TMCD dataset

## ✏️ Collecting the Data

We collected TikTok data through keyword-based crawling between March and April 2025. The dataset comprises four key components: 
(1) raw video content, 
(2) video metadata (including descriptions, like counts, and comment statistics),
(3) comment metadata (textual content, creator engagement flags, and timestamps), 
and (4) publisher profile (with follower counts and historical video counts).

## 📥 Download Dataset
The dataset is publicly accessible and can be downloaded from [URL will be open after publication].

## 📈 Dataset Analysis
We developed a keyword list of 244 terms sourced from trending topics on both TikTok and X (formerly Twitter). 

After excluding non-English content and videos containing only background music (retaining only videos with substantive creator commentary), 
we obtained a balanced dataset comprising 1,336 controversial and 1,725 non-controversial videos. 

| Category      | Statistic                     | Controversial Video  | Non-Controversial Video  |
|---------------|-------------------------------|----------------------|--------------------------|
| Video         | Avg. Duration of Video (s)    | 95.34                | 83.51                    |
|               | Avg. Number of Forwards       | 118,597              | 123,549                  |
|               | Avg. Number of Likes          | 11,483               | 11,029                   |
|               | Avg. Number of Comments       | 4,782                | 2,212                    |
| Publisher     | Avg. Number of Videos         | 2,447                | 1,434                    |
|               | Avg. Number of Likes Received | 9,837                | 15,090                   |
|               | Avg. Number of Followers      | 2,269,228            | 1,295,616                |

## 📝 Popularity score analysis
To quantify video popularity across TMCD dataset, we develop a normalized scoring metric based on comment count.  
We normalize raw comment counts into range [0,1]:

s(v) = [ln(1+c) - ln(1+c_min)]/[ln(1+c_max) - ln(1+c_min)],

where c represents comment counts, with c_min and c_max representing the minimum and maximum values in the dataset, respectively.
