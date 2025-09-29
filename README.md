# MECP: Early-stage Controversy Detection and Popularity forecasting

## 📌 Overview
MECP is an early-stage multimodal framework that jointly models controversy detection and popularity prediction via a video-guided viewpoint alignment module and a task-aware Mixture-of-Experts component.

<div align=center>
<img src="https://github.com/lambdarw/MECP/blob/main/framework.png" width="70%" >
</div>

## 🧷 Dataset
We construct the first English-language short-video dataset TMCD, with 3,061 videos covering 244 real-world topics, annotated with both controversy and popularity scores.

We evaluate our method for event detection on the [TMCD](https://pan.quark.cn/s/d55564522cd3) and [MMCD](https://github.com/skylie-xtj/MM_Controversy_Detection_Released) datasets.


## 🚀 Quick Start

**Step1: Install the dependencies using pip**

Our dependencies:
``` bash
Python: 3.10.6
torch==2.1.0+cu121
```
Environment requirement:
``` python
pip install -r requirements.txt
```

**Step2: Start training and testing**
``` python
python main.py
```

## 📃 Citation
Please cite our repository if you use MECP in your work.
```bibtex
```
