# ReferEverything [ICCV 2025]

[**Anurag Bagchi**](https://miccooper9.github.io) · 
[**Zhipeng Bao**](https://zhipengbao.github.io) · 
[**Yu-Xiong Wang**](https://yxw.web.illinois.edu) · 
[**Pavel Tokmakov**](https://paveltokmakov.github.io) · 
[**Martial Hebert**](https://www.ri.cmu.edu/ri-faculty/martial-hebert/)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/arxiv-blue)](https://arxiv.org/abs/2410.23287)
[![Project Page](https://img.shields.io/badge/Website-orange)](https://refereverything.github.io/)


Official PyTorch implementation of the ICCV 2025 paper **"ReferEverything"**.

---

## TL;DR:
 
> We present **Refer Everything Model (REM)** by re-purposing Text-to-Video generation models to zero-shot segment any concept in a Video using Text. 




[![Watch the video](https://img.youtube.com/vi/NEZoGG-xUS8/0.jpg)](https://www.youtube.com/watch?v=NEZoGG-xUS8)



## 📰 News

 
- **[Oct, 2025]** Released the **code** and pretrained **checkpoints** for **ModelScopeT2V-1.4B** and **Wan2.1-14B**.  




## 📦 Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/ReferEverything.git
cd ReferEverything
```



### 2. Install the Modelscope REM environment 
```bash
conda env create -f MS_env.yml
conda activate MS_env
```


### 3. Install the Wan REM environment 
```bash
conda env create -f Wan_env.yml
conda activate Wan_env
```

### 4. Download Checkpoints

Finetuned checkpoints for both models can be downloaded from [Huggingface](https://huggingface.co/anuragba/refereverything/tree/main)






## Run REM on your samples

### Modelscope

```bash
bash run_REM_MS_sample.sh #Change the arguments in the script accordingly.
```

### Wan2.1

The Wan2.1-T2V-14B model is quite large. Please download the base Wan2.1-T2V-14B model from [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) to an approriate disk with enough space.


```bash
bash run_REM_Wan14b_sample.sh #Change the arguments in the script accordingly.
```

## Data Preparation

We use RefCOCO/+/g and Refer-Youtube to train REM. Please follow [ReferFormer](https://github.com/wjn922/ReferFormer) to prepare the training data.


## Train REM

### ModelScope

Train the spatial weights on Refcoco/+/g
```bash
bash train_REM_MS_imgs.sh #Change the arguments in the script accordingly.
```

Train on Refer-Youtube
```bash
bash train_REM_MS_vid.sh #Change the arguments in the script accordingly.
```

### Wan2.1

To save memory during training we pre-compute the T5 text embeddings using ```utils/encode_wantxt_T5.py```

Train jointly on Refer-Youtube and Refcoco/+/g
```bash
bash train_REM_Wan.sh #Change the arguments in the script accordingly.
```

## Infer on datasets

Please follow the instructions in [Ref-Davis](./ref-davis/ref_davis.md), [Ref-Youtube](./ref-yt/ref_yt.md), [Burst](./burst/burst.md), [VSPW-stuff](./stuff/vspw.md)