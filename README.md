# [TCSVT'2025] ThiefCloud 

The official code of paper [*"ThiefCloud: A Thickness Fused Thin Cloud Removal Network for Optical Remote Sensing Image With Self-Supervised Learnable Cloud Prior"*](https://ieeexplore.ieee.org/document/11071276), <u>*TCSVT'2025*</u>, Accepted.

Abstract:Optical remote sensing images are frequently contaminated by thin clouds, thus causing great challenges for subsequent applications. To address this issue, numerous methods guided by cloud features have been developed. However, <u> the cloud features utilized in these methods are generally either unlearnable or lack cloud thickness data constraints</u>, which may further mislead the cloud removal. In this paper, a THIcknEss Fused thin cloud removal network (ThiefCloud) with self-supervised learnable cloud prior is proposed. Firstly, in order to provide reliable cloud prior, a self-supervised cloud prior model (SCPM) is introduced. Secondly, an adaptive feature extraction (AFE) module efficiently extracts the cloud information of the original cloud image, and a physically guided feature fusion (PGFF) module, inspired by the atmospheric scattering model, accurately restores more realistic details. Finally, to enhance the generalizability of SCPM in real scenarios, a staged training strategy is adopted. SCPM is trained independently on the simulated thickness maps and cloud images in advance, then SCPM can guide ThiefCloud. During the training of ThiefCloud, SCPM is frozen initially and then tunable. The frozen SCPM provides effective cloud prior to the non-converged ThiefCloud. The tunable SCPM makes the cloud prior learnable, better aligning with real-world cloud removal. Experimental results demonstrate that compared with other 11 methods, ThiefCloud could achieve competitive results on three public datasets, namely T-CLOUD, RICE and SateHaze1k datasets.
![](Figure/Fig1.png)

*This research has been conducted at the [Multi-temporal Artificial Intelligent Remote Sensing (MAIRS)](https://jszy.whu.edu.cn/lixinghua2/zh_CN/index.htm).*
    
This is the official <u>*PyTorch*</u> implementation of the thin cloud removal method for for optical remote sensing image.
    
### Table of content
 1. [Preparation](#preparation)
 2. [Usage](#usage)
 3. [Paper](#paper)
 4. [Results](#results)
 5. [Acknowledgement](#acknowledgement)
 6. [License](#license)

---
### Preparation
- Package requirements: The scripts in this repo are tested with `torch==1.9` and `torchvision==0.10` using a single NVIDIA RTX A5000 GPU with 24 GB of memory.
- Remote sensing datasets used in this repo:
  - [T-CLOUD dataset](https://github.com/haidong-Ding/Cloud-Removal)
  - [RICE dataset](https://github.com/BUPTLdy/RICE_DATASET)
  - [SateHaze1k dataset](https://www.kaggle.com/datasets/mohit3430/haze1k/data) 


---
### Usage
- overall workflow

  ![](Figure/Fig2.png)

<ba>

- Thin cloud simulation
  please refer to [SatelliteCloudGenerator](https://github.com/strath-ai/SatelliteCloudGenerator)
<ba>
<ba>
- Thin cloud detection
  ```
  train: python ThiefCloud/SCPM/src/train_cd.py
  test: python ThiefCloud/SCPM/src/test_cd.py
  ```
  noticed: SCPM is trained on the paired cloud simulation dataset.

<ba>
<ba>

- Thin cloud removal
  ```
  train: python ThiefCloud/ThiefCloud/train.py
  test: python ThiefCloud/ThiefCloud/test.py
  ```

---
### Results
- Example results and details of different methods on the T-CLOUD dataset:
![](Figure/Fig3.png)

- Example results and details of different methods on the RICE dataset:
![](Figure/Fig4.png)


---
### Paper
**[ThiefCloud: A Thickness Fused Thin Cloud Removal Network for Optical Remote Sensing Image With Self-Supervised Learnable Cloud Prior](https://ieeexplore.ieee.org/document/11071276)**

Please cite the following paper if the code is useful for your research:

```
@ARTICLE{11071276,
  author={Zhao, Anqi and Feng, Ruitao and Li, Xinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={ThiefCloud: A Thickness Fused Thin Cloud Removal Network for Optical Remote Sensing Image With Self-Supervised Learnable Cloud Prior}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2025.3585720}}
```

---
### Acknowledgement
Cloud detection network is improved based on [DCNet](https://github.com/YLiu-creator/deformableCloudDetection),  Cloud removal network is improved based on [EMPF-Net](https://ieeexplore.ieee.org/document/10287960).

The authors would like to thank the assistance provided by the aforementioned links.



### License
The code can be used for academic purposes only.
