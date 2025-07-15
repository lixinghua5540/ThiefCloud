###  Encoder-free Multi-axis Physics-aware Fusion Network for Remote Sensing Image Dehazing

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TGRS-blue)](https://ieeexplore.ieee.org/document/10287960)  [![Project](https://img.shields.io/badge/Project-GitHub-gold)](https://github.com/chdwyb/EMPF-Net)

**Abstract**: Current methods for remote sensing image dehazing confront noteworthy computational intricacies and yield suboptimal dehazed outputs, thereby circumscribing their pragmatic applicability. To this end, we propose EMPF-Net, a novel encoder-free multi-axis physics-aware fusion network that exhibits both light-weighted characteristics and computational efficiency. In our pipeline , we contend that conventional u-shaped networks allocate substantial computational resources to encode haze-degraded features, which play a subordinate role in the reconstruction process. Consequently, our encoder stages solely incorporate down-sampling operations. To improve the representation efficiency and enhance the generalization capabilities, we devise a multi-axis partial queried learning block (MPQLB) that primarily concentrates on learning dimension-wise queries, instead of relying solely on strictly-correlated content of the input features. Furthermore, we augment the reconstruction procedure by incorporating ground truth supervision into each stage via a supervised cross-scale transposed attention module (SCTAM). It calculates attention maps under the guidance of clean images, thereby suppressing less informative features to propagate to the subsequent level. In addition, to address the challenge of ineffective intral-level feature fusion, which result in insufficient elimination of haze-degraded information and negatively impact the quality of reconstructed images, we introduce a physics-aware intra-level fusion module (PIFM). This module harnesses a physical inversion model to facilitate the intra-level feature interaction and alleviate the interference of dehazing-irrelevant information. Our proposed EMPF-Net is evaluated on 12 publicly available datasets, and the experimental results substantiate our superiority in terms of both metrical scores and visual quality, despite being equipped with a modest parameter count of 300 K.

### Requirements
```python
python 3.8.6
torch 1.9.0
torchvision 0.11.0
pillow 9.2.0
scikit-image 0.19.3
timm 0.6.7
tqdm 4.64.0
opencv-python 4.5.2.54
```
### Demo
We offer the `demo.py` file to facilitate quick verifications. Kindly access the pre-trained models of our EMPF-Net through the provided links for download.

| Dataset           | Link                                                         | Dataset              | Link                                                         |
| ----------------- | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| StateHaze1K-thick | [[Baidu Cloud](https://pan.baidu.com/s/1ka78xdfXE1DHwQhvHeO4sQ), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1UryeNoDfF9VYC-OM9ySGEoFuICfgOzqI/view?usp=sharing)] | StateHaze1K-moderate | [[Baidu Cloud](https://pan.baidu.com/s/1qYAC3fZmojMoJcx_1mmP1A), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1qSiFiBTIdjBWPtiJHDEjsQkpcceCzFpA/view?usp=sharing)] |
| StateHaze1K-thin  | [[Baidu Cloud](https://pan.baidu.com/s/15JdjFfkmpAQQxoWH0K2skQ), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1oOUrEj9jEG0tvxvLhzBdP8cCDErp0Ult/view?usp=sharing)] | RS-Haze              | [[Baidu Cloud](https://pan.baidu.com/s/1lO43sfqxZZZuO3EXCZEF2Q), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1vMEynk9CCoSmskP63Zkqjo9_bQSA9t1y/view?usp=sharing)] |
| LHID              | [[Baidu Cloud](https://pan.baidu.com/s/1OFSAmGXU3cn8PP1M33Nqmg), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1mZFia198BTZyz-EY4N2skCRTtcVZlRB0/view?usp=sharing)] | DHID                 | [[Baidu Cloud](https://pan.baidu.com/s/1QjHeNoj06yMDxb7o-MSImg), code: rsid]   [[Google Drive](https://drive.google.com/file/d/182EwHYiwNTYBaZvRd7k80NXn_ZX1f1Hu/view?usp=sharing)] |
| RICE1             | [[Baidu Cloud](https://pan.baidu.com/s/1MVDapqY6ZsFDM1OuwgVruA), code: rsid]   [[Google Drive](https://drive.google.com/file/d/13k5bLCaKirrG4QUm2HnQRC_ihAPK8JSA/view?usp=sharing)] | RICE2                | [[Baidu Cloud](https://pan.baidu.com/s/1gytFQlWuMjYAV9qlZpb0Mw), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1pomHZNaBq6G8wGVKBL2iirXnZFbXj_8D/view?usp=sharing)] |
| RSID              | [[Baidu Cloud](https://pan.baidu.com/s/1YTe77eQg5jOTBnJ4aP3jPQ), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1NAJqjfhy8o4uZhydcx4tzTRquzopWzoE/view?usp=sharing)] | Dense-Haze           | [[Baidu Cloud](https://pan.baidu.com/s/1JLeji4_HXutS-pp4VI5mNg), code: rsid]   [[Google Drive](https://drive.google.com/file/d/1G62Jji01P_HDd8YO-Z77i790qAEcNL1N/view?usp=sharing)] |
| NH-Haze           | [[Baidu Cloud](https://pan.baidu.com/s/1sg1X64DuCI-2HqV-DTB5sw), code: rsid]   [[Google Drive](https://drive.google.com/file/d/19i4fqAmIihZHP_saUUWUscOFFJFRKnXS/view?usp=sharing)] |                      |                                                              |

Once you have downloaded the models, you can process a remote sensing hazy image using the following example usage.
```python
python demo.py --input_image ./data/input/0001.png --target_image ./data/target/0001.png --result_dir ./data/result --expand_factor 128 --result_save True --resume_state ./models/RS-Haze.pth --only_last True --cuda True
```
### Train
Our code is built upon the [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox, and we express our gratitude for their valuable contributions. Additionally, for those who wish to perform rapid training on their custom datasets, we provide a straightforward training code in the `train.py` file, enabling training of our EMPF-Net. Please refer to the example usage within the file to train the model on your datasets.
```python
python train.py --seed 2023 --epoch 200 --batch_size_train 10 --batch_size_val 10 --patch_size_train 512 --patch_size_val 512 --lr 1e-3 --lr_min 1e-8 --train_data ./StateHaze1K-thick/train --val_data ./StateHaze1K-thick/val --resume_state ./model_resume.pth --save_state ./model_best.pth --cuda True --val_frequency 3 --loss_weight 0.04 --only_last False --autocast True --num_works 4
```
### Test
To evaluate our EMPF-Net on your own datasets or publicly available datasets, you can utilize the following example usage to conduct experiments.

```python
python test.py --val_data ./Haze1k_thick/test --result_dir ./Haze1k_thick/test/result/ --resume_state ./models/Haze1K-thick.pth --expand_factor 128 --result_save True --cuda True --only_last True --num_works 4
```
### Dataset

If you intend to conduct experiments on our collected real-world remote sensing hazy dataset, named RRSD300, please download it from [[Baidu Cloud](https://pan.baidu.com/s/1lM9vEvDwgDrCoyPJAW490A), code: rsid] or [[Google Drive](https://drive.google.com/file/d/198dmAL5Vrw1qm_f5t4nW8l1Jmw-HNLuy/view?usp=sharing)].

### Citation

If you find our work helpful for your research, please consider citing our work following this.

```python
@article{wen2023encoder,
  title={Encoder-free Multi-axis Physics-aware Fusion Network for Remote Sensing Image Dehazing},
  author={Wen, Yuanbo and Gao, Tao  and Zhang, Jing and Li, Ziqi and Chen, Ting},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE},
  doi={10.1109/TGRS.2023.3325927}
}
```

### Contact  us

If I have any inquiries or questions regarding our work, please feel free to contact us at [wyb@chd.edu.cn](mailto:wyb@chd.edu.cn).
