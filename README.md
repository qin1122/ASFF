# ASFF
This repository contains the codes corresponding to the MICCAI 2024 paper:

Yuqi Fang, Wei Wang, Qianqian Wang, Hong-Jun Li, Mingxia Liu, "Attention-Enhanced Fusion of Structural and Functional MRI for Analyzing HIV-Associated Asymptomatic Neurocognitive Impairment", International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2024.

### Environment
- Python==3.11
- CUDA==11.8
- Pytorch==2.4.0
- einops=0.8.0
- info-nce-pytorch==0.1.4
- scikit-learn=1.5.2

### Dataset
We used a private dataset, to protect the patients' privacy, we can't share the dataset.

### Training & Inference
```
python main.py
```
### Result
You can find the results of this project in "ASFF/checkpoints_twoindivbr_separate"

### Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@inproceedings{fang2024attention,
  title={Attention-Enhanced Fusion of Structural and Functional MRI for Analyzing HIV-Associated Asymptomatic Neurocognitive Impairment},
  author={Fang, Yuqi and Wang, Wei and Wang, Qianqian and Li, Hong-Jun and Liu, Mingxia},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={113--123},
  year={2024},
  organization={Springer}
}
```
