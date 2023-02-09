# DFCN

Unofficial Pytorch implementation of “[Multi-slice compressed sensing MRI reconstruction based on deep fusion connection network](https://pubmed.ncbi.nlm.nih.gov/35944808/)“

DFCN is a recent work on Multi-slice MRI reconstruction. According to the information provided in the paper, I implemented the model structure of Pytorch version and packaged it to make it applicable to **Multi-coil** MRI data. If you are interested in this work, please click the [link](https://pubmed.ncbi.nlm.nih.gov/35944808/) to learn more about the DFCN paper. Thanks to DFCN authors for their contributions to MRI multi-slice reconstruction.

### Abstract

![./figures/figure1.png]()

Recently, magnetic resonance imaging (MRI) reconstruction based on deep learning has become popular. Nevertheless, reconstruction of highly undersampled MR images is still challenging due to severe aliasing effects. In this study we built a deep fusion connection network (DFCN) to efficiently utilize the correlation information between adjacent slices. The proposed method was evaluated with online public IXI dataset and Calgary-Campinas-359 dataset. The results show that DFCN can generate the best reconstruction images in de-aliasing and restoring tissue structure compared with several state-of-the-art methods. The mean value of the peak signal-to-noise ratio could reach 34.16 dB, the mean value of the structural similarity is 0.9626, and the mean value of the normalized mean square error is 0.1144 on T2-weighted brain data of IXI dataset under 10× acceleration. Additionally, the mean value of the peak signal-to-noise ratio could reach 30.17 dB, the mean value of the structural similarity is 0.9259, and the mean value of the normalized mean square error is 0.1294 on T1-weighted brain data of Calgary-Campinas-359 dataset under 10× acceleration. With the correlation information between adjacent slices as prior knowledge, our method can dramatically eliminate aliasing effects and enhance the reconstruction quality of undersampled MR images.

![./figures/figure2.png]()

### Usage

```python
import torch
from model.MSMC_model import MSMC_model
from model.DFCN import DFCN

## Please refer to fastmri for data processing
mask_kspace = torch.randn(2, 16, 7, 256, 256,2)  ## batch,coil,slices,h,w,2
mask = torch.randn(2, 16, 7, 256, 256,2)  ## batch,coil,slices,h,w,2  dtype is bool

model = MSMC_model(DFCN(slices=7),sens_chans=8,sens_pools=4)
recon = model(mask_kspace,mask)
print(model)
print(recon)
```

Tips: I did not provide the complete training and testing code. I believe that for most students looking for open source code (at least for me), the data processing of many MRI reconstruction open source code is always so tedious and the data forms are so diverse that we usually have to look at the data processing code for a long time to understand what data to input into the model. And they name variables so casually. Even some only give the processed sample data, we have to download and interpret, these are too troublesome. We just want to apply it to our own data.😔

### Citations

```bibtex
@article{PengShangguan2022MultisliceCS,
  title={Multi-slice compressed sensing MRI reconstruction based on deep fusion connection network},
  author={Peng Shangguan and Wenjie Jiang and Jiechao Wang and Jian Wu and Congbo Cai and Shuhui Cai},
  journal={Magnetic Resonance Imaging},
  year={2022}
}
```

