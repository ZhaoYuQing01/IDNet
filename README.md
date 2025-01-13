# IDNet
This an official Pytorch implementation of our paper "IDNet". The specific details of the framework are as follows.
![image](https://github.com/ZhaoYuQing01/IDNet/blob/main/figure/IDNet.png)
# Datasets
* Houston2013 Dataset, which captured scenes over the University of Houston campus and adjacent urban areas, consisted of HSI and LiDAR-based Digital Surface Model(DSM), both with a data size of 349 × 1905 pixels and a spatial resolution of 2.5m. The HSI consists of 144 bands with a wavelength range of 0.38 to 1.05$\mu$m. The dataset has a total of 15029 ground-truth samples, covering 15 categories.
* Trento dataset acquired HSI and LiDAR data from a rural area south of Trento, Italy, with a size of 166 × 600 pixels and a spatial resolution of 1m. HSI has 63 spectral bands with a wavelength range of 0.42 to 0.99$\mu$m. The dataset contained a total of 30214 truth value samples and studied six distinguishable class labels.
* MUUFL dataset obtained HSI and LiDAR data from the University of Southern Mississippi, Gulfport Campus in Long Beach, Mississippi, USA, with a size of 325 × 220 pixels. HSI has 64 available spectral channels ranging from 0.38 to 1.05 $\mu$m with a spatial resolution of 0.54 × 1.0 m. LiDAR data is captured by an ALTM sensor using a laser at a wavelength of 1064 nm. The spatial resolution is 0.60 × 0.78 m. The dataset contains a total of 53,687 truth samples, and 11 distinguishable class labels are studied.
* Augsburg dataset captures the RS scenario for the city of Augsburg. The HSI data includes 180 spectral bands in the 400-2500 nm wavelength range. The LiDAR data consists of 332 × 485 pixels and depicts seven unique land cover categories.
# Train FDNet
 ```
python demo.py
```
# Results
All the results presented here are referenced from the original paper.
| Dataset | OA(%) | AA(%) | AA(%) |
| --- | --- | --- | --- |
| Houston2013 | 96.74 | 97.25 | 96.47 |
| Trento | 99.62 | 99.37 | 99.49 |
| MUUFL | 89.12 | 89.36 | 85.78 |
| Augsburg | 88.74 | 78.61 | 84.33 |
# Citation
```
@ARTICLE{,
  author={},
  journal={},
  title={},
  year={},
  volume={},
  pages={},
  doi={}
}
```
# Contact
