# IDNet
This an official Pytorch implementation of our paper "IDNet: Intensity-Constrained Detail-Enhanced Network for Hyperspectral and LiDAR Collaborative Classification". The specific details of the framework are as follows.
![image](https://github.com/ZhaoYuQing01/IDNet/blob/main/figure/IDNet.png)
# Datasets
* [The Houston2013 dataset](https://github.com/songyz2019/fetch_houston2013) includes a hyperspectral image (HSI) and a LiDAR-based digital surface model (DSM), collected by the National Center for Airborne Laser Mapping (NCALM) using the ITRES CASI-1500 sensor over the University of Houston campus in June 2012. The HSI comprise 144 spectral bands covering a wavelength range from 0.38 to 1.05 µm while LiDAR data are provided for a single band. Both the HSI and LiDAR data share dimensions of 349 × 1905 pixels with a spatial resolution of 2.5 m. The dataset contains 15 categories, with a total of 15,029 real samples available.
* [The Trento dataset](https://github.com/pagrim/TrentoData) comprises HSI and LiDAR data obtained from southern Trento, Italy. The HSI was collected by an AISA Eagle sensor, consisting of 63 spectral bands with a wavelength range from 0.42 to 0.99 µm. LiDAR data with 1 raster were acquired by the Optech ALTM 3100EA sensor. The scene consists of 166 × 600 pixels, with a spatial resolution of 1 m. This dataset contains 6 land cover types with a total of 30,214 real samples.
* [The MUUFL dataset](https://github.com/GatorSense/MUUFLGulfport) was acquired in November 2010 over the area of the campus of University of Southern Mississippi Gulf Park, Long Beach Mississippi, USA. The HSI data was gathered using the ITRES Research Limited (ITRES) Compact Airborne Spectral Imager (CASI-1500) sensor, initially comprising 72 bands. Due to excessive noise, the first and last eight spectral bands were removed, resulting in a total of 64 available spectral channels ranging from 0.38 to 1.05 µm. LiDAR data was captured by an ALTM sensor, containing two rasters with a wavelength of 1.06 µm. The dataset consists of 53,687 groundtruth pixels, encompassing 11 different land-cover classes.
* [The Augsburg dataset](https://github.com/danfenghong/ISPRS_S2FL?tab=readme-ov-file) was captured over Augsburg, Germany. HSI data were gathered using the DAS-EOC HySpex sensor, while LiDAR-based DSM data were acquired through the DLR-3K system. To facilitate multimodal fusion, both images were down-sampled to a uniform resolution of 30m. This dataset contains HSI data with 180 bands ranging from 0.4 to 2.5µm and DSM data in a single band. With dimensions of 332 × 485 pixels, the dataset represents 7 distinct land-cover categories.
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
