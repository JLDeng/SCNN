# Sctructured Component-based Neural Network (SCNN)
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtained the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). Then place the downloaded data under the folder `./dataset`.

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
bash ./scripts/long_term_forecast/ETT_script/SCNN_ETTh1.sh
```

## Citation

If you find this repo useful, please cite our paper.

```
@article{deng2024disentangling,
  title={Disentangling Structured Components: Towards Adaptive, Interpretable and Scalable Time Series Forecasting},
  author={Deng, Jinliang and Chen, Xiusi and Jiang, Renhe and Yin, Du and Yang, Yi and Song, Xuan and Tsang, Ivor W},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Jinliang Deng (jinliang.deng@student.uts.edu.au)

or describe it in Issues.

## Acknowledgement

This repo is constructed based on the following library:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library

All the experiment datasets are public and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer
