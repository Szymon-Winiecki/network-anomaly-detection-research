# Research repository for master's thesis on Anomaly detection in network traffic using autoencoders and clustering

The primary objective is to investigate the effectiveness of combining clustering algorithms with autoencoders, which may be beneficial due to the characteristic of network traffic—comprising various protocols and services—that often leads to the formation of natural data clusters. For the purposes of this study, all analyzed models were implemented and their parameters optimized. A unified data preprocessing method was proposed, along with a novel cross-validation algorithm for semi-supervised learning. The experimental results demonstrate that the appropriate integration of clustering methods with autoencoders can improve anomaly detection performance. The best-performing model—Clustering-based Deep Autoencoder (CAE)—which applies a modified K-means algorithm in the latent space during the autoencoder training process, achieved an AUROC of 0.971 on the KDDCUP99 dataset, 0.901 on UNSW-NB15, and 0.996, 0.997, 0.999, and 0.997 on scenarios 8, 9, 10, and 13 of the CTU-13 dataset, respectively.

Keywords: Anomaly detection, autoencoders, clustering, network traffic, network attacks.

This repository contains implementations of the analyzed models, data processing, experiments methodology, experiments tracking tools and results visualization utilities.

## Stack

PyTorch, PyTorch Lightning, Scikit-learn, NumPy, pandas, Matplotlib, MLflow.

## Implemented models

For the purpose of the research, five autoencoder based architectures have been implemented to evaluate their capabilities of anomaly detection in network traffic:

1. Standard autoencoder with reconstruction error as a anomlay score (AE(RE))
2. Shrink Autoencoder with number of one-class classifiers in the latent space (SAE) [[1]](#1)
3. Clustering-based Deep Autoencoder (CAE) [[2]](#2)
4. K-means Shrink Autoencoder (KSAE) [[3]](#3)
5. BIRCH Autoencoder (BAE) [[4]](#4)

## Datasets

Read [`data/README.md`](data/README.md) for more information.

## Data preprocessing

To prepare data from each dataset run the following scripts:

```
code/preprocessing/preprocess_UNSW-NB15.py

code/preprocessing/preprocess_KDDCUP99.py

code/preprocessing/preprocess_CICIDS2017.py

code/preprocessing/preprocess_CTU13.py
```

Once the preprocessing is done, one can remove original data files from `data/*/raw` directories, since prepared data is now stored in `data/*/preprocessed` directories.

## MLFlow

MLFlow is used to experiment tracking and result comparison.

Start MLFlow server with `mlflow server --host 127.0.0.1 --port 8080`


## References

<a id="1">[1]</a>  V. L. Cao, M. Nicolau and J. McDermott, "Learning Neural Representations for Network Anomaly Detection," in IEEE Transactions on Cybernetics, vol. 49, no. 8, pp. 3074-3087, Aug. 2019, doi: 10.1109/TCYB.2018.2838668.

<a id="2">[2]</a>  Nguyen, Van & Viet Hung, Nguyen & Le-Khac, Nhien-An & Cao, Van Loi. (2020). Clustering-Based Deep Autoencoders for Network Anomaly Detection. 10.1007/978-3-030-63924-2_17. 

<a id="3">[3]</a>  T. C. Bui, V. L. Cao, M. Hoang and Q. U. Nguyen, "A Clustering-based Shrink AutoEncoder for Detecting Anomalies in Intrusion Detection Systems," 2019 11th International Conference on Knowledge and Systems Engineering (KSE), Da Nang, Vietnam, 2019, pp. 1-5, doi: 10.1109/KSE.2019.8919446.

<a id="4">[4]</a>  Wang, Dongqi & Nie, Mingshuo & Chen, Dongming. (2023). BAE: Anomaly Detection Algorithm Based on Clustering and Autoencoder. Mathematics. 11. 3398. 10.3390/math11153398. 

