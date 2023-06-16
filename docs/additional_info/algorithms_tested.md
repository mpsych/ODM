# Algorithms

After normalization and image feature calculation, we performed unsupervised outlier detection on Datasets A, B, C, A*, and B*, using 26 different categories of algorithms as referenced in [Zhao et al. (2019, 2021)](#).

## Probabilistic Detectors

- **Copula-Based Outlier Detector (COPOD):** Constructs a distribution function and identifies outliers at the extremes of the distribution [Li et al. (2020)](#).
- **Empirical Cumulative Distribution Functions (ECOD):** Uses a modeled step-function [Li et al. (2022)](#).
- **Kernel Density Estimation for Unsupervised Outlier Detection (KDE):** Employs the negative log probability density [Latecki et al. (2007)](#).
- **Outlier detection based on Sampling:** Uses statistical sampling techniques [Sugiyama et al. (2013)](#).
- **Stochastic Outlier Selection (SOS):** Uses the concept of affinity for proportional similarity between data points [Janssens et al. (2012)](#).

## Linear Models

- **Principal Component Analysis Outlier Detector (PCA):** Employs a linear dimensionality reduction using singular value decomposition [Shyu et al. (2003), Aggarwal et al. (2016)](#).
- **Linear Model Deviation-based outlier detection (LMDD):** Uses a smoothing factor to indicate dissimilarity [Arning et al. (1996), Zhao et al. (2019)](#).
- **One-class Support Vector Machine detector (OCSVM):** Optimizes a high-dimensional distribution controlled through weight vectors [Schölkopf et al. (2001)](#).

## Proximity-based Techniques

- **Nearest Neighbors Detector using the mean distance (AvgKNN):** Uses the mean distance to neighbors as the outlier score [Angiulli et al. (2002), Ramaswamy et al. (2000)](#).
- **Clustering Based Local Outlier Factor (CBLOF):** Recasts detection as a clustering problem and calculates the distance to the nearest large cluster [He et al. (2003)](#).
- **Connectivity-Based Outlier Factor (COF):** Yields the ratio of average chaining distance for nearest neighbors [Tang et al. (2002)](#).
- **Histogram-based Outlier Detection (HBOS):** Uses binning according to the Birge-Rozenblac method [Birgé et al. (2006), Goldstein et al. (2012)](#).
- **k-Nearest Neighbors Detector (KNN):** Measures the distance between neighbors [Angiulli et al. (2002), Ramaswamy et al. (2000)](#).
- **Local Outlier Factor (LOF):** Measures the local deviation of density with respect to its neighbors [Breunig et al. (2000)](#).
- **Nearest Neighbors Detector using the median distance (MedKNN):** Uses the median distance to neighbors as the outlier score [Angiulli et al. (2002), Ramaswamy et al. (2000)](#).
- **Subspace Outlier Detection (SOD):** Compares axis-parallel subspaces in a high-dimensional feature space [Kriegel et al. (2009)](#).

## Ensembles

- **Feature bagging detector (FB):** Combines multiple outlier detection methods and maximizes the scores [Lazarevic et al. (2005)](#).
- **IsolationForest Outlier Detector (IForest):** Analyzes path lengths in created tree structures of features [Liu et al. (2008, 2012)](#).
- **Lightweight on-line detector of anomalies (LODA):** Leverages an ensemble of weak detectors [Pevny et al. (2016)](#).
- **Scalable Unsupervised Outlier Detection (SUOD):** Optimizes a modular acceleration system [Zhao et al. (2021)](#).

## Neural Networks

- **Anomaly Detection with Generative Adversarial Networks (AnoGAN):** Where two artificial neural networks compete with each other to make accurate outlier predictions [Schlegl et al. (2017)](#).
- **Fully-connected Auto Encoder (AE):** For dimensionality reduction and outlier detection in latent space [Aggarwal et al. (2016)](#).
- **Deep One-Class Classification for outlier detection (DeepSVDD):** Trains an artificial neural network while minimizing the volume of a hyper-sphere that surrounds the data and calculating the distance to the center [Ruff et al. (2018)](#).
- **Single-Objective Generative Adversarial Active Learning (SO-GAAL):** Based on a mini-max game between generator and discriminator networks [Liu et al. (2019)](#).
- **Variational Auto Encoder (VAE):** For continuous representations in the latent space for reducing the dimensionality [Kingma et al. (2013), Burgess et al. (2018)](#).
