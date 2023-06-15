# Methodology: Exploring 5-Bin Histograms

Our method for filtering low-quality mammograms heavily depends on normalization and binning techniques. We observed that the configuration of histogram bins played a vital role in identifying the optimal-performing algorithm. Further analysis revealed an intriguing pattern with a specific number of bins.

## Ablation Studies on Histogram Bins

During the preliminary trials with Gaussian normalization, the most promising results in detecting low-quality images were obtained using a two-bin histogram. However, this approach misclassified a substantial portion (approximately 20% to 25%) of satisfactory images, as outliers often exhibited values exceeding 15,000 in the second bin. Interestingly, a few labeled outliers with a "1" in the second bin went undetected by machine learning algorithms and the simple threshold-based approach when we used only two-bin histograms.

Intrigued by these results, we performed ablation studies to investigate whether adding more bins or adjusting the Gaussian blur could improve the two-bin methodology. We also assessed whether alternative normalization techniques configured to produce only two bins could yield comparable results. Min-max normalization was found to outperform Gaussian normalization when constrained to generate two bins.

## 5-Bin Histograms and Outlier Detection

Our exploratory analysis revealed that when histograms were constructed with small bin sizes, the last bin retained the properties of the second bin found in the two-bin histograms. A distinct pattern emerged when utilizing five bins. Specifically, outliers labeled with a "1" in the last bin displayed a value below 2,000 in the second bin, with nearly all other cases demonstrating a much higher value.

## Final Methodology

Capitalizing on these insights, we implemented a method involving a five-bin histogram configuration. By identifying and applying thresholds to the two key bins within this histogram space, we were able to significantly improve the detection of unwanted images while minimizing the inclusion of acceptable ones. This approach, however, still has certain limitations and warrants further investigation to assess its performance across different datasets and imaging modalities.
