# Responses to Reviewers

## Response to Reviewer DJjf:

1. **Non-robust normalization:** We acknowledge the potential limitation of the min-max intensity normalization method in dealing with intensity outliers. It's crucial to note, however, that our initial data cleaning step largely eliminates images that may present extreme pixel intensity values. Nevertheless, to cater to potential outliers, we're considering more robust normalization techniques, such as percentile-based normalization, for future work.

2. **Insufficient 5-bin histogram explanation:** We appreciate the feedback on the clarity of our 5-bin histogram method. To address this, we've added a detailed explanation of the technique and its workings to the ['Methods'](./methods.md) section of our repository. This explanation covers how the histogram bins are calculated, the rationale behind the chosen thresholds, and how these thresholds contribute to the detection of low-quality scans.

3. **Feasibility of discarding images with small artifacts:** We understand the potential concerns regarding the removal of images with small artifacts. In some cases, these images may still hold valuable information for breast cancer detection. We believe, however, that the presence of artifacts may interfere with machine learning model training and performance. To strike a balance, we're exploring ways to enhance our pipeline to detect and possibly correct these minor artifacts instead of discarding the images outright.

## Response to Reviewer Aruq:

1. **Publicly available dataset:** Thank you for pointing this out. We have updated our repository to include the link to access the dataset. Please note that due to privacy considerations, some restrictions may apply.

2. **Parameter analysis:** We agree that a thorough analysis of the parameters used in our method is necessary. We have included in our repository a new section, ['Parameter Analysis'](./parameter_analysis.md), that provides insights into our choice of parameters for both the 5-bin histogram filtering and the variational autoencoder. This analysis covers how we arrived at our specific settings and the impact of these parameters on the performance of our pipeline.
