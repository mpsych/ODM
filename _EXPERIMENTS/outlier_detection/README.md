# Completed Whitelist Outlier Detection
## Prior Steps
Having filtered our entire mammogram (2D) and digital breast tomosynthesis (3D) database to a DeepSight-classifier-ready subset [see subsetting directory here](https://github.com/mpsych/omama/tree/main/omama/analysis) 
we proceeded with running the remaining 191,547 2D images through the DeepSight Classifier and retaining all resultant error-free images.

Large-scale classification was scheduled and performed in parallel (parallel_deepsight_run_script.sh, deep_sight_executor.py) 
and subsequent results aggregated (deep_sight_result_scanner.py, scanner_client_code.ipynb) [see code directory here](https://github.com/mpsych/omama/tree/main/omama/deep_sight) 
## Current Steps
We are to employ unsupervised anomaly detection methods to further reduce our collection of 2D images into a set of exemplary mammograms, free from abnormal views or atypical screens.
Before attempting any automated learning on the entire images themselves (a high dimensional problem), we reduce dimensionality to a single dimension by generating an image histogram per image and drawing comparisons across all such histograms.
An image histogram is a type of histogram that acts as a graphical representation of the tonal distribution in a digital image. We use the mahotas library to generate our histograms and experiment with different learning algorithms from the pyOD library 
[see overview of this easy-to-use toolbox, here](https://www.jmlr.org/papers/volume20/19-011/19-011.pdf?ref=https://githubhelp.com) 
## Next Steps
The aforementioned classification and outlier detection steps must be repeated for our remaining 67,534 3D images. 
