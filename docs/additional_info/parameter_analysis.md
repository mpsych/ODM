# Parameter Analysis

The effectiveness of our outlier detection pipeline hinges on a careful selection of parameters. In this section, we elucidate our reasoning behind the choice of certain key parameters and how they influence our model's performance.

## 5-Bin Histogram Filtering

Why not a 256-bin histogram? A 256-bin histogram, while offering a much more detailed distribution of the pixel intensities, also significantly increases the complexity of the data analysis. A larger number of bins can make the patterns less discernible and harder to interpret. The extra granularity might not necessarily provide additional useful information for the task at hand.

With a 5-bin histogram, we were able to identify key patterns (specifically, values below 2,000 in the second bin and a "1" label in the last bin for outliers) in the data that effectively differentiated between satisfactory and unsatisfactory mammograms. These patterns were clear and easy to understand, and increasing the number of bins risked obscuring these important features.

The choice to use a 5-bin histogram over a 2-bin histogram was motivated by our preliminary analysis of the data. While a 2-bin histogram did yield promising results initially, it misclassified a considerable portion of satisfactory images as outliers, indicating that more granularity was needed in our approach.

Upon further exploration of the data, we found that 5-bin histograms provided the balance we needed. Notably, we discovered that outliers generally exhibited specific patterns across the bins: values below 2,000 in the second bin and a "1" label in the last bin, while most other cases displayed a higher value. By using these patterns as thresholds, we were able to improve our detection of outliers significantly while minimizing false positives. This specific pattern in our ground truth labeled data guided the choice of five bins and their respective thresholds.

## Variational Autoencoder (VAE) Parameters

Our model employs a variational autoencoder, which utilizes a set of parameters chosen based on a mix of empirical evidence and common practices in the field. We used default settings for most parameters, given their proven success in a variety of applications. However, users can easily adjust these according to their specific needs.

- **Latent Dimension (`latent_dim`):** The dimensionality of the latent space was set to 2. This was a balancing act between model complexity (higher with larger latent space) and sufficient representation of the data (poorer with a smaller latent space).

- **Activations (`hidden_activation`, `output_activation`):** We used ReLU activation for the hidden layers due to its advantage in mitigating the vanishing gradient problem. For the output layer, a sigmoid function was chosen to map the outputs to a probability, aiding in binary classification tasks.

- **Loss (`loss`):** Mean Squared Error (MSE) was used as the loss function. This is a common choice in regression problems and performed well with our data.

- **Optimizer (`optimizer`):** The Adam optimizer was chosen for its efficient handling of sparse gradients and adaptive learning rates.

- **Batch Size (`batch_size`):** The batch size of 32 represents a compromise between computational efficiency and the granularity of the gradient estimate.

- **Dropout Rate (`dropout_rate`):** Dropout is used as a form of regularization to prevent overfitting. The rate of 0.2 was chosen as it provides a good balance between retaining network capacity and enforcing robustness to noise.

- **L2 Regularizer (`l2_regularizer`):** An L2 regularization term was added to the loss function to help prevent overfitting by penalizing large weights.

Although these parameters were suitable for our specific task, they may need to be tuned for different datasets or problem domains. For that reason, we have designed our code to allow for easy adjustment of these parameters, as needed.

Through this analysis, we strive to shed light on our decision-making process and provide a foundation for others to build upon and explore their own parameter settings in the context of this pipeline. As always, careful validation and testing should be carried out when altering these parameters, to ensure optimal performance.
