# MNIST Classification with Fitted Device (PiecewiseStepDevice)

## Experiment Overview

This folder contains the results of MNIST handwritten digit classification using an **Analog Neural Network** with **Fitted Device** (PiecewiseStepDevice) based on experimental organic memristor data.

### Model Architecture
- **Input Layer**: 784 neurons (28x28 flattened images)
- **Hidden Layer 1**: 256 neurons (Sigmoid activation)
- **Hidden Layer 2**: 128 neurons (Sigmoid activation)
- **Output Layer**: 10 neurons (LogSoftmax activation)
- **Total Parameters**: ~236,000 weights

### Optimal Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.2 | Initial learning rate for AnalogSGD |
| Step Size | 15 | Epochs between LR decay |
| Gamma | 0.7 | LR decay factor |
| Initialization | Xavier | Xavier uniform initialization |
| Epochs | 50 | Total training epochs |
| Batch Size | 128 | Mini-batch size |

### Final Results
- **Best Validation Accuracy**: 95.77%
- **Misclassified Samples**: 423/10,000 (4.23%)

---

## Folder Structure and Data Interpretation

### 01_metrics/
Training progress metrics over 50 epochs.

#### Files:
- **metrics.xlsx**: Raw training data
- **metrics.png / metrics_paper.png**: Visualization plots

#### Data Columns (metrics.xlsx):
| Column | Description | Unit | Interpretation |
|--------|-------------|------|----------------|
| Epoch | Training iteration number | 1-50 | x-axis for plots |
| Train_Loss | NLLLoss on training set | - | Lower is better, should decrease |
| Train_Accuracy | Correct predictions on training set | % | Should increase, may plateau |
| Val_Loss | NLLLoss on validation set | - | Key metric for generalization |
| Val_Accuracy | Correct predictions on validation set | % | **Primary performance metric** |

#### How to Interpret:
1. **Convergence**: Both losses should decrease; if Val_Loss increases while Train_Loss decreases, model is overfitting
2. **Train-Val Gap**: Small gap (<5%) indicates good generalization
3. **Plateau**: Accuracy stabilizing around epoch 30-40 indicates convergence
4. **Best Epoch**: The epoch with highest Val_Accuracy (not necessarily the last)

---

### 02_weight_distribution/
Analysis of synaptic weight values in the analog network.

#### Files:
- **weight_distribution.xlsx**: Histogram data for all epochs
- **weight_stats.xlsx**: Statistical summary per epoch
- **weight_distribution_paper.png**: Publication-quality visualization

#### Data Columns (weight_distribution.xlsx):
| Column | Description |
|--------|-------------|
| Bin_Center | Center of histogram bin (-1.2 to +1.2) |
| Epoch_1_Density | Probability density at epoch 1 |
| Epoch_10_Density | Probability density at epoch 10 |
| ... | Additional epochs (10, 20, 30, 40, 50) |

#### Data Columns (weight_stats.xlsx):
| Column | Description | Ideal Range |
|--------|-------------|-------------|
| Epoch | Training epoch | - |
| Mean | Average weight value | Close to 0 |
| Std | Standard deviation | 0.1-0.5 typical |
| Min | Minimum weight | > -1.0 |
| Max | Maximum weight | < +1.0 |
| Median | Median weight value | Close to 0 |

#### How to Interpret:
1. **Saturation Boundaries (±1)**: Weights hitting ±1 are "saturated" - the device cannot change further
2. **Distribution Shape**:
   - Initial (Xavier): Gaussian-like, centered at 0
   - Final: May show bimodal peaks near boundaries (saturation effect)
3. **Weight Spread**: Wider distribution = more diverse learned features
4. **Saturation Analysis**:
   - Few weights at ±1 boundary = good (device operating in linear region)
   - Many weights at ±1 = concerning (device non-linearity limiting learning)

#### Key Insight for Analog Devices:
The PiecewiseStepDevice has non-linear weight update characteristics. Weight saturation at boundaries (±1) indicates the device's conductance limits. A healthy distribution should have most weights within [-0.8, +0.8] range.

---

### 03_confusion_matrix/
Classification performance breakdown by digit class.

#### Files:
- **confusion_matrix.xlsx**: 10x10 matrix of predictions
- **confusion_matrix.png**: Heatmap visualization

#### Data Structure (confusion_matrix.xlsx):
- **Rows**: True labels (0-9)
- **Columns**: Predicted labels (0-9)
- **Cell Value**: Count of samples with that (true, predicted) pair

#### How to Interpret:
1. **Diagonal Values**: Correct predictions (higher = better)
2. **Off-diagonal Values**: Misclassifications
3. **Row Sum**: Total samples per class (should be ~1000 each for MNIST test set)
4. **Common Confusions**:
   - 4 vs 9: Similar shapes
   - 3 vs 5, 3 vs 8: Curved strokes
   - 7 vs 1: Vertical strokes

#### Example Reading:
If cell (4, 9) = 15, it means 15 images of digit "4" were incorrectly classified as "9".

#### Per-Class Accuracy Calculation:
```
Class_Accuracy[i] = Diagonal[i] / Sum(Row[i]) * 100%
```

---

### 04_misclassified_samples/
Analysis of incorrectly classified images.

#### Files:
- **misclassified_samples.xlsx**: Details of each misclassification
- **misclassified_samples.png**: Visualization of example errors

#### Data Columns (misclassified_samples.xlsx):
| Column | Description |
|--------|-------------|
| Sample_Index | Index in test dataset (0-9999) |
| True_Label | Actual digit (ground truth) |
| Predicted_Label | Model's prediction |
| Confidence | Softmax probability of predicted class |
| True_Label_Prob | Probability assigned to correct class |

#### How to Interpret:
1. **High Confidence Errors**: Model was "sure" but wrong - indicates systematic bias
2. **Low Confidence Errors**: Model was uncertain - ambiguous samples
3. **Pattern Analysis**:
   - Check if certain digit pairs are frequently confused
   - Identify if errors cluster in specific true labels

#### Visualization (PNG):
Shows 25 randomly selected misclassified samples with:
- Image display
- Title format: "T:X P:Y" (True:X Predicted:Y)
- Useful for qualitative error analysis

---

### 05_tsne/
t-SNE (t-distributed Stochastic Neighbor Embedding) visualization of learned feature space.

#### Files:
- **tsne_data.xlsx**: 2D coordinates and labels for each sample
- **tsne_visualization.png**: Scatter plot visualization

#### Data Columns (tsne_data.xlsx):
| Column | Description |
|--------|-------------|
| tsne_x | t-SNE dimension 1 coordinate |
| tsne_y | t-SNE dimension 2 coordinate |
| true_label | Actual digit class (0-9) |
| predicted_label | Model's prediction |
| correct | Boolean: prediction == true_label |

#### How to Interpret:
1. **Cluster Separation**: Distinct, well-separated clusters = good feature learning
2. **Cluster Overlap**: Overlapping regions indicate classes the model confuses
3. **Outliers**: Points far from their cluster centroid may be ambiguous samples
4. **Color Coding**: Each digit class has unique color

#### t-SNE Parameters Used:
- **Perplexity**: 30 (balance between local and global structure)
- **Input**: Final hidden layer activations (128-dimensional)
- **Output**: 2D visualization

#### What Good Results Look Like:
- 10 distinct clusters (one per digit)
- Minimal overlap between clusters
- Compact, tight clusters
- Misclassified points typically at cluster boundaries

---

### 06_per_class_metrics/
Detailed performance metrics for each digit class.

#### Files:
- **per_class_metrics.xlsx**: Precision, Recall, F1 for each class
- **per_class_metrics.png**: Bar chart comparison

#### Data Columns (per_class_metrics.xlsx):
| Column | Description | Formula | Range |
|--------|-------------|---------|-------|
| Class | Digit (0-9) | - | 0-9 |
| Precision | Positive predictive value | TP/(TP+FP) | 0-1 |
| Recall | True positive rate (Sensitivity) | TP/(TP+FN) | 0-1 |
| F1-Score | Harmonic mean of P and R | 2*P*R/(P+R) | 0-1 |
| Support | Number of test samples | - | ~1000 |

#### Metric Definitions:
- **TP (True Positive)**: Correctly predicted as class X
- **FP (False Positive)**: Incorrectly predicted as class X (actually other class)
- **FN (False Negative)**: Actually class X but predicted as other class

#### How to Interpret:
1. **High Precision, Low Recall**: Model is conservative - when it predicts X, it's usually right, but misses many X
2. **Low Precision, High Recall**: Model predicts X too often - catches most X but with false alarms
3. **F1-Score**: Balanced metric - best single indicator of class performance
4. **Support Imbalance**: MNIST is balanced (~1000/class), but check for any discrepancy

#### Typical MNIST Class Difficulty:
- **Easy** (F1 > 0.97): 0, 1, 6
- **Medium** (F1 0.94-0.97): 2, 3, 7
- **Hard** (F1 < 0.94): 4, 5, 8, 9

---

### 07_comparison_with_ideal/
Comparison between Fitted Device and Ideal/Linear baseline devices.

#### Files:
- **fitted_*.xlsx/png**: Fitted Device (PiecewiseStepDevice) results
- **ideal_*.xlsx/png**: Ideal Device (perfect analog) results
- **linear_*.xlsx/png**: Linear Device (uniform step response) results
- **metrics_comparison.png**: Side-by-side accuracy comparison
- **weight_distribution_comparison.png**: Weight distribution comparison

#### Device Descriptions:
| Device | Description | Characteristics |
|--------|-------------|-----------------|
| **Ideal Device** | Perfect analog behavior | No non-linearity, no saturation, ideal weight updates |
| **Linear Device** | Uniform piecewise steps | Linear response, no asymmetry |
| **Fitted Device** | Experimental organic memristor | Non-linear response, asymmetric up/down, saturation effects |

#### How to Interpret Comparison:
1. **Accuracy Gap**:
   - Fitted vs Ideal: Shows impact of device non-idealities
   - Gap < 2%: Device is suitable for this task
   - Gap > 5%: Device limitations significantly affect performance

2. **Weight Distribution Differences**:
   - Ideal: Smooth, Gaussian-like, no saturation
   - Linear: Similar to Ideal but discrete steps
   - Fitted: May show asymmetry, saturation at boundaries

3. **Convergence Speed**:
   - Compare epochs to reach similar accuracy
   - Fitted Device may need more epochs due to non-linear updates

#### Performance Summary:
| Device | Best Val Accuracy | Notes |
|--------|-------------------|-------|
| Ideal Device | 97.16% | Upper bound (theoretical best) |
| Linear Device | 97.28% | Near-ideal with discrete steps |
| Fitted Device | 95.77% | ~1.4% gap from ideal |

---

## Key Takeaways

### 1. Model Performance
- **95.77% accuracy** is competitive for a simple MLP on MNIST
- The ~1.4% gap from Ideal Device shows device non-idealities have moderate impact

### 2. Device Characteristics Impact
- Non-linear weight updates cause some weights to saturate at boundaries
- Asymmetric up/down pulse responses visible in weight distribution shape

### 3. Error Analysis
- Most errors occur between visually similar digits (4/9, 3/5, 3/8)
- t-SNE shows reasonable cluster separation despite device non-idealities

### 4. Recommendations for Further Improvement
1. **Learning Rate Scheduling**: Try warmup + cosine decay
2. **Weight Regularization**: Add L2 penalty to prevent saturation
3. **Data Augmentation**: Random rotation/shift for robustness
4. **Architecture**: Consider convolutional layers for better feature extraction

---

## Citation

If using this data in publications, please reference:
- IBM AIHWKIT: https://github.com/IBM/aihwkit
- PiecewiseStepDevice model from organic memristor experimental data

## Contact

For questions about the data or methodology, please refer to the experiment scripts in the parent directory.
