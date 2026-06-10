# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** dolor sit amet  
**Team Members:** Aditya Arsh, Saksham Agrawal, Shobhnik Kriplani, Varivashya Poladi  
**Submission Date:** 13th Oct. 2025

---

## 1. Executive Summary
*We designed a multimodal machine learning solution for product price prediction that leverages both text embeddings from catalog content and structured metadata features. By combining semantic embeddings from SentenceTransformers with unit/value features and missing-value indicators, and training an MLP optimized via Optuna, our approach achieved strong performance on the validation set.*

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The challenge required predicting product prices based on catalog descriptions and product images. 

Predicting product prices in an e-commerce catalog is a real-world problem with significant business implications, including optimizing inventory, recommending products, and setting competitive pricing strategies. Each product is represented by diverse information sources: unstructured text (product name, bullet points, and detailed descriptions), structured metadata (units, quantities, and other numerical/categorical attributes), and product images. Leveraging all these modalities effectively is essential to accurately estimate prices.

**Key Observations:**

The **major challenges identified** in this problem are:

1. **Heterogeneous Data Modalities:**  
   Text, structured metadata, and images each require different preprocessing and feature extraction techniques. Combining them meaningfully is non-trivial.

2. **Noisy and Inconsistent Text:**  
   Product descriptions often include typos, irregular formatting, missing information, or ambiguous terms. Units and quantities are expressed inconsistently, necessitating careful cleaning and standardization.

3. **Missing or Incomplete Data:**  
   Some products may lack images, metadata values, or descriptive text. Handling these missing entries is critical to avoid biasing the model and degrading prediction accuracy.

4. **Highly Skewed Price Distribution:**  
   Prices can vary widely across products, spanning multiple orders of magnitude. Direct regression can be unstable, so transformations like log-scaling and derived features (e.g., price per unit) are required for robust learning.

5. **Complex Interactions Across Features:**  
   Price depends not only on individual attributes but also on interactions between text, metadata, and visual features. Capturing these cross-modal dependencies requires an advanced multimodal modeling approach.

6. **Computational Constraints:**  
   Processing high-dimensional embeddings from both text and images at scale demands efficient model design and training strategies to remain practical in real-world scenarios.

This problem exemplifies the challenges of real-world multimodal learning. Accurately predicting product prices requires a model capable of integrating heterogeneous data, handling missing and noisy inputs, and capturing complex relationships between textual, visual, and structured features.

### 2.2 Solution Strategy
Our solution combines semantic text embeddings, structured metadata, and visual image features into a multimodal deep learning pipeline. The base learner is a feedforward neural network (MLP) trained on concatenated representations from three sources:
- Textual features via SentenceTransformer embeddings of catalog descriptions.
- Metadata features such as unit, value, price-per-unit, and missingness indicators.
- Image embeddings from pretrained CLIP/DINOv2 models.

**Approach Type:** Hybrid Multimodal Ensemble  
**Core Innovation:** The key innovation lies in the fusion of heterogeneous modalities (semantic embeddings, structured metadata, and visual features) into a unified MLP pipeline. This design enables the model to capture both semantic meaning and numeric patterns. To maximize accuracy, we applied Optuna-based hyperparameter tuning (hidden layer sizes, dropout, learning rate, etc.), which significantly improved validation SMAPE.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Pipeline:

Catalog Content → Text Embeddings (SentenceTransformer)
+
Metadata (unit, value, missing indicators)
↓
Concatenation Layer
↓
Multilayer Perceptron (ReLU + Dropout)
↓
Final Dense Layer (predict log-price)


### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps:
  - Split `catalog_content` into lines.  
  - Extract product name from the first line.  
  - Extract bullet points from intermediate lines.  
  - Extract product description from the line labeled `"Product Description"`.  
  - Compute missing flags for empty bullet points or missing description.  
  - Compute log-transformed lengths of bullet points and product description.  
  - Convert text fields (name, bullet points, description) to semantic embeddings using `SentenceTransformer("google/embeddinggemma-300m")`.  
  - Standardize embeddings separately using `StandardScaler`.  

- Model type: GEMMA embeddings (`google/embeddinggemma-300m`)
- Key parameters:
  - Embedding dimensions: 300 per text field  
  - Standardization: `StandardScaler` fit separately for name, bullet points, and product description embeddings  
  - Log-transform applied to target prices (`np.log1p`)  
  - Missing flags and log-lengths appended as additional features  


**Image Processing Pipeline:**
- Preprocessing steps:
  - Extract the image filename from the `image_link` column.
  - Download all images from URLs if not already available locally (`download_images` function).
  - Open each image using PIL and convert to RGB.
  - Preprocess images using `CLIPProcessor` from `openai/clip-vit-base-patch32` (resizing, normalization, tensor conversion).
  - Move inputs to GPU if available.
  - Pass the preprocessed image through `CLIPModel` to obtain 512-dimensional image embeddings.
  - Normalize embeddings to unit length.
  - Save embeddings as pickle files (`clip_image_embeddings.pkl`) for later use.
  - Scale embeddings using `StandardScaler` before feeding into the model.

- Model type: CLIP (ViT-B/32) for feature extraction 
- Key parameters:
  - Image model: `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`
  - Processor: `CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")`
  - Embedding dimension: 512
  - Normalization: L2 normalization per image
  - Scaler: `StandardScaler` fit on all embeddings
  - Missing images handled by inserting zero-vector embeddings
 
**Metadata Processing Pipeline**

- **Preprocessing Steps:**
  - Extract structured features: `unit`, `value`, `price_per_unit`, and other numeric/categorical fields.
  - Compute missing indicators for any absent metadata values.
  - Log-transform skewed numeric features (e.g., `price_per_unit`) for stabilization.
  - Standardize numeric features using `StandardScaler`.
  - Encode categorical features using one-hot encoding or embedding layers if high cardinality.

- **Model Type:** Dense feedforward network
- **Key Parameters:**
  - Input dimension: Number of metadata features + missing flags
  - Hidden layers: Tuned via Optuna
  - Activation: ReLU
  - Dropout: Tuned via Optuna

**Feature Fusion**

- Concatenate processed text embeddings, metadata features, and image embeddings into a single unified feature vector.
- Optionally apply batch normalization to stabilize learning.
- Dropout applied to prevent overfitting.
- Input to the regression head (MLP).

**Regression Head (MLP)**

- **Architecture:**
  - Fully connected layers with ReLU activations.
  - Dropout layers to prevent overfitting.
  - Final dense layer outputs predicted log-price.
- **Key Parameters:**
  - Hidden layer sizes: Tuned via Optuna
  - Dropout rates: Tuned via Optuna
  - Learning rate and optimizer: Tuned via Optuna (Adam/AdamW recommended)
  - Loss function: Mean Squared Error on log-transformed prices

**Hyperparameter Optimization (Optuna)**

- Hyperparameters tuned include:
  - Hidden layer sizes for text, metadata, and image pathways
  - Dropout rates
  - Learning rate and weight decay
  - Batch size
- Optuna performs automated search to maximize validation performance (e.g., minimize SMAPE).

---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** [your best validation SMAPE]
- **Other Metrics:** [MAE, RMSE, R² if calculated]


## 5. Conclusion
*Our hybrid multimodal model successfully integrated text embeddings with structured metadata for robust price prediction. The use of Optuna hyperparameter search improved convergence and validation SMAPE. Lessons learned include the importance of engineered features (units and values) and handling missing fields.*

---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*


### B. Additional Results
*Include any additional charts, graphs, or detailed results*

---

**Note:** This is a suggested template structure. Teams can modify and adapt the sections according to their specific solution approach while maintaining clarity and technical depth. Focus on highlighting the most important aspects of your solution.
