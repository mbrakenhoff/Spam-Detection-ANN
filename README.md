# Detecting Spam - Artificial Neural Network with Parameter Tuning

## Overview

This project aims to predict whether an email is spam using an artificial neural network (ANN). The dataset consists of 4601 observations and 57 independent variables. The model is built and tuned to achieve optimal performance.

## Key Components

- **Data Preparation:** Data is preprocessed, including dummy variable creation and response variable transformation.
- **Model Building:** An ANN is created using the Keras library in R, with parameter tuning to optimize its performance.
- **Parameter Tuning:** Multiple grid searches and nested cross-validation are used to find the best hyperparameters.
- **Evaluation:** Model performance is evaluated using accuracy and ROC curve metrics.

## Repository Structure

- **`detect_spam.Rmd`**: Main RMarkdown file with detailed code and explanations.
- **`final_workspace.RData`**: Saved workspace after tuning of parameters.
- **`README.md`**: Project overview and instructions.

## Getting Started

### Prerequisites

- R (version 4.0 or higher)
- RStudio
- Libraries: `data.table`, `MLmetrics`, `keras`, `fastDummies`, `tidyverse`, `knitr`, `pROC`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/username/detect_spam.git
   cd detect_spam
   ```

2. Install required libraries:
   ```r
   install.packages(c('data.table', 'MLmetrics', 'keras', 'fastDummies', 'tidyverse', 'knitr', 'pROC'))
   ```

3. Load the saved workspace:
   ```r
   load('final_workspace.RData')
   ```

### Running the Analysis

Open `detect_spam.Rmd` in RStudio and knit the document to run the analysis. The RMarkdown file includes all necessary code to reproduce the results, including model building, parameter tuning, and evaluation.

## Results

- **Best Parameters:**
  - Activation 1: `tanh`
  - Activation 2: `relu`
  - Optimizer: `adam`
  - Batch Size: `30`
  - Patience: `30`

- **Performance:**
  - Mean accuracy: 94%
  - ROC AUC: 0.97

## Conclusion

The ANN model developed in this project demonstrates high accuracy and effective spam detection capabilities. Parameter tuning significantly improved the model's performance, making it a robust solution for spam detection.

## Author

Michael Brakenhoff
