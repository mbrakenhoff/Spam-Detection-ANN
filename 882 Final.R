# Load required libraries
library(data.table)
library(MLmetrics)
library(keras)
library(fastDummies)
library(tidyverse)

# Load custom function for nested cross-validation
source('nested_cv.R')

# Read the spam dataset
spam = read.table('spam.tsv', header = TRUE, sep = '\t')

# Define neural network model function
nn_mod = function(dat, response, params) {
  # Create dummy variables
  dat = dummy_cols(dat, remove_selected_columns = TRUE, remove_first_dummy = TRUE)
  response = paste0(response, '_spam')
  
  # Prepare predictors and response
  X = as.matrix(dat[, -which(colnames(dat) == response)])
  y = to_categorical(dat[, response])
  
  # Define neural network model architecture
  mod = keras_model_sequential()
  mod %>% 
    layer_dense(units = 200, input_shape = ncol(X)) %>% 
    layer_activation(activation = params$ACT1) %>% 
    layer_dense(units = 150, kernel_regularizer = regularizer_l1(0.01)) %>% 
    layer_activation(activation = params$ACT2) %>% 
    layer_dense(units = 2, kernel_regularizer = regularizer_l1(0.01)) %>%
    layer_activation(activation = 'sigmoid')
  
  # Compile the model
  mod %>% 
    compile(loss = 'binary_crossentropy',
            optimizer = params$OPT,
            metrics = c('accuracy'))
  
  # Define class weights
  class_weight = list('0' = 6., '1' = 4.)
  
  # Fit the model
  mod %>% 
    fit(X, y, epochs = 200, batch_size = params$BATCH, validation_split = 0.15, class_weight = class_weight,
        callbacks = list(callback_early_stopping(monitor = 'val_loss', min_delta = 0.001, patience = params$PATIENCE, mode = 'min')))
  
  return(mod)
}

# Define function to compute outer performance
nn_outer_perf = function(mod, dat, response) {
  dat = dummy_cols(dat, remove_selected_columns = TRUE, remove_first_dummy = TRUE)
  response = paste0(response, '_spam')
  
  X = as.matrix(dat[, -which(colnames(dat) == response)])
  y = dat[, response]
  
  preds = predict(mod, X)[, 2]
  preds = ifelse(preds > 0.5, 1, 0)
  
  return(Accuracy(preds, y))
}

# Define function to compute inner performance
nn_inner_perf = function(mod, dat, response) {
  dat = dummy_cols(dat, remove_selected_columns = TRUE, remove_first_dummy = TRUE)
  response = paste0(response, '_spam')
  
  X = as.matrix(dat[, -which(colnames(dat) == response)])
  y = dat[, response]
  
  preds = predict(mod, X)[, 2]
  preds = ifelse(preds > 0.5, 1, 0)
  
  return(Accuracy(preds, y))
}

# Define function to compute model scores
nn_score = function(mod, dat, response) {
  dat = dummy_cols(dat, remove_selected_columns = TRUE, remove_first_dummy = TRUE)
  response = paste0(response, '_spam')
  
  X = as.matrix(dat[, -which(colnames(dat) == response)])
  y = dat[, response]
  
  probs = predict(mod, X)[, 2]
  
  return(probs)
}

# Define grid of hyperparameters
nn_grid = data.frame(list(ACT1 = c('relu', 'tanh'), ACT2 = c('relu', 'tanh'), OPT = c('adam', 'sgd'), BATCH = 30, 
                          PATIENCE = c(10, 30)) %>% cross_df())

# Perform nested cross-validation to select optimal hyperparameters
nn_sel = nested_cv(cv_k1 = 4, cv_k2 = 5, seed = 1, model = nn_mod, outer_perf_f = nn_outer_perf, inner_perf_f = nn_inner_perf,
                   perf_type = 'high', grid = nn_grid, score_f = nn_score, dat = spam, response = 'type')

# Print performance of selected hyperparameters
nn_sel$Performance
mean(nn_sel$Performance$Perf)

# Select top performing hyperparameters
top4 = nn_grid[c(2, 9, 10, 12), ]

# Define new grid of hyperparameters
nn_grid2 = data.frame(list(ACT1 = 'tanh', ACT2 = 'relu', OPT = 'adam', BATCH = 30, PATIENCE = c(10, 20, 30, 40, 50))
                      %>% cross_df())

# Perform nested cross-validation with the new hyperparameters
nn_sel2 = nested_cv(cv_k1 = 4, cv_k2 = 5, seed = 1, model = nn_mod, outer_perf_f = nn_outer_perf, inner_perf_f = nn_inner_perf,
                    perf_type = 'high', grid = nn_grid2, score_f = nn_score, dat = spam, response = 'type')

# Print performance of selected hyperparameters
nn_sel2$Performance

# Select final hyperparameters
nn_params = nn_grid2[3, ]

# Perform nested cross-validation with the final hyperparameters
nn_final = nested_cv(cv_k1 = 5, cv_k2 = 2, seed = 1, model = nn_mod, outer_perf_f = nn_outer_perf, inner_perf_f = nn_inner_perf,
                     perf_type = 'high', grid = nn_params, score_f = nn_score, dat = spam, response = 'type')

# Print performance of final hyperparameters
nn_final$Performance
mean(nn_final$Performance$Perf)


