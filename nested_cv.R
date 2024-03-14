library(caret)
library(MLmetrics)

nested_cv = function(cv_k1 = 10, cv_k2 = 10, seed = 1, 
                     best_perf_method = 'mean', 
                     perf_type = 'low', grid = NULL,  
                     inner_perf_f = NULL, outer_perf_f, score_f,
                     model,
                     dat, response) {
  #' Performs a nested cross-validation to determine performance
  #' with a grid-search of parameters. Totally generalized to work
  #' for either regression or classification on any performance metric.
  #' 
  #' @param cv_k1 - the number of folds in the model performance CV
  #' @param cv_k2 - the number of folds in the grid-search CV
  #' @param seed - a random seed
  #' @param best_perf_method - 'mean' or 'freq'
  #' @param perf_type - 'low' or 'high', is low or high perf best?
  #' @param grid - matrix or data.frame of parameters, one per column
  #' @param model - a function that returns a fit model
  #' @param inner_perf_f - a function that returns a performance number
  #' @param outer_perf_f - a function that returns a performance number
  #' @param score_f - function that returns number for scoring observations
  #' @param dat - data.frame of data
  #' @param response - string indicating column with response
  #' 
  set.seed(seed)
  
  i_folds = createFolds(dat[,response], k = cv_k1)
  
  outer_perf = matrix()
  best_params = matrix()
  scores = list()
  y = list()
  
  for(i in 1:cv_k1) {
    message(paste0('Fold ', i), appendLF = T)
    i_train = unlist(i_folds[-i])
    i_test = unlist(i_folds[i])
    
    j_folds = createFolds(dat[,response][i_train], k = cv_k2)
    
    # It is somewhat inefficient to create another copy
    # of the data here. It just makes the code easier
    # to understand.
    dat_i = dat[i_train,]

    if(!is.null(grid)) {
      inner_perf = list()
      
      for(j in 1:cv_k2) {
        message(paste0('HP Fold ', i,'.',j), appendLF = T)
        j_train = unlist(j_folds[-j])
        j_test = unlist(j_folds[j])
        
        perf_rows = matrix()
        
        # Fit the model to every set of parameters for the grid-search.
        # Save the performance metric for each set.
        for(row in 1:nrow(grid)) {
          message(paste0('HP Fold ', i,'.',j, ':', 'row', row), appendLF = T)
          fit = model(dat_i[j_train,], response, grid[row,])
          perf_rows[row] = inner_perf_f(fit, dat_i[j_test,], response)
        } # end for row in grid
        
        
        # Save the set of performance metrics for this fold of the 
        # grid-search CV.
        inner_perf[[j]] = perf_rows
        
      } # end for j in 1:cv_k2
      
      inner_perf = as.matrix(do.call(rbind, inner_perf))
      
      if(best_perf_method == 'mean') {
        if(perf_type == 'low') {
          best = which.min(colMeans(inner_perf, na.rm=T))
        } else {
          best = which.max(colMeans(inner_perf, na.rm=T))
        }
      } else {
        if(perf_type == 'low') {
          best = mfv(apply(inner_perf, 1, which.min))[1]
        } else {
          best = mfv(apply(inner_perf, 1, which.max))[1]
        }
      }
      
      # Now fit the model on the selected parameters.
      fit = model(dat_i, response, grid[best,])
      
      outer_perf[i] = outer_perf_f(fit, dat[i_test,], response)
      scores[[i]] = score_f(fit, dat[i_test,], response)
      y[[i]] = dat[i_test, response]
      best_params[i] = best
    } else {
      # Now fit the model on the selected parameters.
      fit = model(dat_i, response)
      scores[[i]] = score_f(fit, dat[i_test,], response)
      y[[i]] = dat[i_test, response]
      outer_perf[i] = outer_perf_f(fit, dat[i_test,], response)
      best_params[i] = NA
    }
    
  } # end for i in 1:cv_k1
  
  return(list(Performance = data.frame(Perf = outer_perf, BestParams = best_params),
              Scores = unlist(scores),
              Y = unlist(y)))
  
} # end nested_cv

