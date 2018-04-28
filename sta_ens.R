# stacked ensemble
# random forest with hyperparameter tuning 
library(gdata)
library(caret)
library(caTools)
library(glmnet)
library(tidyverse)
library(broom)
library(ggplot2)
library(stabs)
library(lars)
library(dplyr)
library(mboost)

require(randomForest)
require(xgboost)
require(MASS)
library(mlr)
library(h2o)

setwd("~/Desktop")


data <-read.xls('tg_refined.xlsx')
raw <-as.data.frame(data)
df <-normalizeFeatures(raw, target ='Emergence.days.after.planting')
df$Flowering.time.days.after.planting<-NULL

# train <-sample(1:nrow(X_mat), nrow(X_mat)/2)
#fir_col <-data[, 1]
# target<-do.call(cbind, fir_col)
data_split_on_target<-as.factor(data$Emergence.days.after.planting)
table(data_split_on_target)
Inx <-createDataPartition(data_split_on_target, p =0.7, list =FALSE, times =1)

# target <-log(data$Emergence.days.after.planting)
# create training data
train_dat <-df[Inx, ]
train_dat$Flowering.time.days.after.planting<-NULL
train_dat$Emergence.days.after.planting <-log(train_dat$Emergence.days.after.planting)
# tr_target <-target[Inx]

# create test data
test_dat <-df[-Inx, ]
test_dat$Flowering.time.days.after.planting<-NULL
test_dat$Emergence.days.after.planting <-log(test_dat$Emergence.days.after.planting)
# te_target <-target[-Inx]

# create super learner with h2o package
h2o.init(nthreads = -1, max_mem_size = "2G")
# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5
y<- 'Emergence.days.after.planting'
x<- setdiff(names(train_dat), y)
train <-as.h2o(train_dat)
test <-as.h2o(test_dat)

# train & cross-validate a gbm
my_gbm <-h2o.gbm(x =x, y =y, training_frame = train, 
                 distribution = 'gaussian', 
                 ntrees =50, 
                 max_depth =, 
                 min_rows =2, 
                 learn_rate =0.2, 
                 nfolds =nfolds, 
                 fold_assignment = 'Modulo', 
                 keep_cross_validation_predictions = TRUE, 
                 seed =1)
perf <- h2o.performance(my_gbm, newdata = test)
h2o.rmse(perf)
# [1] 0.1211558

my_gbm <-h2o.gbm(x =x, y=y, training_frame =train)

# train & cross-validate a rf
my_rf <-h2o.randomForest(x =x,
                         y =y,
                         training_frame = train,
                         distribution = 'gaussian',
                         ntrees =50,
                         nfolds = nfolds,
                         stopping_metric = 'RMSE',
                         fold_assignment = 'Modulo',
                         keep_cross_validation_predictions = TRUE,
                         seed =1)
# [1] 0.1088897
# [1] 0.09487376

my_rf <-h2o.randomForest(x =x, y=y, training_frame = train)

my_xgb1 <-h2o.xgboost(x =x,
                      y =y,
                      training_frame = train,
                      distribution = 'gaussian',
                      ntrees = 100,
                      max_depth = 3,
                      min_rows = 2,
                      learn_rate = 0.2,
                      nfolds = nfolds,
                      fold_assignment = "Modulo",
                      keep_cross_validation_predictions = TRUE,
                      seed = 1
                      )
# [1] 0.1137605



my_xgb2 <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = train,
                       distribution = "gaussian",
                       ntrees = 50,
                       max_depth = 8,
                       min_rows = 1,
                       learn_rate = 0.1,
                       sample_rate = 0.7,
                       col_sample_rate = 0.9,
                       nfolds = nfolds,
                       fold_assignment = "Modulo",
                       keep_cross_validation_predictions = TRUE,
                       seed = 1)
# [1] 0.1012243
# [1] 0.09481084

############## ensemble stage
ensemble <-h2o.stackedEnsemble(x =x,
                               y =y, 
                               training_frame = train, 
                               model_id = 'obj_ensemble', 
                               base_models = list(my_rf, my_xgb2))

perf <- h2o.performance(ensemble, newdata = test)

# Train a stacked ensemble using the H2O and XGBoost models from above
base_models <- list(my_rf@model_id,   
                     my_xgb2@model_id, my_xgb1@model_id)

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = base_models)

perf <- h2o.performance(ensemble, newdata = test)
h2o.rmse(perf)
# [1] 0.09204387

###########grid search
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = 'gbm',
                    
                     x = x,
                     y = y,
                     training_frame = train,
                     ntrees = 10,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)


rf_grid <- h2o.grid("randomForest", 
                    x = x, 
                    y = y, training_frame = train,
                    distribution ="gaussian",
                    stopping_metric ='RMSE',
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    hyper_params = list(ntrees = c(100,300,500), 
                                        mtries = c(2,3,4), max_depth = c(3,4,5)), 
                    search_criteria = search_criteria,
                    seed = 1122)


ensemble_ <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = "ensemble_gbm_grid",
                                base_models = rf_grid@model_ids)

h2o.rmse(perf)



