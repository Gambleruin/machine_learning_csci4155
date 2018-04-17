library(gdata)
library(caret)
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

setwd("~/Desktop")


data <-read.xls('tg_refined.xlsx')
df <-as.data.frame(data)

# train <-sample(1:nrow(X_mat), nrow(X_mat)/2)
#fir_col <-data[, 1]
# target<-do.call(cbind, fir_col)
data_split_on_target<-as.factor(data$Emergence.days.after.planting)
table(data_split_on_target)
Inx <-createDataPartition(data_split_on_target, p =0.7, list =FALSE, times =1)

target <-data$Emergence.days.after.planting
# create training data
train_dat <-df[Inx, ]
train_dat$Emergence.days.after.planting <-NULL
train_dat$Flowering.time.days.after.planting<-NULL
tr_target <-target[Inx]

# create test data
test_dat <-df[-Inx, ]
test_dat$Emergence.days.after.planting <-NULL
test_dat$Flowering.time.days.after.planting<-NULL
te_target <-target[-Inx]

############### None of the variables are selected by running this picec of code
stab_lasso<-stabsel(x =train_dat, y =tr_target, fitfun =glmnet.lasso, cutoff =0.75,PFER =1)
mod <-glmboost(tr_target~., data =train_dat)

############### WARNING: Using a fixed penalty (lambda) is usually not permitted and 
##          not sensible. why?
lambda_min <-cv.glmnet(x =as.matrix(train_dat), y =tr_target)$lambda.min
stab_maxCoef <- stabsel(x = train_dat, y = tr_target,
                                            fitfun = glmnet.lasso_maxCoef, 
                                            # specify additional parameters to fitfun
                                            args.fitfun = list(lambda = lambda_min),
                                            cutoff = 0.9, PFER = 1)

############# apply random forest as the base model
x<-as.matrix(train_dat)
y<-as.numeric(tr_target)
rf <-randomForest(y~., data =x)
rf_pred <-predict(rf, test_dat)

# visualize the result
mat <-cbind(te_target, rf_pred)
matplot(mat,type='l',col=c("black","red"),main="In Sample Plot",ylab="prediction")
legend("topright",legend=c("Actual","Predicted"),fill=c("black","red"),bty='n')
grid()

############# improve the model using xgboost
# put into the xgb matrix format
dtrain <-xgb.DMatrix(data =as.matrix(sapply(train_dat, as.numeric)), label =y)
dtest <-xgb.DMatrix(data =as.matrix(sapply(test_dat, as.numeric)),label =te_target)

# these are the datasets the rmse is evaluated at each iteration
watchlist <-list(train=dtrain, test =dtest)

# the boost model
bst <- xgb.train(data = dtrain, 
                max.depth = 8, 
                eta = 0.3, 
                nthread = 2, 
                nround = 1000, 
                watchlist = watchlist, 
                objective = "reg:linear", 
                early_stopping_rounds = 50,
                print_every_n = 500)
# Stopping. Best iteration:
# [10]	train-rmse:1.402227	test-rmse:2.676311

# tuning the hyperparameter for xgb model
bst_slow <- xgb.train(data = dtrain, 
                     max.depth=5, 
                     eta = 0.01, 
                     nthread = 2, 
                     nround = 10000, 
                     watchlist = watchlist, 
                     objective = "reg:linear", 
                     early_stopping_rounds = 50,
                     print_every_n = 500)

#Stopping. Best iteration:
#[308]	train-rmse:1.886909	test-rmse:2.372684

# use validation data to evaluate our model/partition the data as before
Inx_v <-createDataPartition(data_split_on_target, p =0.8, list =FALSE, times =1)
train_dat_v <-df[Inx_v, ]
valid <-df[-Inx_v, ]

train_dat_v$Emergence.days.after.planting <-NULL
train_dat_v$Flowering.time.days.after.planting<-NULL
tr_target_v <-target[Inx_v]

valid$Emergence.days.after.planting <-NULL
valid$Flowering.time.days.after.planting<-NULL
va_target <-target[-Inx_v]

# repeat the same procedure as before but with validate set of data
gb_train <-xgb.DMatrix(data =as.matrix(sapply(train_dat_v, as.numeric)), label =tr_target_v)
gb_valid <- xgb.DMatrix(data = as.matrix(sapply(valid, as.numeric)), label = va_target )

# train xgb, evaluating against the validation
watchlist <-list(train = gb_train, valid = gb_valid)

bst_slow <- xgb.train(data= gb_train, 
                     max.depth = 10, 
                     eta = 0.01, 
                     nthread = 2, 
                     nround = 10000, 
                     watchlist = watchlist, 
                     objective = "reg:linear", 
                     early_stopping_rounds = 50,
                     print_every_n = 500)

#test the model on truly external data
y_hat_valid <-predict(bst_slow, dtest)

test_mse = mean(((y_hat_valid - te_target)^2))
test_rmse = sqrt(test_mse)

# show the result
# [1] 1.114873

# we will utilize grid search to find the best hyperparameter combinations
###
# Grid search first principles (warning,  this will run a long time for data given)
###
max.depths = c(7, 9)
etas = c(0.01, 0.001)

best_params = 0
best_score = 0

count = 1
for( depth in max.depths ){
  for( num in etas){
    
    bst_grid = xgb.train(data = gb_train, 
                         max.depth = depth, 
                         eta=num, 
                         nthread = 2, 
                         nround = 10000, 
                         watchlist = watchlist, 
                         objective = "reg:linear", 
                         early_stopping_rounds = 50, 
                         verbose=0)
    
    if(count == 1){
      best_params = bst_grid$params
      best_score = bst_grid$best_score
      count = count + 1
    }
    else if( bst_grid$best_score < best_score){
      best_params = bst_grid$params
      best_score = bst_grid$best_score
    }
  }
}
#> best_params
#$max_depth
#[1] 9
#$eta
#[1] 0.001
#$nthread
#[1] 2
#$objective
#[1] "reg:linear"
#$silent
#[1] 1
#> best_score
#valid-rmse 
#1.961483 

# this took long time also
bst_tuned = xgb.train( data = gb_train, 
                       max.depth = 9, 
                       eta = 0.001, 
                       nthread = 2, 
                       nround = 10000, 
                       watchlist = watchlist, 
                       objective = "reg:linear", 
                       early_stopping_rounds = 50,
                       print_every_n = 500)

y_hat_xgb_grid = predict(bst_tuned, dtest)
test_mse = mean(((y_hat_xgb_grid - te_target)^2))
test_rmse = sqrt(test_mse)
# [1] 1.227832

# visualize the result
mat <-cbind(te_target, y_hat_xgb_grid)
matplot(mat,type='l',col=c("black","red"),main="In Sample Plot",ylab="prediction")
legend("topright",legend=c("Actual","Predicted"),fill=c("black","red"),bty='n')
grid()




