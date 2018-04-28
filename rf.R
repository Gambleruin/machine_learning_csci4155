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

setwd("~/Desktop")


data <-read.xls('tg_refined.xlsx')
raw <-as.data.frame(data)
df <-normalizeFeatures(raw, target ='Emergence.days.after.planting')

# train <-sample(1:nrow(X_mat), nrow(X_mat)/2)
#fir_col <-data[, 1]
# target<-do.call(cbind, fir_col)
data_split_on_target<-as.factor(data$Emergence.days.after.planting)
table(data_split_on_target)
Inx <-createDataPartition(data_split_on_target, p =0.7, list =FALSE, times =1)

target <-log(data$Emergence.days.after.planting)
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

############# apply random forest as the base model(default package)
x<-as.matrix(train_dat)
y<-as.numeric(tr_target)
rf <-randomForest(y~., data =x)
rf_pred <-predict(rf, test_dat)
test_mse = mean(((rf_pred - te_target)^2))

test_rmse = sqrt(test_mse)

mat <-cbind(rf_pred, te_target)
matplot(mat,type='l',col=c("black","red"),main="In Sample Plot",ylab="prediction")
legend("topright",legend=c("Actual","Predicted"),fill=c("black","red"),bty='n')
grid()

########### apply random forest on mlr package 
train_dat <-df[Inx, ]
train_dat$Flowering.time.days.after.planting<-NULL

test_dat <-df[-Inx, ]
test_dat$Flowering.time.days.after.planting<-NULL
train_dat$Emergence.days.after.planting <-log(train_dat$Emergence.days.after.planting)
test_dat$Emergence.days.after.planting <-log(test_dat$Emergence.days.after.planting)
ml_task <- makeRegrTask(data=train_dat,target="Emergence.days.after.planting")

cv_folds<-makeResampleDesc("CV", iters =3)
model <- makeLearner("regr.randomForest", predict.type = "se")
# model$par.vals <- list( ntree = 100L, importance=TRUE)
#set 5 fold cross validation
rdesc <- makeResampleDesc("CV",iters=5L)
# Define model tuning algorithm ~ Random tune algorithm
random_tune <- makeTuneControlRandom(maxit = 1L) 

# Define parameters of model and search grid ~ !!!! MODEL SPECIFIC !!!!
model_Params = makeParamSet(
  makeIntegerLearnerParam(id = "ntree", default = 500L, lower = 1L, upper = 1000L),
  makeIntegerLearnerParam(id = "nodesize", default = 1L, lower = 1L, upper = 50L)
)

# Define number of CPU cores to use when training models
# parallelStartSocket(8)
# Tune model to find best performing parameter settings using random search algorithm
tuned_model <- tuneParams(learner = model,
                          task = ml_task,
                          resampling = cv_folds,
                          measures = rsq,       # R-Squared performance measure, this can be changed to one or many
                          par.set = model_Params,
                          control = random_tune,
                          show.info = FALSE)

# Apply optimal parameters to model
model <- setHyperPars(learner = model,
                      par.vals = tuned_model$x)

# Verify performance on cross validation folds of tuned model
resample(model,ml_task,cv_folds,measures = list(rsq,mse))

# trained model (random forest)
rf_mlr <-train(learner =model, task =ml_task)

# predicts on test set
preds <-predict(rf_mlr, newdata =test_dat)
test_mse = mean(((preds$data$response - preds$data$truth)^2))
test_rmse = sqrt(test_mse)
# 0.096359

###
plot_df <-data.frame(preds)
mat <-cbind(plot_df$truth, plot_df$response)
matplot(mat,type='l',col=c("black","red"),main="In Sample Plot",ylab="prediction")
legend("topright",legend=c("Actual","Predicted"),fill=c("black","red"),bty='n')
grid()


