############################ SVM Digit Recogniser ###################################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################################

# 1. Business Understanding: 

#We are required to develop a model using Support Vector Machine which should correctly 
#classify the handwritten digits based on the pixel values given as features.

#####################################################################################################
# 2. Data Understanding:

# Number of Instances: 60,000
# Number of Attributes: 785 

######################################################################################################
#3. Data Preparation: 

#installing packages
#install.packages("caret")
#install.packages("kernlab")
#install.packages("dplyr")
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("gridExtra")

#Loading Neccessary libraries
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)

mnist_train<-read.csv("mnist_train.csv",header = FALSE)
mnist_test<-read.csv("mnist_test.csv",header = FALSE)

#Understanding Dimensions

dim(mnist_train)
dim(mnist_test)

#Structure of the dataset

str(mnist_train)
str(mnist_test)

#printing first few rows of training data

head(mnist_train)

#Exploring the data

summary(mnist_train)

#checking missing value

sapply(mnist_train, function(x) sum(is.na(x)))

#Making our target class to factor

mnist_train$V1<-factor(mnist_train$V1)
mnist_test$V1<-factor(mnist_test$V1)

#taking 15% of training data
set.seed(1)
train.indices = sample(1:nrow(mnist_train), 0.15*nrow(mnist_train))
train = mnist_train[train.indices, ]

#####################################################################################################

#Constructing Model

#Linear Kernel
#Accuracy : 0.9166 

Model_linear <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, mnist_test)
confusionMatrix(Eval_linear,mnist_test$V1)


#RBF Kernel
#Accuracy : 0.9587   

Model_RBF <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, mnist_test)
confusionMatrix(Eval_RBF,mnist_test$V1)

# We can conclude that RBF model is performing better than Linear  


######################## Hyperparameter tuning and Cross Validation ##################################

#traincontrol function Controls the computational nuances of the train function.
# i.e. method = CV means  Cross Validation.
#      Number = 3 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=3)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
# Cross Validation folds = 3
# Range of sigma = 1.63e-7, 2.63e-7  and 3.63e-7
# Range of C = 1 2 3

set.seed(2)

grid <- expand.grid(.sigma = c(1.63e-7,2.63e-7,3.63e-7),.C=c(1,2,3))

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)

# We see that accuracy is highest for a model having sigma = 3.63e-07 and C = 3

final_model <- ksvm(V1~., data=train, scale=FALSE, kernel="rbfdot", C=3, kpar=list(sigma=3.63e-7))

final_model

Eval_final_model <- predict(final_model,mnist_test)
confusionMatrix(Eval_final_model,mnist_test$V1)
#Accuracy : 0.9691