# Setting working directory
setwd("~/Desktop/Coursera/Practical Machine Learning/Peer Assessment/")

# Loading packages
library(ggplot2)
library(caret)

# Files url:
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Downloading files
download.file(trainUrl, destfile = "./pml-training.csv", method = "curl")
download.file(testUrl, destfile = "./pml-testing.csv", method = "curl")
rm(trainUrl, testUrl)

# Uploading files into R
training <- read.csv("./pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testing <- read.csv("./pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))


# Data Exploration
dim(training)
dim(testing)

table(training$classe)
barchart(training$classe, xlab = "Frequency", ylab = "Classe", col = "blue")

# Checking for missing values
nrow(training[!complete.cases(training),])
# All rows have NAs, but let's see how many variables have NAs
ncol(training[,!complete.cases(t(training))])

# Eliminating variables with missing values
NA_list <- !complete.cases(t(training))
training <- training[,!NA_list]
testing <- testing[,!NA_list]
rm(NA_list)

# Remove unnecessary columns from testing and training
training <- training[,-(1:7)]
testing <- testing[,-(1:7)]

# Removing extra column "id_problem" from testing
testing <- testing[,-ncol(testing)]

# Transforming the variable classe into factor
training$classe <- factor(training$classe)

# PreProcessing data
set.seed(12121)
inTrain <- createDataPartition(y = training$classe, p = 0.60, list = FALSE)
train <- training[inTrain,]
valid <- training[-inTrain,]

# Modelling with cross validation
control <- trainControl(method = "cv", number = 4)
model <- train(classe ~ . , data = train, 
               method = "rpart", trControl = control, tuneLength = 5)
print(model)

rfmod <- train(classe ~ . , data = train, 
               method = "rf", trControl = control)

# Validation
test_prediction <- predict(model, newdata = valid)
table(test_prediction)
table(valid$classe)
confusionMatrix(valid$classe, test_prediction)


# Testing the model
prediction <- predict(model, newdata = testing)
print(prediction)

# Table of predictions 
table(prediction)

