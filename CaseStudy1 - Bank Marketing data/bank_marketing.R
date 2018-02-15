################################################################
#Brunda Chouthoy
#CSC 529: CASE STUDY 1
#Bank Marketing Dataset
################################################################

#Load Library or packages
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(readr) #CSV file I/O, e.g. the read_csv function
library(randomForest)
library(caTools) #For stratified split
library(e1071)


################################################################
#Exploratory Analysis of data
################################################################
bank_full <- read.csv(file="bank-additional-full.csv",header=TRUE,sep=";")
head(bank_full)
str(bank_full) ## Describes each variable

#Check the class of each attribute
summary(bank_full)

#as.numeric function to convert and change the integer values to numeric
bank_full$age <- as.numeric(bank_full$age)
bank_full$duration <- as.numeric(bank_full$duration)
bank_full$campaign <- as.numeric(bank_full$campaign)
bank_full$pdays <- as.numeric(bank_full$pdays)
bank_full$previous <- as.numeric(bank_full$previous)

#Check if there are missing values
table(is.na(bank_full)) 
sapply(bank_full, function(bank_full) sum(is.na(bank_full)))

## Using Box plots (Only for continuous variables)- To Check Ouliers
boxplot(bank_full$age~bank_full$y, main="Age",ylab="age of customers",xlab="Subscribed")
boxplot(bank_full$duration~bank_full$y, main="Duration",ylab="Last duration of contact",xlab="Subscribed")
boxplot(bank_full$campaign~bank_full$y, main="Campaign:NUM CONTACTS",ylab="Number of contacts",xlab="Subscribed")
boxplot(bank_full$pdays~bank_full$y, main="pdays",ylab="No of days after Previous contact",xlab="Subscribed")
boxplot(bank_full$previous~bank_full$y, main="No of contacts",ylab="Previous days of contact",xlab="Subscribed")
boxplot(bank_full$emp.var.rate~bank_full$y, main="emp.var.rate",ylab="employment variation rate - quarterly indicator",xlab="Subscribed")
boxplot(bank_full$nr.employed~bank_full$y, main="nr.employed",ylab="number of employees - quarterly indicator",xlab="Subscribed")

#Histogram for age
x<-bank_full$age
h<-hist(x, breaks=10, col="red", xlab="Age of customers", main="Histogram for Age") 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

#Histogram for duration
x<-bank_full$duration
h<-hist(x, breaks=100, col="red", xlab="Duration of Contact", main="Histogram for Duration") 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

x<-bank_full$campaign
h<-hist(x, breaks=30, col="red", xlab="Campaign:no of Contacts", main="Histogram for Campaign") 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

x<-bank_full$previous
h<-hist(x, breaks=10, col="red", xlab="number of contacts performed before this campaign", main="Histogram for previous") 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

## Barplots for Categorical Variables
barplot(table(bank_full$job),col="light blue",main="JOB")
barplot(table(bank_full$marital),col="light blue",main="Marital")
barplot(table(bank_full$education),col="light blue",main="Education")
barplot(table(bank_full$default),col="light blue",main="Credit Default")
barplot(table(bank_full$housing),col="light blue",main="Housing Loan")
barplot(table(bank_full$loan),col="light blue",main="Personal Loan")
barplot(table(bank_full$contact),col="light blue",main="Last communication type")
barplot(table(bank_full$month),col="light blue",main="Last Month")
barplot(table(bank_full$day_of_week),col="light blue",main="Day of the week contacted")
barplot(table(bank_full$poutcome),col="light blue",main="poutcome: Outcome of the previous marketing campaign")

## Since Credit Default is highly skewed towards NO, this shall be removed for further analysis
bank_full[5]<-NULL
str(bank_full)


table(bank_full$y)
table(bank_full$y)/nrow(bank_full)

## Correlation Matrix among input (or independent) continuous variables
bank_full.num<-data.frame(bank_full$age,bank_full$campaign,bank_full$previous,bank_full$pdays,
                          bank_full$emp.var.rate,bank_full$cons.conf.idx,bank_full$cons.price.idx)
str(bank_full.num)
cor(bank_full.num)

#################################################
#Data Modeling - Decision Tree Model using RPart
#################################################
size <- nrow(bank_full) * 0.8
validation_index <- sample(1:nrow(bank_full), size = size)
test_tree <- bank_full[-validation_index,]
train_tree <- bank_full[validation_index,]

#Decision tree model using RPart package
bank_rpart <- rpart(y ~ ., data = train_tree)
fancyRpartPlot(bank_rpart)
#plot(bank_rpart)
#text(bank_rpart)
summary(bank_rpart)

pred_rpart<- predict(bank_rpart, train_tree, type = "class")
t <- table(pred_rpart,train_tree$y)
confusionMatrix(t)

pred_rpart<- predict(bank_rpart, test_tree, type = "class")
t <- table(pred_rpart,test_tree$y)
confusionMatrix(t)



#################################################
#Data Modeling - Using Random Forest
#################################################
set.seed(1234)
ind <- sample(2, nrow(bank_full),replace=TRUE, prob=c(0.8, 0.2))
bank_train <- bank_full[ind==1,]
bank_test <- bank_full[ind==2,]

table(bank_train$y)
table(bank_train$y)/nrow(bank_train)

table(bank_test$y)
table(bank_test$y)/nrow(bank_test)

#Make a Formula
varNames <- names(bank_train)
# Exclude ID or Response variable
varNames <- varNames[!varNames %in% c("y")]

# add + sign between exploratory variables
varNames1 <- paste(varNames, collapse = "+")

# Add response variable and convert to a formula object
rf.form <- as.formula(paste("y", varNames1, sep = " ~ "))

#build 500 decision trees using Random Forest.
rf <- randomForest(rf.form,data=bank_train,ntree=500,importance=T)
plot(rf)

# Variable Importance Plot
#Top 10 variables are selected and plotted based on Model Accuracy and Gini value. 
#We can also get a table with decreasing order of importance based on a measure (1 for model accuracy and 2 node impurity)
varImpPlot(rf,sort = T, main="Variable Importance", n.var=10)

# Variable Importance Table
var.imp <- data.frame(importance(rf,
                                 type=2))
# make row names as columns
var.imp$Variables <- row.names(var.imp)
var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),]

#Generic predict function can be used for predicting response variable using Random Forest object.
# Predicting response variable
bank_train$predicted.response <- predict(rf ,bank_train)

# Create Confusion Matrix
confusionMatrix(data=bank_train$predicted.response,
                reference=bank_train$y,
                positive='no')

# Predicting response variable
bank_test$predicted.response <- predict(rf ,bank_test)

# Create Confusion Matrix
confusionMatrix(data=bank_test$predicted.response,
                reference=bank_test$y,
                positive='no')

################################################################
#End of code
################################################################



