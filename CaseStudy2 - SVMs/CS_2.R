################################################################################################
#Brunda Chouthoy
#CSC 529: Case Study 2
#Zip Info data: USPS hand written digits dataset
################################################################################################

working_dir <- "~/Desktop/MS-PA/QuarterV-Winter/CSC 529 /CaseStudy2/zipinfo"
setwd(working_dir)
colors_list <- c("white", "black")

library(RColorBrewer)
library(ElemStatLearn)
library(e1071)
library(ggplot2)
library(ROCR)

### Set Color colorRampPalette(COLORS) ( 4 ) 
CUSTOM_COLORS <- colorRampPalette(colors = colors_list)
CUSTOM_COLORS_PLOT <- colorRampPalette(brewer.pal(10, "Set3"))


################################################################################################
## Exploratory Analysis
################################################################################################
### Load data
train_data <- as.data.frame(zip.train)
test_data <- as.data.frame(zip.test)

#Analysis of TRAIN data set
head(train_data)
#check Dimension
dim(train_data)
#Check datatype of all features and descriptions of each variable
str(train_data)
sapply(train_data[1, ],class)
summary(train_data)
#Check if there are missing values in the Training data
table(is.na(train_data)) 
sapply(train_data, function(train_data) sum(is.na(train_data)))

#Analysis of TEST dataset
head(test_data)
#check Dimension
dim(test_data)
#Check datatype of all features and descriptions of each variable
str(test_data)
sapply(test_data[1, ],class)
summary(test_data)
#Check if there are missing values in the TEST dataset
table(is.na(test_data)) 
sapply(test_data, function(test_data) sum(is.na(test_data)))

#For Train Data --> Transform Label as Factor(categorical) and change column names to D.1 to D.256
train_data[, 1] <- as.factor(train_data[, 1])  # As Category
colnames(train_data) <- c("Y", paste("D.", 1:256, sep = ""))
class(train_data[, 1])
levels(train_data[, 1])
##Check datatypes of each variable after label transformation
sapply(train_data[1, ], class)

#For Testing Dataset --> Transform Label as Factor(categorical) and change column names to D.1 to D.256
test_data[, 1] <- as.factor(test_data[, 1])  # As Category
colnames(test_data) <- c("Y", paste("D.", 1:256, sep = ""))
class(test_data[, 1])
levels(test_data[, 1])
##Check datatypes of each variable after label transformation
sapply(test_data[1, ], class)

#Display digits of training data set 
#Source: http://www.r-bloggers.com/the-essence-of-a-handwritten-digit/
par(mfrow = c(4, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
images_digits_0_9 <- array(dim = c(10, 16 * 16))
for (digit in 0:9) {
  print(digit)
  images_digits_0_9[digit + 1, ] <- apply(train_data[train_data[, 1] == 
                                                          digit, -1], 2, sum)
  images_digits_0_9[digit + 1, ] <- images_digits_0_9[digit + 1, ]/max(images_digits_0_9[digit + 
                                                                                           1, ]) * 255
  z <- array(images_digits_0_9[digit + 1, ], dim = c(16, 16))
  z <- z[, 16:1]  ##right side up
  image(1:16, 1:16, z, main = digit, col = CUSTOM_COLORS(256))
}

#Generate a PDF file for all the digits of the training dataset
##Source: http://www.r-bloggers.com/the-essence-of-a-handwritten-digit/
pdf('train_letters.pdf')
help("pdf")
par(mfrow=c(4,4),pty='s',mar=c(3,3,3,3),xaxt='n',yaxt='n')
for(i in 1:nrow(train_data))
{
  z<-array(as.vector(as.matrix(train_data[i, -1])),dim=c(16,16))
  z<-z[,16:1] ##right side up
  image(1:16,1:16,z,main=train_data[i,1],col=CUSTOM_COLORS(256))
  print(i)
}

#Displaying total number of digits under each category in the TRAIN dataset
resultTable <- table(train_data$Y)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1)  # increase y-axis margin
plot <- plot(train_data$Y, col = CUSTOM_COLORS_PLOT(10), main = "Total Number of Digits (Training Set)", ylim = c(0, 1500), ylab = "Number of digits",xlab = "Digit categories 0-9")
text(x = plot, y = resultTable + 50, labels = resultTable, cex = 0.75)

#Displaying total number of digits under each category in the TEST dataset
resultTable <- table(test_data$Y)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2))  # increase y-axis margin.
plot <- plot(test_data$Y, col = CUSTOM_COLORS_PLOT(10), main = "Total Number of Digits (Testing Set)", 
             ylim = c(0, 400), ylab = "No of digits",xlab = "Digit categories 0-9")
text(x = plot, y = resultTable + 20, labels = resultTable, cex = 0.75)

################################################################################################
#DATA MODELING
##Applying Support Vector machine algorithm for classification
################################################################################################
## SVM with linear kernel
pc <- proc.time()
model.svm.linear <- svm(Y ~ ., kernel = "linear", method = "class", data = train_data)
proc.time() - pc
summary(model.svm.linear)

##Confusion Matrix for SVM
predict.svm.linear <- predict(model.svm.linear, newdata = test_data, type = "class")
mat.radial<-table(`Actual Class` = test_data$Y, `Predicted Class` = predict.svm.linear)

error.rate.svmlinear <- sum(test_data$Y != predict.svm.linear)/nrow(test_data)
print(paste0("Accuary (Precision): ", 1 - error.rate.svmlinear))

##SVM with kernel - RBF
#Tuning parameters gamma and cost using the tune.svm function
tuned <- tune.svm(Y ~ ., data = train_data, gamma = 10^(-6:-1), cost = 10^(-1:1))
summary(tuned)

#model
pc <- proc.time()
model.svm <- svm(Y ~ ., kernel = "radial", method = "class", data = train_data, gamma = 0.001, cost = 10)
proc.time() - pc
summary(model.svm)

##Confusion Matrix for SVM
predict.svm <- predict(model.svm, newdata = test_data, type = "class")
mat.radial<-table(`Actual Class` = test_data$Y, `Predicted Class` = predict.svm)

#Precision for each class - RBF kernel
precision<-(precision <- diag(mat.radial) / rowSums(mat.radial))
print(paste0("Precision: ", precision))

#Recall for each class
recall <- (diag(mat.radial) / colSums(mat.radial))
print(paste0("Recall: ", recall))

error.rate.svm <- sum(test_data$Y != predict.svm)/nrow(test_data)
print(paste0("Accuary (Precision): ", 1 - error.rate.svm))

#Predict digit using the SVM model
# All Prediction for Row 1
row <- 1
predict.digit <- as.vector(predict(model.svm, newdata = test_data[row,  ], type = "class"))
print(paste0("Current Digit: ", as.character(test_data$Y[row])))

print(paste0("Predicted Digit: ", predict.digit))

z <- array(as.vector(as.matrix(test_data[row, -1])), dim = c(16, 16))
z <- z[, 16:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:16, 1:16, z, main = test_data[row, 1], col = CUSTOM_COLORS(256))

##Errors with SVM
errors <- as.vector(which(test_data$Y != predict.svm))
print(paste0("Error Numbers: ", length(errors)))

predicted <- as.vector(predict.svm)
par(mfrow = c(4, 4), pty = "s", mar = c(3, 3, 3, 3), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z_digit <- array(as.vector(as.matrix(test_data[errors[i], -1])), dim = c(16, 16))
  z_digit <- z_digit[, 16:1]  ##right side up
  image(1:16, 1:16, z_digit, main = paste0("C.D.:", as.character(test_data$Y[i])," - Pr.D.:", predicted[errors[i]]), col = CUSTOM_COLORS(256))
}

################################################################################################
#SVM with kernel -- Polynomial
################################################################################################
#Tuning parameters coef0 and degree using the tune.svm function
mytunedsvm <- tune.svm(Y ~ ., kernel = "polynomial", data = train_data, coef0 = (-1:4), degree = (1:4))
summary(mytunedsvm)
plot (mytunedsvm,xlab="degree", ylab="coef0")

#Model
pc <- proc.time()
model.svm.polynomial <- svm(Y ~ ., method = "class", data = train_data, kernel = 'polynomial', degree = 3,coef0 = 1)
proc.time() - pc
summary(model.svm.polynomial)

##Confusion Matrix for SVM - polynomial kernel
predict.svm.poly <- predict(model.svm.polynomial, newdata = test_data, type = "class")
matrix<-table(`Actual Class` = test_data$Y, `Predicted Class` = predict.svm.poly)
matrix

#Precision for each class
precision<-(precision <- diag(matrix) / rowSums(matrix))
print(paste0("Precision: ", precision))

#Recall for each class
recall <- (diag(matrix) / colSums(matrix))
print(paste0("Recall: ", recall))

#Error rate and Accuracy
error.rate.svm <- sum(test_data$Y != predict.svm.poly)/nrow(test_data)
print(paste0("Accuary (Precision): ", 1 - error.rate.svm))

#Predict digit using the SVM model
# All Prediction for Row 1
row <- 1
predict.digit <- as.vector(predict(model.svm.polynomial, newdata = test_data[row,  ], type = "class"))
print(paste0("Current Digit: ", as.character(test_data$Y[row])))

print(paste0("Predicted Digit: ", predict.digit))

z <- array(as.vector(as.matrix(test_data[row, -1])), dim = c(16, 16))
z <- z[, 16:1]  ##right side up
par(mfrow = c(1, 3), pty = "s", mar = c(1, 1, 1, 1), xaxt = "n", yaxt = "n")
image(1:16, 1:16, z, main = test_data[row, 1], col = CUSTOM_COLORS(256))

##Errors with SVM
errors <- as.vector(which(test_data$Y != predict.svm.poly))
print(paste0("Error Numbers: ", length(errors)))

#Displaying all misclassified digits
predicted <- as.vector(predict.svm.poly)
par(mfrow = c(4, 4), pty = "s", mar = c(3, 3, 3, 3), xaxt = "n", yaxt = "n")
for (i in 1:length(errors)) {
  z_digit <- array(as.vector(as.matrix(test_data[errors[i], -1])), dim = c(16, 16))
  z_digit <- z_digit[, 16:1]  ##right side up
  image(1:16, 1:16, z_digit, main = paste0("C.D.:", as.character(test_data$Y[i])," - Pr.D.:", predicted[errors[i]]), col = CUSTOM_COLORS(256))
}
################################################################################################

