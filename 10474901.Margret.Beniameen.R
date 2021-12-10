#Student Number: 10474901 
#Student Name: Margret Beniameen
#Assignment 2

install.packages(c("tidyverse","dendextend","cluster","factoextra","gplots",
                   "mvabund","vegan","RColorBrewer", 
                   "GGally","corrplot","car","glmnet","naniar","rpart","mlbench","pROC" ,"ipred"))
install.packages("caret", dependencies = TRUE)
library(tidyverse)  
library(cluster)  #For AGNES and DIANA
library(dendextend)  #For dendrogram and for calculating the cophenetic correlation
library(factoextra)  #For visualisation of clusters
library(gplots)  #For the heatmaps
library(mvabund)  #For spider dataset
library(vegan)  #For vegdist(.) function
library(RColorBrewer)  #For more colour palettes
library(GGally)  #For scatter plot matrix
library(caret)  #Classification and Regression Training package
library(corrplot)  #For visualisng correlation matrix
library(car)  #For the VIF function
library(naniar)
library(glmnet)  #For penalised regression modelling
library(mlbench) #ForDatasets
library(pROC) #For AUC
library(rpart)
library(ipred)  #Bagging

#//////////////////////////////////"Part 1"//////////////////////////////////#
#(A)
mydata <- read.csv("MLDATASET_PartiallyCleaned.csv", 
                   header=TRUE);  #Read in the data
View(mydata)  
dim(mydata)
str(mydata)
sum(is.na(mydata))
#(B)
#(i)For How.Many.Times.File.Seen, set all values = 65535 to NA;
boxplot(mydata$How.Many.Times.File.Seen)
mydata$How.Many.Times.File.Seen[mydata$How.Many.Times.File.Seen == 65535] <- NA
boxplot(mydata$How.Many.Times.File.Seen)
mydata$How.Many.Times.File.Seen

#(ii)Convert Threads.Started to factors 1,2,3,4,5 
mydata$Threads.Started[mydata$Threads.Started > 5] <- 5
mydata$Threads.Started = as.factor(mydata$Threads.Started)
mydata$Threads.Started

#(iii)Log-transform Characters.in.URL 
hist(mydata$Characters.in.URL, col='steelblue', main='Original')
log_y <- log10(mydata$Characters.in.URL)
mydata$Characters.in.URL = log_y
hist(mydata$Characters.in.URL, col='coral2', main='Log Transformed')


#(iv)Select only the complete cases 
MLDATASET.cleaned <- na.omit(mydata)
view(MLDATASET.cleaned)
str(MLDATASET.cleaned)
dim(MLDATASET.cleaned)


#Create the training and test datasets

set.seed(10474901)  #Set the random seed.

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(MLDATASET.cleaned$Actually.Malicious, #The outcome variable
                                       p=0.3, #proportion of data to form the training set
                                       list=FALSE  #Don't store the result in a list
);

# Step 2: Create the training  dataset
trainData <- MLDATASET.cleaned[trainRowNumbers,]

# Step 3: Create the test dataset
testData <- MLDATASET.cleaned[-trainRowNumbers,]

write.csv(trainData,"trainData.csv")
write.csv(testData,"testData.csv")

#//////////////////////////////"Part 2"///////////////////////////////////////#
#Compare the performances of different machine learning algorithms
library(tidyverse)
set.seed(10474901)
models.list1 <- c("Logistic Ridge Regression",
                  "Logistic LASSO Regression",
                  "Logistic Elastic-Net Regression")
models.list2 <- c("Classification Tree",
                  "Bagging Tree",
                  "Random Forest")
myModels <- c("Binary Logistic Regression",
              sample(models.list1,size=1),
              sample(models.list2,size=1))
myModels %>% data.frame 

#Convert the categorical features into factors
temp <- lapply(MLDATASET.cleaned[,(c("Download.Source","TLD","Download.Speed","Executable.Code.Maybe.Present.in.Headers",
                                     "Threads.Started","Evidence.of.Code.Obfuscation","Actually.Malicious","Initial.Statistical.Analysis"))],as.factor); #2 mean apply to all colustr(temp)
str(temp)
MLDATASET.cleaned <- data.frame(Sample.ID =MLDATASET.cleaned$Sample.ID ,Mean.Word.Length.of.Extracted.Strings=MLDATASET.cleaned$Mean.Word.Length.of.Extracted.Strings,
                         Similarity.Score=MLDATASET.cleaned$Similarity.Score,
                         Characters.in.URL=MLDATASET.cleaned$Characters.in.URL,
                         Calls.To.Low.level.System.Libraries=MLDATASET.cleaned$Calls.To.Low.level.System.Libraries,
                         How.Many.Times.File.Seen =MLDATASET.cleaned$How.Many.Times.File.Seen,
                         File.Size.Bytes=MLDATASET.cleaned$File.Size.Bytes,
                         Ping.Time.To.Server=MLDATASET.cleaned$Ping.Time.To.Server, temp); 

str(MLDATASET.cleaned)
#Create the training and test datasets after it converted to factors
set.seed(10474901)  #Set the random seed.

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(MLDATASET.cleaned$Actually.Malicious, #The outcome variable
                                       p=0.3, #proportion of data to form the training set
                                       list=FALSE  #Don't store the result in a list
);

# Step 2: Create the training  dataset
trainData <- MLDATASET.cleaned[trainRowNumbers,]

# Step 3: Create the test dataset
testData <- MLDATASET.cleaned[-trainRowNumbers,]

trainData <- trainData[,2:15]
view(trainData)
view(testData)
str(trainData)
str(testData)

#######################Binary Logistic regression modelling#######################

mod.MLDATASET.lg <- glm(Actually.Malicious ~., family="binomial", data=trainData);
summary(mod.MLDATASET.lg) 

#predicted probability of actually malicious on the test data
pred.prob <- predict(mod.MLDATASET.lg,new=testData,type="response") 
pred.class <- ifelse(pred.prob>0.5,"YES","NO")


#Confusion matrix with re-ordering of "Yes" and "No" responses
cf.lg <- table(pred.class %>% as.factor %>% relevel(ref="YES"), 
               testData$Actually.Malicious %>% as.factor %>% relevel(ref="YES"));  

prop <- prop.table(cf.lg,2); prop %>% round(digit=3) #Proportions by columns

#Summary of confusion matrix
confusionMatrix(cf.lg);

############################Logistic Ridge Regression############################

lambdas <- 10^seq(-3,3,length=100) #A sequence 100 lambda values
set.seed(10474901)
mod.MLDATASET.ridge <- train(Actually.Malicious~., #Formula
                         data = trainData, #Training data
                         method = "glmnet", #Penalised regression modelling
                         #Set to c("center", "scale") to standardise data
                         preProcess = NULL,
                         #Perform 10-fold CV, 2 times over.
                         trControl = trainControl("repeatedcv",
                                                  number = 10,
                                                  repeats = 2),
                         tuneGrid = expand.grid(alpha = 0, #Ridge regression
                                                lambda = lambdas)
)

#Optimal lambda value
mod.MLDATASET.ridge$bestTune
  
# Model coefficients
coef(mod.MLDATASET.ridge$finalModel, mod.MLDATASET.ridge$bestTune$lambda)

#predicted probability of actually malicious on the test data
pred.class.ridge <- predict(mod.MLDATASET.ridge,new=testData) 

#Confusion matrix with re-ordering of "Yes" and "No" responses
cf.ridge <- table(pred.class.ridge %>% as.factor %>% relevel(ref="YES"), 
                  testData$Actually.Malicious %>% as.factor %>% relevel(ref="YES"));  

prop <- prop.table(cf.ridge,2); prop %>% round(digit=3) #Proportions by columns

#Summary of confusion matrix
confusionMatrix(cf.ridge)


#//////////////////////////////////////////////////////////////////////////////#
#Bagging Tree 
set.seed(10474901)
btree <- bagging(Actually.Malicious~.,
                    data=trainData,
                    nbagg=25,  
                    coob=TRUE); 
btree

## --------------------------------------------------------------------------------
#Summary of predictions on test set
test.pred <- predict(btree,newdata=testData,type="class"); 

test.cf <- confusionMatrix(test.pred %>% relevel(ref="YES"),
                              testData$Actually.Malicious %>% relevel(ref="YES"))
test.cf


## --------------------------------------------------------------------------------
#Intialise the hyperparamter search grid
grid <- expand.grid(nbagg=seq(25,150,25),  #A sequence of nbagg values
                       cp=seq(0,0.5,0.1),  #A sequence of cp values
                       minsplit=seq(5,20,5),  #A sequence of minsplits values
                       #Initialise columns to store the OOB misclassification rate
                       OOB.misclass=NA, 
                       #Initialise columns to store sensitivity, specificity and
                       #accuracy of bagging at each run.
                       test.sens=NA,
                       test.spec=NA,
                       test.acc=NA)
View(grid)


for (I in 1:nrow(grid))
{
  set.seed(10474901)
  
  #Perform bagging
  btree <- bagging(Actually.Malicious~.,
                      data=trainData,
                      nbagg=grid$nbagg[I],  
                      coob=TRUE,
                      control=rpart.control(cp=grid$cp[I],
                                            minsplit=grid$minsplit[I]));
  
  #OOB misclassification rate
  grid$OOB.misclass[I] <- btree$err*100
  
  #Summary of predictions on test set
  test.pred <- predict(btree,newdata=testData,type="class");
  
  if(nlevels(test.pred)==1) {
    
    levels(test.pred) <- c("NO","YES")
    
  };
  
  test.cf <- confusionMatrix(test.pred %>% relevel(ref="YES"),
                             testData$Actually.Malicious %>% relevel(ref="YES"))
  
  
  prop.cf <- test.cf$table %>% prop.table(2)
  grid$test.sens[I] <- prop.cf[1,1]*100  #Sensitivity
  grid$test.spec[I] <- prop.cf[2,2]*100  #Specificity
  grid$test.acc[I] <- test.cf$overall[1]*100  #Accuracy
}

#Sort the results by the OOB misclassification rate and display them.
grid[order(grid$OOB.misclass,decreasing=FALSE)[1:10],] %>% round(2)

#
str(testData)
#confusionMatrix(data= testData$Initial.Statistical.Analysis, reference =testData$Initial.Statistical.Analysis );

length(which(testData$Initial.Statistical.Analysis == "Correctly Identified as Clean")) 
length(which(testData$Initial.Statistical.Analysis == "Incorrectly Identified as Clean"))
length(which(testData$Initial.Statistical.Analysis == "Correctly Identified as Malware"))
length(which(testData$Initial.Statistical.Analysis == "Incorrecty Identified as Malware"))
