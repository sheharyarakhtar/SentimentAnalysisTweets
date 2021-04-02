library(qdapRegex)
library(tm)
library(wordcloud)
library(gmodels)
library(e1071)


train <- read.csv('train.csv')
test <- read.csv('test.csv')
stop_words <- as.vector(read.csv('stop_words.txt')$i)

#Save the sentiment as a categorical variable to view the proportions of
#training and test sets
train$Sentiment <- as.factor(train$Sentiment)
test$Sentiment <- as.factor(test$Sentiment)
prop.table(table(train$Sentiment))
prop.table(table(test$Sentiment))

#Make all the letters in tweets small letters
#because regex will not remove handle tags if the @ is followed by a capital
train$Tweet <- tolower(train$Tweet)
test$Tweet <- tolower(test$Tweet)

#remove hashes, handles and urls from all tweets
train$Tweet <- rm_hash(rm_tag(rm_url(train$Tweet)))
test$Tweet <- rm_hash(rm_tag(rm_url(test$Tweet)))

#create corpus for train and test set
trainCorp <- Corpus(VectorSource(train$Tweet))
testCorp <- Corpus(VectorSource(test$Tweet))


##the added stop words were added after running the word cloud and finding
#that these words were used heavily in all three categories of sentiment
#which will not help the algorithm. If excluded, the accuracy decreased to 62%
stop_words <- append(stop_words,
                     c('flight', 
                       'usairways',
                       'americanair', 
                       'flights',
                       'southwestair', 
                       'jetblue',
                       stopwords()))


#data cleaning of train and test Corpus
CC <- tm_map(trainCorp, removePunctuation)
CC <- tm_map(CC, removeNumbers)
CC <- tm_map(CC, removeWords, stop_words)
CC <- tm_map(CC, stemDocument)
CC <- tm_map(CC, stripWhitespace)
trainCorp <- CC
CC <- tm_map(testCorp, removePunctuation)
CC <- tm_map(CC, removeNumbers)
CC <- tm_map(CC, removeWords, stop_words)
CC <- tm_map(CC, stemDocument)
CC <- tm_map(CC, stripWhitespace)
testCorp <- CC
rm(CC)

#View word cloud and if there are terms that are too repitative in both the
#train and test dataset, remove them from both
wordcloud(trainCorp, min.freq = 10, random.order = F)
wordcloud(testCorp, min.freq = 5, random.order = F)


#Updating the original tweet datasets in their dataframe
x <- data.frame(text = sapply(trainCorp, as.character), stringsAsFactors = F)
train$Tweet <- x$text
x <- data.frame(text = sapply(testCorp, as.character), stringsAsFactors = F)
test$Tweet <- x$text
rm(x)

#visualising the words in different categories of tweets
negative <- subset(train, Sentiment=='negative')
positive <- subset(train, Sentiment== 'positive')
neutral <- subset(train, Sentiment == 'neutral')
wordcloud(negative$Tweet, max.words = 70, random.order = F)
wordcloud(positive$Tweet, max.words = 100, random.order = F)
wordcloud(neutral$Tweet, max.words = 70, random.order = F)


#Create a dictionary of the terms recurring most often in the training set
#and remove the sparse terms from both datasets
trainDTM <- DocumentTermMatrix(trainCorp)
sms_dict<-findFreqTerms(trainDTM,5)
trainDTM<-DocumentTermMatrix(trainCorp,list(dictionary=sms_dict))
testDTM<-DocumentTermMatrix(testCorp,list(dictionary=sms_dict))

#Convert function to change the values of the terms in the DTM
#into a binary yes/no answer
convert_count <- function(x)
{
  x <- ifelse(x>0,1,0)
  x <- factor(x, level = c(0,1), label = c('No', 'Yes'))
  return(x)
}
trainDTM <- apply(trainDTM, MARGIN = 2, convert_count)
testDTM <- apply(testDTM, MARGIN = 2, convert_count)


#model
start.time <- Sys.time()
tweet_classifier <- naiveBayes(as.matrix(trainDTM),train$Sentiment)
tweet_test_pred <- predict(tweet_classifier, as.matrix(testDTM))
total.time <- Sys.time() - start.time
total.time

CrossTable(x = test$Sentiment, y= tweet_test_pred, prop.chisq = FALSE, 
           prop.t = FALSE, dnn = c('actual', 'predicted'))
