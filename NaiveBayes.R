library(qdapRegex)
library(tm)
library(wordcloud)
library(gmodels)
library(e1071)


train <- read.csv('train.csv')
test <- read.csv('test.csv')
stop_words <- as.vector(read.csv('stop_words.txt')$i)

train$Sentiment <- as.factor(train$Sentiment)
test$Sentiment <- as.factor(test$Sentiment)
prop.table(table(train$Sentiment))
prop.table(table(test$Sentiment))

train$Tweet[1:10]
train$Tweet <- tolower(train$Tweet)
test$Tweet <- tolower(test$Tweet)

train$Tweet <- rm_hash(rm_tag(rm_url(train$Tweet)))
test$Tweet <- rm_hash(rm_tag(rm_url(test$Tweet)))

trainCorp <- Corpus(VectorSource(train$Tweet))
testCorp <- Corpus(VectorSource(test$Tweet))


stop_words <- append(stop_words, c('flight', 'usairways','americanair', 'flights','southwestair', 'jetblue', stopwords()))


inspect(CC[1:10])
CC <- tm_map(trainCorp, removePunctuation)
CC <- tm_map(CC, removeNumbers)
CC <- tm_map(CC, removeWords, stop_words)
CC <- tm_map(CC, stemDocument)
CC <- tm_map(CC, stripWhitespace)
trainCorp <- CC

inspect(CC[1:10])
CC <- tm_map(testCorp, removePunctuation)
CC <- tm_map(CC, removeNumbers)
CC <- tm_map(CC, removeWords, stop_words)
CC <- tm_map(CC, stemDocument)
CC <- tm_map(CC, stripWhitespace)
testCorp <- CC
rm(CC)

wordcloud(trainCorp, min.freq = 10, random.order = F)
wordcloud(testCorp, min.freq = 5, random.order = F)

x <- data.frame(text = sapply(trainCorp, as.character), stringsAsFactors = F)
train$Tweet <- x$text
x <- data.frame(text = sapply(testCorp, as.character), stringsAsFactors = F)
test$Tweet <- x$text
rm(x)

negative <- subset(train, Sentiment=='negative')
positive <- subset(train, Sentiment== 'positive')
neutral <- subset(train, Sentiment == 'neutral')

wordcloud(negative$Tweet, max.words = 70, random.order = F)
wordcloud(positive$Tweet, max.words = 100, random.order = F)
wordcloud(neutral$Tweet, max.words = 70, random.order = F)


trainDTM <- DocumentTermMatrix(trainCorp)
testDTM <- DocumentTermMatrix(testCorp)

convert_count <- function(x)
{
  x <- ifelse(x>0,1,0)
  x <- factor(x, level = c(0,1), label = c('No', 'Yes'))
  return(x)
}

trainDTM <- apply(trainDTM, MARGIN = 2, convert_count)
testDTM <- apply(testDTM, MARGIN = 2, convert_count)


start.time <- Sys.time()
tweet_classifier <- naiveBayes(as.matrix(trainDTM),train$Sentiment)
tweet_test_pred <- predict(tweet_classifier, as.matrix(testDTM))

total.time <- Sys.time() - start.time
total.time

CrossTable(x = test$Sentiment, y= tweet_test_pred, prop.chisq = FALSE, prop.t = FALSE, dnn = c('actual', 'predicted'))
