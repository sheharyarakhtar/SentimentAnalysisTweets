tweet_raw <- rbind(read.csv('train.csv'), read.csv('test.csv'))
tweet_raw$Sentiment <- as.factor(tweet_raw$Sentiment)
library(tm)
stop_word <- read.csv('stop_words.txt')
tweet_corpus <- Corpus(VectorSource(tweet_raw$Tweet))
inspect(tweet_corpus[1])
 
corpus_clean <- tm_map(tweet_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

tweet_dtm <- DocumentTermMatrix(corpus_clean)

tweet_raw_train <- tweet_raw[1:11680,]
tweet_raw_test <- tweet_raw[11681:14601,]

tweet_dtm_train <- tweet_dtm[1:11680,]
tweet_dtm_test <- tweet_dtm[11681:14601,]

tweet_corpus_train <- corpus_clean[1:11680]
tweet_corpus_test <- corpus_clean[11681:14601]


install.packages('wordcloud')
library(wordcloud)

wordcloud(tweet_corpus_train, min.freq = 40, random.order = F)
negative <- subset(tweet_raw_train, Sentiment='negative')
positive <- subset(tweet_raw_train, Sentiment = 'positive')
neutral <- subset(tweet_raw_train, Sentiment = 'neutral')

wordcloud(negative$Tweet, max.words = 70, scale = c(3,0.5), random.order = F)
wordcloud(positive$Tweet, max.words = 70, scale = c(3,0.5), random.order = F)
wordcloud(neutral$Tweet, max.words = 70, scale = c(3,0.5), random.order = F)

#remove terms with freq less than 5
tweet_dict<-findFreqTerms(tweet_dtm_train,  1)
tweet_train<-DocumentTermMatrix(tweet_corpus_train,list(dictionary=tweet_dict))
tweet_test<-DocumentTermMatrix(tweet_corpus_test,list(dictionary=tweet_dict))

#naive bayes is trained on categorical data
#so the repition of words is not important, only their presense
#We make a function to convert the count to Y/N factor
convert_count <- function(x)
{
  x <- ifelse(x>0,1,0)
  x <- factor(x, level = c(0,1), label = c('No', 'Yes'))
  return(x)
}

tweet_train <- apply(tweet_raw_train, MARGIN = 2, convert_count)
tweet_test <- apply(tweet_raw_test, MARGIN = 2, convert_count)

#library for naive bayes
library(e1071)
tweet_classifier <- naiveBayes(as.matrix(tweet_train), tweet_raw_train$Sentiment)
tweet_test_pred <- predict(tweet_classifier, as.matrix(tweet_test))

CrossTable(x = tweet_raw_test$Sentiment, y= tweet_test_pred, prop.chisq = FALSE, prop.t = FALSE, dnn = c('actual', 'predicted'))
