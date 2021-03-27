install.packages('class')
install.packages('quanteda')
install.packages('caret')
library(caret)
library(quanteda)
library(class)


test <- read.csv('test.csv')
train <- read.csv('train.csv')
stop_word <- read.csv('stop_words.txt')

train.tokens <- tokens(train$Tweet, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE, remove_url = TRUE, split_hyphens = TRUE)

# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)

train.tokens <- tokens_select(train.tokens, stop_word, 
                              selection = "remove")


# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")

# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.labels <- train$Sentiment



test.tokens <- tokens(test$Tweet, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE, remove_url = TRUE)

# Lower case the tokens.
test.tokens <- tokens_tolower(test.tokens)

test.tokens <- tokens_select(test.tokens, stop_word, 
                             selection = "remove")


# Perform stemming on the tokens.
test.tokens <- tokens_wordstem(test.tokens, language = "english")

# Create our first bag-of-words model.
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)


# Transform to a matrix and inspect.
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.labels <- test$Sentiment


#Make both matrix columns the same length based on the ones that overlap
train.tokens.dataframe <- train.tokens.dataframe[,intersect(colnames(test.tokens.dataframe), colnames(train.tokens.dataframe))]
test.tokens.dataframe <- test.tokens.dataframe[,intersect(colnames(test.tokens.dataframe),colnames(train.tokens.dataframe))]

#Normalize the count for all the features
normalize <- function(x)
{
  return((x-min(x))/(max(x)-min(x)))
}

train_n <- as.data.frame(lapply(train.tokens.dataframe, normalize))
test_n <- as.data.frame(lapply(test.tokens.dataframe, normalize))

#Using kNN model to test the testdataset
pred <- knn(test = test_n, train= train_n, k = 97, cl = train.labels)

#Evauating result
CrossTable(x = pred, y = test.labels, chisq = T)