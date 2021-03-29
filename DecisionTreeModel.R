
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
train.tokens.matrix <- train.tokens.matrix[,intersect(colnames(test.tokens.matrix), colnames(train.tokens.matrix))]
test.tokens.matrix <- test.tokens.matrix[,intersect(colnames(test.tokens.matrix),colnames(train.tokens.matrix))]
train.tokens.dataframe <- as.data.frame(train.tokens.matrix)
test.tokens.dataframe <- as.data.frame(test.tokens.matrix)

rm(train.tokens.matrix)
rm(test.tokens.matrix)
rm(train.tokens.dfm)
rm(test.tokens.dfm)
rm(train)
rm(test)


convert_count <- function(x)
{
  x <- ifelse(x>0,1,0)
  x <- factor(x, level = c(0,1), label = c('No', 'Yes'))
  return(x)
}

train.tokens.dataframe <- sapply(train.tokens.dataframe, convert_count)
test.tokens.dataframe <- sapply(test.tokens.dataframe, convert_count)
train.tokens.dataframe <- as.data.frame(train.tokens.dataframe)
test.tokens.dataframe <- as.data.frame(test.tokens.dataframe)


test.labels <- as.factor(test.labels)
train.labels <- as.factor(train.labels)
library(C50)
start.time <- Sys.time()
m <- C5.0(train.tokens.dataframe, train.labels, trials = 5)
pred <- predict(m, test.tokens.dataframe)
total.time <- Sys.time() - start.time
total.time

CrossTable(pred, test.labels)
