
# 00 data & library -------------------------------------------------------


library(text2vec)
library(devtools)
data("AirPassengers")

# 01 glove ----------------------------------------------------------------


wiki = readLines(text8_file, n = 1, warn = FALSE)

# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)

vocab <- prune_vocabulary(vocab, term_count_min = 5L)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
glove$fit(tcm, n_iter = 20)

glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
# `glove` object will be modified by `fit()` call !
fit(tcm, glove, n_iter = 20)

word_vectors <- glove$get_word_vectors()

berlin <- word_vectors["paris", , drop = FALSE] - 
  word_vectors["france", , drop = FALSE] + 
  word_vectors["germany", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = berlin, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 5)
# berlin     paris    munich    leipzig   germany 
# 0.8015347 0.7623165 0.7013252 0.6616945 0.6540700



# 02 word2vec -------------------------------------------------------------
devtools::install_github("bmschmidt/wordVectors")


