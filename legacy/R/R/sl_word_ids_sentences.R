#' Get indexes for each word in sentence
#'
#' @param sentences list of character vectors for each sentence
#' @param words dictionary of words in corpus
#'
#' @return list of index vectors for each word in sentence
#' @export
#'
#' @examples
#' 
#' my_words <- c("hello","this","is")
#' my_sentences <-list(c("hello","is"),c("this","is"))
#' sl_word_ids_sentences(my_sentences,my_words)
#' 
sl_word_ids_sentences <- function(sentences,words){
  ids <- list()
  for(i in 1:length(sentences)){
    ids[[i]] <- match(sentences[[i]],words)
  }
  return(ids)
}