#' Clean a vector of sentences
#'
#' @param words character vector
#'
#' @return list each sentence is it's own character vector
#' @export
#'
#' @examples
#' sentences <- c("hello my name is","this is another sentence")
#' sl_clean_vector(sentences)
sl_clean_vector <- function(words){
  return(lapply(unlist(strsplit(words,split="[.]")),sl_clean))
}