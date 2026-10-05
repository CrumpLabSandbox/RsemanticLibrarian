#' Get all unique words from corpus
#'
#' @param words character vector
#'
#' @return character vector of unique words
#' @description uses the breakdown function from LSAfun, which converts to lowercase and removes most special characters. After breakdown, all unique words are returned, forming the dictionary for the corpus. 
#' @export
#'
#' @examples
#' 
#' sl_corpus_dictionary("The the three","more stuff")
sl_corpus_dictionary <- function(words){
  neat_words <- words %>%
                as.character() %>%
                strsplit(split=" ") %>%
                unlist() %>%
                LSAfun::breakdown() %>%
                qdapRegex::rm_white_lead_trail() %>%
                strsplit(split=" ") %>%
                unlist() %>%
                unique()
  return(neat_words)
}
