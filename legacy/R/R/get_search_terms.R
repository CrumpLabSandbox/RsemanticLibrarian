#' Get search terms in dictionary
#'
#' @param words character a string with words
#' @param dictionary character vector containing all words in corpus
#'
#' @return character vector with words that are in the dictionary
#' @details uses the breakdown function from LSAfun to get rid of special characters and convert to lowercase. The `dictionary_words` variable contains 100 sample words.
#'
#' @examples
#' search <- "foreward firetruck by the president"
#' corpus <- dictionary_words
#' get_search_terms(search,corpus)
#' @export
get_search_terms <- function(words,dictionary){
  breakdown_words <- LSAfun::breakdown(words)
  search_items <- unlist(strsplit(breakdown_words,split=" "))
  search_items <- search_items[search_items %in% dictionary==TRUE]
  return(search_items)
}