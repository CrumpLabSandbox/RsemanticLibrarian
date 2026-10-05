#' Clean words to lowercase etc.
#'
#' @param words character
#'
#' @return character vector
#' @export
#'
#' @examples
#' sl_clean("Hello, this is a sentence I want to clean.")
#' 
sl_clean <- function(words){
  if(length(words) == 1) {
    neat_words <- words %>%
      as.character() %>%
      strsplit(split=" ") %>%
      unlist() %>%
      LSAfun::breakdown() %>%
      qdapRegex::rm_white_lead_trail() %>%
      strsplit(split=" ") %>%
      unlist() 
    return(neat_words)
  }
}