#' Create author vectors
#'
#' @param at_list character vector of author names
#' @param articles dataframe with abstract info
#' @param a_vectors matrix of semantic article vectors
#'
#' @return matrix of author vectors (sum of abstracts)
#' @export
#'
#' @examples
#' 
sl_author_vectors <- function(at_list,
                           articles = article_df,
                           a_vectors = article_vectors) {
  author_vectors <- matrix(0, ncol = dim(a_vectors)[2], nrow = length(at_list))
  for(i in 1:length(at_list)){
    #auth_ids <- which(articles$authorlist %in% at_list[i])
    auth_ids <- which(stringr::str_detect(articles$authorlist,at_list[i]))
                  
    if(length(auth_ids) > 1) {
      author_vectors[i,] <- colSums(a_vectors[auth_ids,])
    } else {
      author_vectors[i,] <- a_vectors[auth_ids,]
    }
  }
  return(author_vectors)
}