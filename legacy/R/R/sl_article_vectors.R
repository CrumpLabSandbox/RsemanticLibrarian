#' Create article vectors from word vectors
#'
#' @param s_ids list of integer vectors with ids for each word in article
#' @param w_matrix matrix of word vectors
#' @param verbose logical, controls printing default=FALSE
#'
#' @return matrix of article vectors
#' @export
#'
#' @examples
#' 
sl_article_vectors <- function(s_ids,
         w_matrix = word_vectors,
         verbose = FALSE){
  
  article_vectors <- matrix(0, ncol = dim(w_matrix)[2], nrow = length(s_ids))
  for(i in 1:length(s_ids)) {
    if(verbose == TRUE) print(i)
    if(length(s_ids[[i]]) > 1 ){
      article_vectors[i,] <- colSums(w_matrix[s_ids[[i]],])
    } 
    
    if(length(s_ids[[i]]) == 1 ){
      article_vectors[i,] <- w_matrix[s_ids[[i]],]
    }
    
    if(length(s_ids[[i]]) == 0 ){
      article_vectors[i,] <- w_matrix[1,]
    }
    
  }
  return(article_vectors)
  
}