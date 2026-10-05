#' Create random index vectors
#'
#' @param dim integer, number of dimensions (vector length)
#' @param inds integer, number of rows
#' @param sparsity even integer, number of elements in vector
#'
#' @return matrix, each row is an item with dim dimensions
#' @export
#'
#' @examples
#' 
#' sl_create_riv(10,5,1)
#' sl_create(10,5,4)
sl_create_riv <- function(dim,inds,sparsity){
  if(sparsity %% 2 != 0) stop("sparsity must be even integer")
  base_vector <- c(rep(0,(dim-sparsity)),rep(1,(sparsity/2)),rep(-1,(sparsity/2)))
  temp_matrix <- matrix(0,ncol=dim,nrow=inds)
  for(i in 1:inds){
    temp_matrix[i,] <- sample(base_vector)
  }
  return(temp_matrix)
}