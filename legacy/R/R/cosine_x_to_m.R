#' Cosine vector to matrix
#'
#' @param x numeric vector
#' @param m numeric matrix
#'
#' @return vector of cosine similarities between x and m
#' @export
#'
#' @examples
#' vec <- c(1,2,3)
#' mat <- matrix(c(1,2,3,3,2,1,2,1,3),ncol=3)
#' cosine_x_to_m(vec,mat)
#' 
cosine_x_to_m  <- function(x, m) {
  x <- x / as.numeric(sqrt(crossprod(x)))
  sims<-(m %*% x / sqrt(rowSumsSq(m)))
  return(sims)
}
