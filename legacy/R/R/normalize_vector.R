#' Normalize a vector by max absolute value
#'
#' @param x numeric vector, a vector of numbers
#'
#' @return numeric vector, the vector divided by the largest absolute value in the vector
#' @examples
#' a <- c(1,2,3)
#' normalize_vector(a)
#' 
#' @export
normalize_vector <- function (x) {return(x/abs(max(x)))}