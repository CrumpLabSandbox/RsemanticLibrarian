#' Wrap a string
#'
#' @param x character, a string with words in it
#' @param n_chars numeric, number of characters before split
#' @param sep charcter, a character string with the separator, default is " <br> " for html
#'
#' @return character, a string, with the text wrapped (separate) by the `sep` delimiter
#' @export
#'
#' @examples
#' a <- "a really long string that you want to wrap"
#' wrap_string(a,n_chars = 20, sep = )
wrap_string <- function(x, n_chars, sep = " <br> ") {
  paste(strwrap(x,n_chars), collapse=sep)
}