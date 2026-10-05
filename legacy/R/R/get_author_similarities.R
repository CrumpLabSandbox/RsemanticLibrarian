#' Compute Similarities between one author, and all other authors
#'
#' @param a_name character, name of an author
#' @param auth_vectors matrix of semantic vectors for each author
#' @param authors character vector of all author names
#'
#' @return dataframe of similarities between target author and all other authors
#' @export
#'
#' @examples
#' 
#' sims <- get_author_similarities(author_list[1])
get_author_similarities <- function(a_name,
                                    auth_vectors = AuthorVectors,
                                    authors=author_list){
  auth_id <- which(authors %in% a_name)
  get_cos <-  cosine_x_to_m(auth_vectors[auth_id,],auth_vectors)
  selected_author <- rep("other",length(authors))
  selected_author[auth_id] <- "selected"
  selected_author <- as.factor(selected_author)
  author_sims <- data.frame(author=authors,
                            Similarity = round(as.numeric(get_cos),digits=4),
                            selected_author = selected_author,
                            index = 1:length(authors)) %>%
    dplyr::arrange(desc(Similarity))
  return(author_sims)
}
