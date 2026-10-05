#' Compute similarity between an abstract and all other abstracts
#'
#' @param a_title charcter, the title of an abstract
#' @param a_vectors matrix, of semantic vectors for the abstracts (e.g., `article_vectors`)
#' @param a_df dataframe, of abstract information (e.g., `article_df``)
#'
#' @return dataframe, of abstract information with Similiarities to target abstract appended
#' @export
#'
#' @examples
#' 
#' an_article <- article_df[1,]$title
#' sims <- get_article_article_similarities(an_article, article_vectors, article_df)

get_article_article_similarities <- function(a_title,
                                             a_vectors = article_vectors,
                                             a_df = article_df){
  
  a_id <- a_df[a_df$title == a_title,]$index
  
  get_cos <-  cosine_x_to_m(a_vectors[a_id,],a_vectors)
  article_sims <- a_df %>%
    dplyr::mutate(Similarity = round(as.numeric(get_cos),digits=4)) %>%
    dplyr::select(formatted_column,
           title,
           wrap_title,
           year,
           index,
           Similarity) %>%
    dplyr::arrange(desc(Similarity))
  return(article_sims)
}
