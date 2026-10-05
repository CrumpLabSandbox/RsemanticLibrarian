#' Multi-dimensional scaling for author similarity
#'
#' @param num_authors integer, number of authors
#' @param num_clusters intger, number of clusters to find
#' @param input_df dataframe of abstract information (e.g., `article_df`)
#' @param auth_vectors matrix of author vectors (e.g., `AuthorVectors`)
#'
#' @return dataframe of abstract information with MDS coordinates and cluster information
#' @export
#'
#' @examples
#' 
#' author_sims <- get_author_similarities(author_list[1])
#' author_MDS <- get_mds_author_fits(15,3,author_sims,AuthorVectors)
#' 
get_mds_author_fits <- function (num_authors,
                                 num_clusters,
                                 input_df,
                                 auth_vectors=AuthorVectors){
  auth_ids <- input_df[1:num_authors,]$index
  temp_author_matrix <- auth_vectors[auth_ids,]
  mdistance <- lsa::cosine(t(temp_author_matrix))
  fit <- cmdscale(1-mdistance,eig=TRUE, k=2)
  colnames(fit$points) <- c("X","Y")
  cluster <- kmeans(fit$points,num_clusters)
  input_df <- dplyr::slice(input_df,1:num_authors)
  input_df <- cbind(input_df,fit$points,
                    cluster=cluster$cluster)
  return(input_df)
}
