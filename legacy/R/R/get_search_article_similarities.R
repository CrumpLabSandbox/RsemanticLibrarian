#' Similarities between term vector and article vector
#'
#' @param search_terms character vector containing search terms (words must appear in dictionary)
#' @param d_words character vector containing the corpus dictionary, the indexes of the words in the dictionary should match the row indices of the `WordVectors` matrix
#' @param w_vectors matrix containing semantic vectors for each word (e.g., `WordVectors` from the sample data in this package)
#' @param a_vectors matrix containing semantic vectors for each article (e.g., `article_vectors` from the sample data)
#' @param a_df dataframe containing necessary abstract information (e.g., `article_df` from sample data)
#' @param query_type integer, 1 = compound search, 2 = AND search, 3 = OR search
#'
#' @return a dataframe containing article information
#' @details The returned dataframe includes the following: 
#'  - `formatted_column`: html printable abstract information
#'  - `title`: abstract title
#'  - `wrap_title`: wrapped title with html breaks
#'  - `year`: abstract year
#'  - `index`: index refering to the row vector in the `article_vectors` matrix
#'  - `Similarity`: cosine similarity between search terms and current abstract
#' @export
#'
#' @examples
#' search <- c("president")
#' article_dataframe <- get_search_article_similarities(search,query_type=1)
#' knitr::kable(head(article_dataframe))
#' 
get_search_article_similarities <- function(search_terms,
                                            d_words = dictionary_words,
                                            w_vectors = WordVectors,
                                            a_vectors = article_vectors,
                                            a_df = article_df,
                                            query_type){
  
  search_index <- which(d_words %in% search_terms)
  if(length(search_index > 0)){
    # compound vectors
    if (query_type == 1){
      
      if(length(search_index) > 1) {
        query_vector <- colSums(w_vectors[search_index,])
      }else{
        query_vector <- w_vectors[search_index,]
      }
      
      get_cos <-  cosine_x_to_m(query_vector,
                                a_vectors)
      
      article_sims_SS <- a_df %>%
        dplyr::mutate(Similarity = as.numeric(get_cos))%>%
        dplyr::select(formatted_column,
               title,
               wrap_title,
               year,
               index,
               Similarity) %>%
        dplyr::arrange(desc(Similarity))
      return(article_sims_SS)
    } else {
      if(length(search_index) > 1) {
        query_matrix     <- w_vectors[search_index,]
        get_cos_matrix   <- apply(query_matrix,1,
                                  function(x) cosine_x_to_m(x,a_vectors))
        if (query_type == 2) {
          multiply_columns <- apply(get_cos_matrix,1,prod)
        } else if (query_type == 3){
          get_cos_matrix <- apply(get_cos_matrix,2,normalize_vector)
          multiply_columns <- apply(get_cos_matrix,1,max)
        }
        
        article_sims_SS <- a_df %>%
          dplyr::mutate(Similarity = round(multiply_columns,digits=4))%>%
          dplyr::select(formatted_column,
                 title,
                 wrap_title,
                 year,
                 index,
                 Similarity) %>%
          dplyr::arrange(desc(Similarity))
        return(article_sims_SS)
      }
    }
  }
}
