#' Create BEAGLE vectors for words
#'
#' @param s_ids list of sentences as character vectors
#' @param dictionary character vector of all words in the dictionary
#' @param riv integer vector c(dim,sparsity) for number of dimensions and sparsity
#' @param verbose integer printing option to track progress report every n times
#'
#' @return matrix each row is the semantic vector for each word in dictionary
#' @export
#'
#' @examples
#' 
sl_beagle_vectors <- function(s_ids,
                           dictionary = dictionary_words,
                           riv = c(1024,30),
                           verbose = 500){
  all_words_num <- length(dictionary)
  num_sentences <- length(s_ids)
  e_matrix <- sl_create_riv(dim=riv[1],inds=all_words_num,sparsity=riv[2])
  w_matrix <- e_matrix
  for( s in 1:num_sentences ){
    if(s %% verbose==0){
      print(s)
    }
    num_words <- length(s_ids[[s]])
    if(num_words > 1){
      sentence_matrix <- e_matrix[s_ids[[s]],]
      sentence_summed_vector <- colSums(sentence_matrix)
      for(w in 1:num_words){
        w_context <- sentence_summed_vector-sentence_matrix[w,]
        w_matrix[s_ids[[s]][w],] <- w_matrix[s_ids[[s]][w],] + normalize_vector(w_context)
        w_matrix[s_ids[[s]][w],] <- normalize_vector(w_matrix[s_ids[[s]][w],])
      }
    }
  }
  
  return(w_matrix)
}