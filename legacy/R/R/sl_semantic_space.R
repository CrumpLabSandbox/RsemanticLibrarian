#' Create semantic space from article dataframe
#'
#' @param abstract_df dataframe containing abstract info
#' @param col_names names of columns containing text (e.g., title, abstract)
#' @param riv_param integer vector (dims,sparsity)
#'
#' @return list containing
#' - `dictionary_words` character vector of unique words in corpus
#' - `word_vectors` matrix of semantic vectors for each word
#' - `article_vectors` matrix of semantic vectors for each article
#' - `author_list` character vector of authors
#' - `author_vectors` matrix of semantic vectors for each author
#' @export
#'
#' @examples
#' 
sl_semantic_space <- function(abstract_df,
                              col_names,
                              riv_param = c(1024,30)){
  
  # create corpus dictionary
  dictionary_words <- sl_corpus_dictionary(
    unlist(abstract_df[,col_names], use.names=FALSE)
  )
  message("Semantic Librarian: corpus dictionary created...")
  
  # convert to sentences, clean
  clean_sentences <- sl_clean_vector(
    unlist(abstract_df[,col_names], use.names=FALSE)
  )
  message("Semantic Librarian: Cleaned sentences...")
  
  # word ids for sentences
  sentence_ids <- sl_word_ids_sentences(clean_sentences,dictionary_words)
  message("Semantic Librarian: Created word IDs for sentences...")
  
  # Create word vectors
  message("Semantic Librarian: Creating BEAGLE word vectors...")
  word_vectors <- sl_beagle_vectors(s_ids = sentence_ids,
                                    dictionary = dictionary_words,
                                    riv = riv_param,
                                    verbose = 500)
  # get abstracts
  the_abstracts <- unite(abstract_df[,col_names],col="new",sep=".")
  clean_abstracts <- lapply(the_abstracts$new,
                            sl_clean)
  message("Semantic Librarian: Making abstracts...")
  
  abstract_word_ids <- sl_word_ids_sentences(clean_abstracts,dictionary_words)
  message("Semantic Librarian: Making abstract word IDs...")
  
  message("Semantic Librarian: Making abstract vectors...")
  article_vectors <- sl_article_vectors(abstract_word_ids, w_matrix = word_vectors)
  
  # get authors
  message("Semantic Librarian: Making author vectors...")
  
  authors <- as.character(abstract_df$authorlist)
  authors <- unlist(strsplit(authors,split=";"))
  authors <- trimws(authors, which="both")
  author_list <- unique(authors)
  
  author_vectors <- sl_author_vectors(at_list = author_list,
                                      articles = abstract_df,
                                      a_vectors = article_vectors)
  
  message("Semantic Librarian: Corpus Complete")
  
  
  return(list(dictionary_words = dictionary_words,
              word_vectors = word_vectors,
              article_vectors = article_vectors,
              author_list = author_list,
              author_vectors = author_vectors))
}