# Generate reference outputs from the legacy R package for the Python parity tests.
#
# Run from the repository root:
#   LANG=C.UTF-8 Rscript scripts/legacy_reference.R
#
# Requires: dplyr, stringr, tidyr, magrittr, jsonlite. LSAfun (<= 0.6.2), qdapRegex and
# lsa are NOT required: the three functions used from them are reproduced below
# verbatim from their sources (LSAfun 0.6.2 breakdown(), qdapRegex rm_white_lead_trail(),
# lsa cosine()).
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(magrittr)
  library(jsonlite)
})

out_dir <- "tests/fixtures/legacy"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- shims -------------------------------------------------------------------------
# LSAfun 0.6.2 breakdown(). The original source writes the accented characters as
# latin-1 escapes ("\xe4"); the installed package declares them as those characters.
breakdown <- function(x) {
  x <- tolower(x)
  x <- gsub(x = x, pattern = "ä", replacement = "ae")
  x <- gsub(x = x, pattern = "ö", replacement = "oe")
  x <- gsub(x = x, pattern = "ü", replacement = "ue")
  for (p in c("à", "á", "â")) x <- gsub(x = x, pattern = p, replacement = "a")
  for (p in c("è", "é", "ê")) x <- gsub(x = x, pattern = p, replacement = "e")
  for (p in c("ì", "í", "î")) x <- gsub(x = x, pattern = p, replacement = "i")
  for (p in c("ò", "ó", "ô")) x <- gsub(x = x, pattern = p, replacement = "o")
  for (p in c("ù", "ú", "û")) x <- gsub(x = x, pattern = p, replacement = "u")
  x <- gsub(x = x, pattern = "ß", replacement = "ss")
  x <- iconv(x, to = "ASCII//TRANSLIT")
  x <- gsub(x = x, pattern = "[[:punct:]]", replacement = " ")
  x <- gsub(x = x, pattern = "[[:digit:]]", replacement = " ")
  x <- gsub(x = x, pattern = "\n", replacement = " ")
  x <- gsub(x = x, pattern = "\"", replacement = " ")
  x
}
# qdapRegex rm_white_lead_trail() = rm_default(pattern "^\\s+|\\s+$", trim, clean)
rm_white_lead_trail <- function(x) {
  out <- gsub("^\\s+|\\s+$", "", x, perl = TRUE)
  out <- gsub("^\\s+|\\s+$", "", out)
  gsub("\\s+", " ", gsub("\\\\r|\\\\n|\\n|\\\\t", " ", out))
}
# lsa::cosine() on a matrix: cosine between columns
lsa_cosine <- function(x) {
  co <- crossprod(x)
  n <- sqrt(colSums(x^2))
  co / outer(n, n)
}
rowSumsSq <- function(x) rowSums(x^2)

# ---- source the legacy package code with the shims substituted ---------------------
for (f in list.files("legacy/R/R", pattern = "\\.R$", full.names = TRUE)) {
  if (basename(f) %in% c("RcppExports.R", "RsemanticLibrarian-package.R", "utils-pipe.R")) next
  src <- readLines(f, warn = FALSE)
  src <- gsub("LSAfun::breakdown", "breakdown", src, fixed = TRUE)
  src <- gsub("qdapRegex::rm_white_lead_trail", "rm_white_lead_trail", src, fixed = TRUE)
  src <- gsub("lsa::cosine", "lsa_cosine", src, fixed = TRUE)
  eval(parse(text = src), envir = globalenv())
}
for (d in list.files("legacy/R/data", pattern = "\\.rda$", full.names = TRUE)) load(d)

# ---- 1. text cleaning --------------------------------------------------------------
clean_cases <- c(
  "Hello, this is a sentence I want to clean.",
  "The  BEAGLE model (Jones & Mewhort, 2007) encodes order-information.",
  "Don't split   contractions? e.g., i.e., 1990s; 3.14 -- dashes—em.",
  "Café naïve straße Köhler Æsop œuvre",
  "  leading and trailing  ",
  "tabs\there and\nnewlines",
  "Curly “quotes” and ‘single’ … ellipsis",
  "!!!",
  ""
)
cleaned <- lapply(clean_cases, function(s) {
  r <- sl_clean(s)
  if (is.null(r)) character(0) else r
})
bd <- breakdown(clean_cases)
dictionary_cases <- sl_corpus_dictionary(clean_cases)
sentence_cases <- sl_clean_vector(clean_cases[1:7])
write_json(
  list(
    inputs = clean_cases,
    breakdown = bd,
    sl_clean = cleaned,
    sl_corpus_dictionary = dictionary_cases,
    sl_clean_vector_inputs = clean_cases[1:7],
    sl_clean_vector = lapply(sentence_cases, function(x) if (is.null(x)) character(0) else x)
  ),
  file.path(out_dir, "cleaning.json"), auto_unbox = FALSE, pretty = TRUE
)

# ---- 2. BEAGLE (context-only, legacy) on a small corpus ----------------------------
n_docs <- 25
riv <- c(64, 6)
df <- article_df[1:n_docs, ]
for (cn in names(df)) if (is.factor(df[[cn]])) df[[cn]] <- as.character(df[[cn]])
col_names <- c("title", "abstract")
text_values <- unlist(df[, col_names], use.names = FALSE)

dictionary <- sl_corpus_dictionary(text_values)
sentences <- sl_clean_vector(text_values)
sentence_ids <- sl_word_ids_sentences(sentences, dictionary)

set.seed(20190625)
environment_matrix <- sl_create_riv(dim = riv[1], inds = length(dictionary), sparsity = riv[2])
set.seed(20190625)
word_vectors <- sl_beagle_vectors(sentence_ids, dictionary = dictionary, riv = riv, verbose = 1e9)
stopifnot(isTRUE(all.equal(word_vectors[0, ], environment_matrix[0, ])))

the_abstracts <- unite(df[, col_names], col = "new", sep = ".")
clean_abstracts <- lapply(the_abstracts$new, sl_clean)
abstract_ids <- sl_word_ids_sentences(clean_abstracts, dictionary)
doc_vectors <- sl_article_vectors(abstract_ids, w_matrix = word_vectors)

authors <- unique(trimws(unlist(strsplit(df$authorlist, split = ";")), which = "both"))
author_vectors <- sl_author_vectors(at_list = authors, articles = df, a_vectors = doc_vectors)
author_doc_rows <- lapply(authors, function(a) which(stringr::str_detect(df$authorlist, a)))

write_json(
  list(
    riv = riv,
    seed = 20190625,
    col_names = col_names,
    documents = lapply(seq_len(n_docs), function(i) list(
      index = df$index[i], title = df$title[i], abstract = df$abstract[i],
      authorlist = df$authorlist[i]
    )),
    dictionary = dictionary,
    sentences = lapply(sentences, function(x) if (is.null(x)) character(0) else x),
    authors = authors,
    author_doc_rows = author_doc_rows
  ),
  file.path(out_dir, "beagle_parity.json"), auto_unbox = TRUE, pretty = FALSE, digits = NA
)
# Plain little-endian float64 matrices; read in Python with numpy.fromfile.
write_matrix <- function(m, name) {
  con <- file(file.path(out_dir, paste0(name, ".f64")), "wb")
  writeBin(as.numeric(t(m)), con, size = 8, endian = "little")
  close(con)
  c(nrow(m), ncol(m))
}
shapes <- list(
  environment = write_matrix(environment_matrix, "beagle_environment"),
  word_vectors = write_matrix(word_vectors, "beagle_word_vectors"),
  doc_vectors = write_matrix(doc_vectors, "beagle_doc_vectors"),
  author_vectors = write_matrix(author_vectors, "beagle_author_vectors")
)
write_json(shapes, file.path(out_dir, "beagle_shapes.json"), auto_unbox = FALSE)

# ---- 3. search, similarity and MDS on the bundled 100-article sample ---------------
query <- "president of psychology"
terms <- get_search_terms(query, dictionary_words)
search_out <- list()
for (qt in 1:3) {
  r <- get_search_article_similarities(terms, query_type = qt)
  search_out[[as.character(qt)]] <- list(index = r$index, similarity = r$Similarity)
}
art <- get_article_article_similarities(article_df[1, ]$title)
auth <- get_author_similarities(author_list[1])
mds_in <- get_search_article_similarities(terms, query_type = 1)
set.seed(1)
mds <- get_mds_article_fits(25, 3, c(1900, 2000), mds_in)
set.seed(1)
amds <- get_mds_author_fits(15, 3, auth)

write_json(
  list(
    query = query,
    terms = terms,
    search = search_out,
    article_article = list(title = as.character(article_df[1, ]$title),
                           index = art$index, similarity = art$Similarity),
    author_author = list(author = author_list[1], index = auth$index,
                         similarity = auth$Similarity),
    mds_articles = list(index = mds$index, X = mds$X, Y = mds$Y),
    mds_authors = list(index = amds$index, X = amds$X, Y = amds$Y)
  ),
  file.path(out_dir, "search.json"), auto_unbox = TRUE, pretty = FALSE, digits = NA
)
cat("Reference outputs written to", out_dir, "\n")
