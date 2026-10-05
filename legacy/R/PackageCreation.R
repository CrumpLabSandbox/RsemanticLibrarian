library(usethis)
create_package("../pname")
use_description_defaults()
use_gpl3_license()

use_roxygen_md()
use_readme_md()

pkgdown::build_site()
pkgdown::build_articles()
pkgdown::clean_site()
pkgdown::build_news()

use_news_md()

#function creation
use_r("sl_semantic_space")

use_rcpp(name = "rowSumsSq")

use_package("lsa", "Suggests")
usethis::use_package("tidyr")

usethis::use_pipe()

use_package_doc()
use_namespace()

use_vignette("Semantic_Librarian_Tutorial")



### create sample data

article_df <- article_df[1:100,]
article_vectors <- article_vectors[1:100,]
WordVectors <- WordVectors[1:100,]
author_list <- author_list[1:100]
dictionary_words <- dictionary_words[1:100]
AuthorVectors <- AuthorVectors[1:100,]

use_data(article_df,article_vectors,WordVectors,author_list,dictionary_words)
use_data(AuthorVectors)

use_logo("xtra/sl_hex.png")

library(hexSticker)
imgurl <- "xtra/sl.png"
sticker(imgurl, package="RsemanticLibrarian", p_size=3.5, s_x=1, s_y=1, s_width=1,
        l_alpha = .6,
        p_color ="#635860",
        h_color = "#635860",
        h_fill = "#ffffff",
        #spotlight = TRUE,
        #l_x = 1,
        #l_y = 1.4,
        #l_height = 6,
        #l_width = 6,
        url = "crumplab.github.io/RsemanticLibrarian",
        u_x = 1, u_y = 0.08, u_color = "#635860",
        u_family = "Aller_Rg", u_size = 1.2, u_angle = 30,
        white_around_sticker = TRUE,
        filename="xtra/rsl.png",
        dpi=600)
