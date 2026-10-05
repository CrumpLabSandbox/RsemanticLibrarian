#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]

NumericVector rowSumsSq(NumericMatrix x) {
  int nrow = x.nrow(), ncol = x.ncol();
  NumericVector out(nrow);
  
  for (int j = 0; j < ncol; ++j) {
    for (int i = 0; i < nrow; ++i) {
      out[i] += std::pow(x(i, j), 2);
    }
  }
  
  return out;
}