#define sigmoid function
sig <- function(x){
  a = 1/(1+exp(-x))
  return(a)
}

#define Sigmoid Gradient function
sigrad <- function(x){
  sg = sig(x)*(1-sig(x))
  return(sg)
}

#Randomly Initialize Weights
randinit <- function(l_in, l_out){
  w= matrix(0, nrow = l_out, ncol= (1+l_in))
  epsilon_init = .12
  w= (matrix(runif(l_out*(1+l_in)),nrow =l_out, ncol=(1 + l_in )) * 2 * epsilon_init) - epsilon_init
  return(w)
}


#debug initialize weights
debug <- function(f_in, f_out){
  w = matrix(0, nrow = f_out, ncol= (1+ f_in))
  w = matrix(sin(1:length(w)),nrow = nrow(w), ncol = (ncol(w)))/10
  return(w)
}

