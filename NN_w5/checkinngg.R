#compute numerical Gradent

cnumgrad <- function(j, theta){
  input_layer_size = 3
  hidden_layer_size =5
  num_labels = 3
  m = 5
  
  Theta1 = debug(hidden_layer_size, input_layer_size)
  Theta2 = debug(num_labels, hidden_layer_size)
  nn_params = c(Theta1, Theta2)
  g = grad(j(theta, input_layer_size, hidden_layer_size, num_labels,X,y,3))
  
}
lambda = 3

checking_grad <- (lambda){
  input_layer_size = 3
  hidden_layer_size =5
  num_labels = 3
  m = 5
  
  Theta1 = debug(hidden_layer_size, input_layer_size)
  Theta2 = debug(num_labels, hidden_layer_size)
  
  X = debug(m, (input_layer_size - 1))
  y = 1 + t((1:m)%%num_labels)
  
  nn_params = c(Theta1, Theta2)
  
  j_g <- nncost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
  
  return(j_g)
  
}
