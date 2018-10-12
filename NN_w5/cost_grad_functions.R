Costfunction = function(input_layer_size, hidden_layer_size, num_labels, X, y,lambda){
    function(nn_params){
      Theta1 <- matrix(nn_params[1: (hidden_layer_size * (input_layer_size + 1))], 
                       nrow = hidden_layer_size,
                       ncol = input_layer_size + 1)
      
      Theta2 <- matrix(nn_params[((hidden_layer_size * (input_layer_size + 1))+1) : length(nn_params)],
                       nrow= num_labels,
                       ncol= hidden_layer_size + 1)
      
      m <- dim(X)[1] 
      a1= cbind(rep(1,nrow(X)), X)
      z2 = a1%*% t(Theta1)
      a2 = 1/(1+exp(-z2))
      a2 = cbind(1, a2)
      z3 = (a2)%*% t(Theta2)
      h_theta = 1/(1+exp(-z3))
    
     #Good to here
      
      
  #    j = 0
  #    for (i in 1:10){
  #      for (k in 1:5000){
  #        J = sum(-y_matrix[k,i]* log(h_theta[k,i]) - (1-y_matrix[k,i])* log(1- h_theta[k,i]))
  #      }
  #    }
      J <- - sum(y_matrix * log(h_theta)) - sum((1 - y_matrix) * log(1 - h_theta))
      J = J/m
      Theta1_b <- Theta1[ ,2: ncol(Theta1)]
      Theta2_b <- Theta2[ ,2: ncol(Theta2)]
      
      t1 <- sum(Theta1_b ^2)
      t2 <- sum(Theta2_b^2)
      
#      t1 = sum(Theta1[, -1])^2
#      t2 = sum(Theta2[,-1])^2
#      reg = (lambda/(2*m)) * (t1 + t2)
      J = J + (lambda/(2 * m)) * (t1 + t2) 
      return(J)
  }
}



Gradfunction = function( input_layer_size, hidden_layer_size, num_labels, X, y,lambda){
  function(nn_params){
    Theta1 <- matrix(nn_params[1: (hidden_layer_size * (input_layer_size + 1))], 
                     nrow = hidden_layer_size,
                     ncol = input_layer_size + 1)
    
    Theta2 <- matrix(nn_params[((hidden_layer_size * (input_layer_size + 1))+1) : length(nn_params)],
                     nrow= num_labels,
                     ncol= hidden_layer_size + 1)
    
    #backprop
    
    a1 = cbind(rep(1,nrow(X)), X)
    z2 = Theta1%*%t(a1)
    a2 = sig(z2)
    a2 = rbind(1,a2)
    z3 = Theta2%*%a2
    a3 = sig(z3)
    
    delta_3 = a3 - y_mat
    z2 = rbind(1,z2)
    delta_2 = (t(Theta2)%*%delta_3) * sigrad(z2)
    delta_2 = delta_2[-1,]
    
    Theta1_grad = Theta1_grad + delta_2 %*% a1
    Theta1_grad = (1/m) * Theta1_grad

    Theta2_grad = Theta2_grad + delta_3 %*% t(a2)
    Theta2_grad = (1/m) * Theta2_grad
    
    Theta1_grad <- Theta1_grad + lambda / m * cbind(rep(0,dim(Theta1)[1]), Theta1[,-1])
    Theta2_grad <- Theta2_grad + lambda / m * cbind(rep(0,dim(Theta2)[1]), Theta2[,-1])
    
    grad = c(as.vector(Theta1_grad), as.vector(Theta2_grad))
    
    return(grad)
  }    
}

