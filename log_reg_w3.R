
#load data 
dat <- read.table("/users/brenden.leavitt/documents/ng/logistic_regression_w3/ex2/ex2data1.txt", sep = ",")

X = dat[,c(1,2)]
y = matrix(dat[,3])
m = nrow(X)
n = ncol(X)
X = as.matrix(cbind(rep(1, n),X))



sigmoid <- function(z){
  f = 1/(1+exp(-z))
  return(f)  
}

# initialize theta
initial_theta = matrix(rep(0,n+1),n+1, 1)

#function to compute gradient and cost

computecost <- function(theta,X, y){
  m =length(y)
  J = 0
  grad = as.matrix(rep(0), length(theta))
  h_theta = sigmoid(X%*% theta)
  J = (1/m) * ((t(-y) %*% log(h_theta) - t(1-y)%*%log(1- h_theta)))
  grad  = (1/m) * (t(h_theta - y) %*% X)
  return(J)
}
computegrad <- function(theta,X, y){
  m =length(y)
  J = 0
  grad = as.matrix(rep(0), length(theta))
  h_theta = sigmoid(X%*% theta)
  J = (1/m) * ((t(-y) %*% log(h_theta) - t(1-y)%*%log(1- h_theta)))
  grad  = (1/m) * (t(h_theta - y) %*% X)
  return(grad)
}


test_theta <- matrix(c(-24,.2,.2),3,1)  
computecost(initial_theta, X, y)
computegrad(initial_theta, X,y)
computecost(test_theta, X, y)
computegrad(test_theta, X, y)

head(dat)
colnames(dat) <- c('x', 'x2', 'y')
lg <- glm(y~ x + x2, data = dat, family= binomial)
summary(lg)
