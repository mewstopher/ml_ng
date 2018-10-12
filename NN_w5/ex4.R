library(R.matlab)
library(numDeriv)
library(lbfgsb3)
path = ""
source(paste(path,"/r/nn_w5/nn_tools.R",sep = ""))
source(paste(path,"/r/nn_w5/lbfgsb3.R", sep=""))
source(paste(path,"/r/nn_w5/cost_grad_functions.R",sep=""))
source(paste(path,"/r/nn_w5/predict.R",sep=""))
source(paste(path,"/r/nn_w5/display_data.R",sep=""))
# Import data

ds_data <- readMat(paste(path,"/nn_w5/ex4/ex4data1.mat", sep="")) 

X <- ds_data$X
y <- ds_data$y

m = nrow(X)


weights <- readMat(paste(path,"/nn_w5/ex4/ex4weights.mat",sep="")) 
#theta1- 25x401
#theta1- 10x26
#Theta1 <- matrix(unlist(weights[1]), ncol= 401)
#Theta2 <- matrix(unlist(weights[2]), ncol = 26)


input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

Theta1_grad = matrix(0, ncol = input_layer_size+1, nrow = hidden_layer_size)
Theta2_grad = matrix(0, ncol= hidden_layer_size+1, nrow = num_labels)

#unroll Parameters - these parameters are already trained
#nn_params = c(Theta1, Theta2)

init_theta1 = randinit(input_layer_size,hidden_layer_size)  
init_theta2 = randinit(hidden_layer_size, num_labels)
initial_nn_params = c(init_theta1, init_theta2)
lambda = 0

nn_params= initial_nn_params

lambda = 3

y_mat = matrix(0, nrow=10, ncol = 5000)
for (i in 1:5000){
  for (j in 1:10){
    if(y[i,1] == j){
      y_mat[j,i] = 1
    }
  }
}
y_matrix = t(y_mat)

#rename functions
sigmoidGradient <- sigrad

lambda = 1
costF <- Costfunction(input_layer_size, hidden_layer_size, num_labels, X, y,lambda)
costG <- Gradfunction(input_layer_size, hidden_layer_size, num_labels, X, y,lambda)


opt <- lbfgsb3_(initial_nn_params, fn= costF, gr=costG,
                control = list(trace=1,maxit=200))

nn_params <- opt$prm
Theta1 <- matrix(nn_params[1: (hidden_layer_size * (input_layer_size + 1))], 
                 nrow = hidden_layer_size,
                 ncol = input_layer_size + 1)

Theta2 <- matrix(nn_params[((hidden_layer_size * (input_layer_size + 1))+1) : length(nn_params)],
                 nrow= num_labels,
                 ncol= hidden_layer_size + 1)

pred <- predict(Theta1,Theta2,X)
mean(pred==y) * 100



rp <- sample(m)

for (i in 1:m){
  # Display
  cat(sprintf('\nDisplaying Example Image. Press Esc to End\n'))
  displayData(X[rp[i], ])
  
  pred <- predict(Theta1, Theta2, X[rp[i],])
  cat(sprintf('\nNeural Network Prediction: %d (y %d) (digit %d)\n', pred  , y[rp[i]]  ,pred %% 10))
  
  # line <- readLines(con = stdin(),1)
  #cat(sprintf('Program paused. Press enter to continue.\n')
  #line <- readLines(con = stdin(),1)
  Sys.sleep(2)
  #press esc to quit the loop in Rstudio
}


