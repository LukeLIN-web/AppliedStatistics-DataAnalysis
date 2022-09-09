##   Objects and Data II: Matrices and Arrays


# Matrices
# matrix(data, nrow, ncol, byrow=F, dimnames = NULL)

matrix(1:6)
matrix(1:6, nrow=3)
matrix(1:6, nrow=3, byrow=T)

(A <-  matrix(1:6, nrow=3, byrow=T))
(B <-  matrix(seq(0, 10, 2), 3, 2))

A+B
A-B

A-5
B/2

(D <- matrix ((1:8)*3,2))
A %*% D

A*B

## Other Functions
# t
(XX <- matrix(c(2,3,4,1,5,3),ncol=3))
t(XX)
XX %*% t(XX)
t(XX) %*% XX

# diag
diag(XX)
diag(t(XX))
diag(t(XX) %*% XX)
diag(1:4)
diag(3)

## det and solve
(YY <- matrix(c(12,3,8,16,21,5,7,9,12,18,
                4,3,19,5,21,8), ncol=4))
det(YY)

## Inverse
round(solve(YY),3)
YY%*%solve(YY)
round(YY%*%solve(YY),15)
round(solve(YY)%*%YY,15)

# System of equations
b <- 1:4
(x <- solve(YY,b))
YY %*% x

# dimnames
dimnames(B)
dim(B)
subjects <- c('Patient1','Patient2','Patient3')
variables <- c('Var1', 'Var2')
dimnames(B) <- list(subjects,variables)

dimnames(B)
B

## Extracting values
A[1,2]
A[1,]
YY[2:3,1:2]

B['Patient1','Var2']
B['Patient1',]
B[,'Var2']


## Adding rows or columns
A
(A <-cbind(A,c(3,5,7)))

B
cbind(B,c(3,5,7))
cbind(B,Var3=c(3,5,7))
rbind(B,Patient4 = c(6,12))


# Arrays
(x <- array (1:24, c(3,4,2)))
(x1 <- 1:24)
dim(x1) <- c(3,4,2)
x1

x[,2,]
x[,3,1]
x[,,1]

## Arrays: `aperm`
# aperm(array, perm, resize=TRUE)

str(iris3)
iris3b <- aperm(iris3, c(2,3,1))
str(iris3b)



  
  