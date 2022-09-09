# V3
# Basic Functions

### c

# The concatenation function c combines elements 
# to create a vector.  The elements are listed 
# in order and separated by a comma:

(x <-c(1,1.5,2,2.5)) 


(x <- c(x,3))

(y <- c('this','is','an','example'))

(z <- c(x,'a'))

cat(z)


### seq
# This function is used to form regular sequences of numbers. 
# The basic syntax is 

seq(from=1, to=1, by=((to-from)/length.out-1),
    length.out = NULL)


seq(0,100,5)
seq(1955,1966,1)
seq(10,12,0.2)


seq(1955,1966)
seq(5)
seq(5,-1)

1:5
5:1
50:60/5

### rep
# This function replicates a pattern. The syntax is 
rep(x,times)
# where x is the object to be replicated and times is the number 
# of replications:

rep(10,3)
rep(c(0,5), 4)
rep(1:5,2)
rep(c('wx','yz'),3)
rep(c('wx','yz'), each = 3)

rep(c(10,20),c(2,4))
rep(1:3,1:3)
rep(1:3,rep(4,3))

rep(c(1,2,3,4), length=10)


## Indexing

# To get a particular component from a vector, 
# you can write the position of the element in 
# the vector inside square brackets after 
# the vector's name, as in the following example:

z <- 1:10
z[3]



# 1.- Positive integers 

letters[19]
letters[c(11,13,15)]
letters[seq(11,15,2)]

# 2.- Negative numbers 

letters[-(11:26)]
letters[-seq(1,25,2)]

# 3.- Logical variables. TRUE values will be 
# included while FALSE values will not.

a <- 1:26
a < 11
letters[a<11]
a %% 2 == 0
letters[a %% 2==0]

# Double inequalities such as $1 < a \leq 7$ should be written 
# using the \& symbol: 1 < a \& a <= 7:

letters[1 < a & a <= 7]




# If the indexing vector is shorter than the object 
# to which it is being applied, it is recycled:

letters[c(T,F)]

# This gives the letters that occupy 
# an odd position in the vector.

### which

# This function gives a list of the positions 
# within an object occupied by entries that 
# satisfy a certain condition. 

str(trees)
which(trees$Volume > 50)
trees[which(trees$Volume > 50),]
trees[trees$Volume > 50,]


# The functions which.max and  which.min that give the position 
# of the (first) maximum and (first) minimum values of an vector:

(b <- rep(1:3,2))
which.max(b); which.min(b)



## Sampling

# The sample function generates random samples from a given set. 
# The syntax is 
###
# sample(x, size, replace = FALSE, prob = NULL)
###
# where x is the set from which we want to obtain 
# the sample, size is the size of the sample, 
# replace indicates if repetitions are allowed 
# or not, and prob is a probability vector if we 
# want to get a sample with a non-uniform 
# distribution. 

xy <- c('bad','regular','good')
sample(xy,10,replace=T)
pp <- c(0.1,0.1,0.8)
sample(xy,10,replace=T,prob=pp)

(rnorm(10))
(rbinom(5, 20, 0.5))
(rexp(8))
(rpois(4, lambda=10))


pnorm(1.96)
qnorm(0.975)

qnorm(c(0.025,0.975))

points.x <- seq(-3,3,length=100)
points.den <- dnorm(points.x)
points.fd <- pnorm(points.x)
plot(points.x, points.fd,type='l',xlab='Values',
     ylab='', main='Normal Distribution',col='cyan')
lines(points.x,points.den,col='darkblue')
legend('topleft',c('Dist. Fn','Density'),col=c('cyan','skyblue'),lty=rep(1,2))


## Vectorized Operations
# R performs vector operations componentwise: 
# if we add two vectors of equal length, the 
# result is another vector of the same length, 
# whose components are the sum of the components 
# of the vectors we add. This is also true for 
# any other arithmetic operation, including powers. 

(a <- 5:2)
(b <- (1:4)*2)
a + b
a - b
a * b
a / b
a^b


log(a)

(x <- 10:6)
(y <- seq(1,9,2))

(z <- log((x^2 + 2*y) / (x + y)^2 ))




## NAs and Infs
# Missing data in R are denoted by NA (not 
# available). When we do an operation with an NA 
# the result will be an NA. 

(y <- c(1:5,NA))
2*y
max(y)

(x <- 2/0)
exp(x)
exp(-x)
x - x



