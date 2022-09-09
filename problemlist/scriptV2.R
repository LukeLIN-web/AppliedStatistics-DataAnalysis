## --------------------------
getwd()

ls()

sd

data()
data(cars)
plot(cars)
head(cars)
str(cars)
var
rm(list=ls())
help(log)

?lm

?'*'

??truehist

x <- c(14:16,12,11,13,17)

order(x)

x[order(x)]

dim(x)

apply(iris3, 2, mean) # 用不同维度显示数据.

## --------------------------
# problem list 1 
rep(10:1,c(5:1,1:5)) # 每个的个数不同

seq(100,102,length.out = 10)

1:5 + rep(0:4 , each = 5)

# exercise 2 
# 就是落在圆内的点数量

x <- runif(10000,-1,1)
y <- runif(10000,-1,1)
z <- x^2 + y ^2
asum <- sum(as.numeric(z<1))
(piest4 <- 4*asum/10000)

(error4 = abs(pi-piest4) )


str(mtcars)


# exercise 4

pp = c(0.1,0.2)
sample(samp1,100,replace =T, prob = pp)


table(fact1)


