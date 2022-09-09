##   Some useful functions


## subset

# subset(x, subset, select, ...)

str(iris)
iris.ver <- subset(iris, Species == 'versicolor')
str(iris.ver, vec.len = 2)

iris.ver2 <- subset(iris, Species == 'versicolor', 
                    select = c(Sepal.Length,Sepal.Width))
str(iris.ver2, vec.len = 2)


## with

## with(data,expr,...)

with(iris, Sepal.Length[Petal.Length < 1.5
                        & Petal.Width >= 0.4])
# gives the same result as 
iris$Sepal.Length[iris$Petal.Length < 1.5
                  & iris$Petal.Width >= 0.4]

with(iris, boxplot(Sepal.Length ~ Species, 
                   col = 'cyan3'))

example.df <- data.frame(a = 1:5, b=-5:-1) 
with(example.df, {a+b; 2*a})


## within

with(example.df, a + b) 
with(example.df, {c = a + b}) 
with(example.df, {c = a + b; print(c)})

with(example.df, {c = a + b; example.df})
within(example.df, {c = a + b; example.df})
example.df
(example.df <- within(example.df, {c = a + b}))


## aggregate

# aggregate(x,by,FUN,...)

library(MASS)
str(crabs)
aggregate(crabs[, 4:8], list(sp=crabs$sp,
                             sex=crabs$sex), median)


## \texttt{order}

(x <- c(14:16,12,11,13,17))
order(x)
x[order(x)]

y <- 17:11
z <- 11:17
rbind(x,y,z)[,order(x)]


## apply

# apply(data, dim, function, ...)

str(iris3)
apply(iris3, 2, mean)
apply(iris3, c(2,3), mean)
apply(iris3, c(2,3), mean, trim=0.1)


## sweep

(iris.means <- apply(iris3, c(2,3), mean))
iris.ctd <- sweep(iris3, c(2,3), iris.means,'-')

head(iris.ctd)


## tapply

# tapply(x, index, FUN = NULL, ...)

str(mtcars)
tapply(mtcars$mpg, mtcars$am, mean)
tapply(mtcars$mpg, list(mtcars$am,mtcars$cyl), mean)

## outer

(aa <- 1:4)
(bb <- c(2,4,6))
(ab <- aa %o% bb)

(A  <- matrix(1:12, ncol=3))
(B <- 9:10)
(AB <- A %o% B)

# outer( array1, array2, FUN='*')

x <- y <- seq(-3,3,length=51)
f <- function(x,y){
  exp(-(x^2+y^2)/2)/(2*pi)
  }
z <- outer(x,y,f)
str(z)

persp(x,y,z,phi=20, theta=30,expand = 0.8,col='lightblue')

outer(1:9,1:9)


## split

library(lattice)
str(barley)
barley.lst <- split(barley$yield, barley$site)
str(barley.lst)
barley.lst[[2]]


## expand.grid

expand.grid(age=c(20,40),weight=c(50,70),
            sex=c('Male','Female'))


## Summary statistics
str(cars)
summary(cars)
var(cars)
cov(cars)
cor(cars)

