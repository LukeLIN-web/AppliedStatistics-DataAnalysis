### Objects and Data I

c(T,F,12.3)
c(T,8.3,12+3i)
c(F,12.3,'hi')

mode(letters); length(letters)

(x <- 1:10)
dim(x) <- c(2,5)
x

# Factors
gender <- factor(c('male', 'female','female',
                   'male', 'female' ))
gender

attach(iris)
is.factor(Species)
is.factor(Sepal.Length)
levels(Species)
nlevels(Species)
length(levels(Species))
detach(iris)

opinion <- c(1,2,2,2,1,0,4,4)
fopinion <- factor(opinion,levels=0:4)
levels(fopinion) <- c('awful','bad','regular',
                      'good', 'excellent')

opinion
fopinion

levels(fopinion)
as.numeric(fopinion)

test1 <- factor(c(0,1,2,3,4,5))
test2 <- factor(c(0,1,2,3,5))
levels(test1)
levels(test2)

# Ordered Factors
(xx <- sample(c('high','medium','low'),10,replace=T))
(yy <- factor(xx))
(zz <- ordered(xx))

class(xx)
class(yy)
class(zz)

is.factor(xx)
is.factor(yy)
is.factor(zz)


levels(zz)
zz
levels(zz) <- c('low','medium','high')
zz

xx
(ww <- ordered(xx,levels=c('low','medium','high')))


# Tables
(fact1 <- sample(c('f','m'),10,replace=T))
table(fact1)

table(rpois(100,3))

with(mtcars, table(cyl, gear))
with(mtcars, table(cyl, gear,am))
with(mtcars, ftable(cyl, gear,am))













