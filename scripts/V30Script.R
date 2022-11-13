#  | STAT 210
#| Applied Statistics and Data Analysis
#| Linear Regression IV: 
#| Coefficient of Determination

library(MASS); attach(crabs)
lm1 <- lm(FL~CL)
lm2 <- lm(FL[sp=='B'] ~ CL[sp=='B'],data = crabs)
lm3 <- lm(FL[sp=='O'] ~ CL[sp=='O'],data = crabs)
lm4 <- lm(Volume ~ Height, data=trees)
lm5 <- lm(Volume ~ Girth, data = trees)
detach(crabs)



# Coefficient of Determination
## Coefficient of Determination

## Example 1: Crabs, one model
summary(lm1)

## Example 2: Crabs, two models, sp = B
summary(lm2)

## Example 2: Crabs, two models, sp = O
summary(lm3)

## Example 3: trees, Volume ~ Height
summary(lm4)

## Example 3: trees, Volume ~ Girth
summary(lm5)



## Relation with the Correlation Coefficient

set.seed(98765)
xx <- runif(15,10,20)
zz <- rnorm(15)
(r.sq <-round(summary(lm(zz~xx))$r.squared,3))
plot(xx,zz,pch=16, xlab='x', ylab='y')
abline(lm(zz~xx))
title(main= bquote(R^2 == .(r.sq)))
xx.new <- data.frame(xx=seq(10,20, length.out = 15))
pc <- predict(lm(zz~xx),xx.new, int='c')
matlines(xx.new$xx, pc, lty=c(1,2,2), col=c('black','red','red'))
pp <- predict(lm(zz~xx),xx.new, int='p')
matlines(xx.new$xx, pp, lty=c(1,2,2), col=c('black','red','red'))



yy1 <- 2 + 0.3*xx+zz
plot(xx,yy1,pch=16, xlab='x', ylab='y')
abline(lm(yy1~xx))
(r.sq <-round(summary(lm(yy1~xx))$r.squared,3))
title(main= bquote(R^2 == .(r.sq)))
pc <- predict(lm(yy1~xx),xx.new, int='c')
matlines(xx.new$xx, pc, lty=c(1,2,2), col=c('black','red','red'))
pp <- predict(lm(yy1~xx),xx.new, int='p')
matlines(xx.new$xx, pp, lty=c(1,2,2), col=c('black','red','red'))


yy2 <- 2 + 0.3*xx + zz/10
plot(xx,yy2,pch=16, xlab='x', ylab='y')
abline(lm(yy2~xx))
(r.sq <-round(summary(lm(yy2~xx))$r.squared,3))
title(main= bquote(R^2 == .(r.sq)))
pc <- predict(lm(yy2~xx),xx.new, int='c')
matlines(xx.new$xx, pc, lty=c(1,2,2), col=c('black','red','red'))
pp <- predict(lm(yy2~xx),xx.new, int='p')
matlines(xx.new$xx, pp, lty=c(1,2,2), col=c('black','red','red'))






