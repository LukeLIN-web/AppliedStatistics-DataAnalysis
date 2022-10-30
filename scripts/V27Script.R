### | Linear Regression I: Introduction   


## Linear Regression

library(MASS) 
attach(crabs)
plot(CL,FL, pch=20, xlab='Carapace length', ylab='Frontal lobe size')

lm1 <- lm(FL~CL)
abline(lm1, lw=2, col='blue')

plot(CL,FL, pch=20, xlab='Carapace length', ylab='Frontal lobe size', col=sp)
lm1 <- lm(FL~CL)
abline(lm1, lw=2, col='blue')

## Estimation
## Back to Example 1
summary(lm1)

names(lm1)
str(lm1)

lm1$coefficients
coef(lm1)

str(summary(lm1))








