#  | STAT 210
#  | Applied Statistics and Data Analysis
#  | Linear Regression II: 
#  | Matrix Formulation and Tests


library(MASS) 
attach(crabs)
lm1 <- lm(FL~CL)
detach(crabs)
## Back to Example 1 
summary(lm1)$cov.unscaled
vcov(lm1)
summary(lm1)$sigma
(summary(lm1)$sigma^2)*summary(lm1)$cov.unscaled


# Hypothesis tests
## Back to Example 1
summary(lm1)

## Confidence Intervals
## Example 2: GPA and SAT
library(PASWR) 
attach(Grades)
plot(sat, gpa)

## Parameters for the regression line
y <- gpa
x <- sat
b1 <- sum((x-mean(x))*(y-mean(y)))/sum((x-mean(x))^2)
b0 <- mean(y)-b1*mean(x)
c(b0, b1)

## Normal equations in matrix form
X <- cbind(rep(1,200), x)
Y <- matrix(y, ncol=1)
betahat <- solve(t(X)%*%X)%*%t(X)%*%Y
t(betahat)

## lm function
model <- lm(y~x)
model$coefficients

## Graph
plot(sat, gpa); abline(model, col='red')

## Covariance matrix for the estimated parameters
## MSE(X'X)^{-1}
XTX <- t(X)%*%X
solve(XTX)

(XTXb <- summary(model)$cov.unscaled)

## MSE
sum(resid(model)^2)/(200-2)

(mse <- summary(model)$sigma^2)

## Covariance matrix
(beta.cov <- mse * XTXb)

vcov(model)

## Test of hypothesis
qt(0.995,198)

model$coefficients[2]/sqrt(vcov(model)[2,2])

2*(1-pt(15.912,198) )


summary(model)$coef

## Confidence intervals for parameters
confint(model, level=.99)



