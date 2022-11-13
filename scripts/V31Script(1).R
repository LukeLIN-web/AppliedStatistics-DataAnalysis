#  | STAT 210
#| Applied Statistics and Data Analysis
#| Linear Regression V: 
#| Model Assessment

# Model Assessment
str(anscombe)
summary(anscombe)

ff <- y ~ x
mods <- setNames(as.list(1:4), paste0("lm", 1:4))
for(i in 1:4) {
  ff[2:3] <- lapply(paste0(c("y","x"), i), as.name)
  ## or   ff[[2]] <- as.name(paste0("y", i))
  ##      ff[[3]] <- as.name(paste0("x", i))
  mods[[i]] <- lmi <- lm(ff, data = anscombe)
  print(anova(lmi))
}

## See how close they are (numerically!)
sapply(mods, coef)
lapply(mods, function(fm) coef(summary(fm)))



op <- par(no.readonly = TRUE) 
par(mfrow = c(2, 2), mar = 0.1+c(3,3,1,1), 
          oma =  c(0, 0, 2, 0))
for(i in 1:4) {
  ff[2:3] <- lapply(paste0(c("y","x"), i), as.name)
  plot(ff, data = anscombe, col = "red", pch = 21, bg = "orange", cex = 1.2,
       xlim = c(3, 19), ylim = c(3, 13))
  abline(mods[[i]], col = "blue")
}
mtext("Anscombe's 4 regression data sets", outer = TRUE, cex = 1.2)
par(op)









# Facts About Residuals
## Facts About Residuals


# Homoscedasticity and Linearity
## Homoscedasticity and Linearity

set.seed(456)
xx <- runif(25,10,20)
zz <- rnorm(25)
y1 <- 2 + 0.3*xx + zz
plot(xx, y1, pch=16, xlab='x', ylab='y')
modelA <- lm(y1 ~ xx)
abline(modelA)

summary(modelA)

plot(fitted(modelA), resid(modelA), pch=16, 
     xlab='Fitted values', ylab='Residuals')
abline(h=0, col='red')


y2 <- 2 + 2.3*xx+((xx-10))*zz; plot(xx,y2,pch=16)
modelB <- lm(y2~xx); abline(modelB)

summary(modelB)

plot(fitted(modelB), resid(modelB), pch=16,
     xlab='Fitted values', ylab='Residuals')
abline(h=0, col='red')


z3 <- rnorm(25); y3 <- 2 + 2.3*xx + ((xx-20))*z3
plot(xx,y3,pch=16); modelC <- lm(y3~xx)
abline(modelC)

summary(modelC)
plot(fitted(modelC), resid(modelC), pch=16,
     xlab='Fitted values', ylab='Residuals')
abline(h=0, col='red')


##**Example Q**
  
set.seed(4567) 
xx <- runif(25,10,20)
zz <- rnorm(25,sd=4) 
y4 <-  2 + 1.3*xx+ 3*(xx-10)^2+zz
modelD <- lm(y4~xx)
summary(modelD)

plot(modelD, which = 1)

plot(xx,y4,pch=16)

abline(modelD)


modelE <- lm(y4~xx+I(xx^2))
summary(modelE)
plot(xx,y4,pch=16)
abline(modelD, col='blue', lwd=2)
curve(309.99-59.57*x+3.02*x^2,10,20, add=T,
      col='red', lwd=2)
plot(modelE, which = 1)


# Gaussianity
## Gaussianity

qqnorm(rstandard(modelA))
qqline(rstandard(modelA))

qqnorm(rstandard(modelB)) 
qqline(rstandard(modelB))

qqnorm(rstandard(modelC)) 
qqline(rstandard(modelC))

## Gaussianity: Example Q

plot(modelD, which = 2)

plot(modelE, which = 2)

## Diagnostic Plots

plot(modelA)

plot(modelB)

op <- par(no.readonly = TRUE);
par(mfrow=c(2,2), mar=c(4,4,3,2))
plot(modelA)
par(mfrow=c(1,1))

par(mfrow=c(2,2), mar=c(4,4,3,2))
plot(modelB)
par(op)


