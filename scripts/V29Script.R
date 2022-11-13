#  | STAT 210
#| Applied Statistics and Data Analysis
#| Linear Regression III: 
#  | Confidence Bands and Anova


# Confidence Band for the Regression Line
## Confidence Band for the Regression Line

library(PASWR); 
attach(Grades)
y <- gpa; x <- sat
model <- lm(y~x)

new.data <- data.frame(x=c(900,1100,1300))
predict(model,new.data,interval='c')

plot(sat, gpa, pch = 16, col = 'gray80')
modelA <- lm(gpa~sat, data = Grades)
abline(modelA)
new.sat <- data.frame(sat=seq(700,1600, length.out = 15))
pc <- predict(modelA,new.sat, int='c')
matlines(new.sat$sat, pc, lty=c(1,2,2), lwd=rep(2,3),
         col=c('black','red','red'))

new.data <- data.frame(x=c(900,1100,1300))
predict(model,new.data,interval='p')
predict(model,new.data,interval='c')


plot(sat, gpa, pch = 16, col = 'gray80')
modelA <- lm(gpa~sat, data = Grades)
abline(modelA)
new.sat <- data.frame(sat=seq(700,1600, length.out = 15))
pc <- predict(modelA,new.sat, int='c')
matlines(new.sat$sat, pc, lty=c(1,2,2),lwd=rep(2,3), 
         col=c('black','red','red'))
pp <- predict(modelA,new.sat, int='p')
matlines(new.sat$sat, pp, lty=c(1,2,2),lwd=rep(2,3),
         col=c('black','blue','blue'))
detach(Grades)

# Analysis of Variance in Linear Regression
## Analysis of Variance in Linear Regression

library(MASS); attach(crabs); 
lm1 <- lm(FL~CL)

anova(lm1)

str(trees)
plot(Volume ~ Height, data=trees)
lm4 <- lm(Volume ~ Height, data=trees)
abline(lm4)

summary(lm4)
anova(lm4)


## Diagnostic plots
  
plot(Volume ~ Height, data=trees)
abline(lm4)
abline(h=mean(trees$Volume), col='red')
segments(trees$Height,fitted(lm4),
           trees$Height,trees$Volume, col='blue')


plot(fitted(lm4),resid(lm4))
abline(h=0, lty='dotted')

plot(lm4)


plot(lm4, which = 2)

  
## Example 3: Another model
  
plot(Volume~Girth, data=trees)
lm5 <- lm(Volume ~Girth, data = trees)
abline(lm5)

summary(lm5)
plot(lm5)


plot(lm5, which = 2)
plot(lm5, which = 3)
plot(lm5, which = 4)

  
  
  
  
  
  
  
  
  
  
  
  
  
  