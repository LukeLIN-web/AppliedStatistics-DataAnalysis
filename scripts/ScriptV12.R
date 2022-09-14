# One-sample problems

## Data 
library(MASS)
data(michelson)
str(michelson, vec.len = 2)

# We extract the values for the first experiment.
(mich.exp1 <- michelson[michelson$Expt == 1,1])

# The summary values for the data set are 
summary(mich.exp1)


# Data Analysis
attach(michelson)
mich.mean <- tapply(Speed, Expt, mean)
plot(Speed, col=Expt, pch=16, ylim = c(600, 1100))
points(c(10,30,50,70,90),mich.mean,pch=17, 
       cex=1.7, col=1:5)
abline(h=range(Speed))
abline(h=range(mich.mean), col = 2)

curve(dnorm(x), -3,3,lwd=2, ylab = 'density', col = 4)
abline(h=0); abline(v=0)

curve(dnorm(x,1,1),-3,5, lwd=2, ylim=c(0,.9), 
      ylab='Density')
curve(dnorm(x,1,2),-3,5, lwd=2, col='blue', add = T)
curve(dnorm(x,1,0.5),-3,5, lwd=2, col='red', add = T)
abline(v=0); abline(h=0)
legend('topright',
       legend = c('N(1,1)','N(1,2)','N(1,0.5)'), 
       col=c(1,3,2), 
       lwd=c(2,2,2))

var(mich.exp1); sd(mich.exp1)


points.x <- seq(-3,3,length.out = 101)
points.y <- dnorm(points.x)
plot(points.x,points.y,type='l',xlab='',
     main='Gaussian Density',
     ylab='', axes=FALSE)
abline(h=0);  abline(v=0)
xv <- points.x[points.x <= -1.74]
xv <-c(xv, -1.74, -3)
yv <- points.y[points.x <= -1.74]
yv <- c(yv,yv[1],yv[1])
polygon(xv,yv, col='red')
xw <- points.x[points.x >= 1.74]
xw <-c(xw, 1.74)
yw <- points.y[points.x >= 1.74]
yw <- c(yw,points.y[101])
polygon(xw,yw, col='red')
axis(1,at=c(-1.74,1.74),labels = c('-3.452','3.452'),
     line=0,pos=0,lwd=2)

2*pnorm(-3.452)


truehist(mich.exp1, col = 'azure2')
curve(dnorm(x,mean = mean(mich.exp1),sd = sd(mich.exp1)),
      600, 1100, add = TRUE, lwd=2, col='brown3')
lines(density(mich.exp1),col='blue3',lwd=2)


qqnorm(mich.exp1)
qqline(mich.exp1,lwd=2)

shapiro.test(mich.exp1)

