##  | Graphics in R
## | High-level commands


## curve

## curve(expr, from, to, add = FALSE, ...)
curve(x^3 - 3*x, -2,2,lwd=2, col='darkblue')
curve(x^2 -2 , add= TRUE, col='red3', lwd=3, lty = 3)

## Boxplots
boxplot(iris[,1:4], col='lightblue')
boxplot(Sepal.Length~Species, data=iris, notch = T,
        col='lightblue')


## Barplots

tN <- table(Ni <- rpois(100, lambda=5))
barplot(tN)
barplot(tN, col=rainbow(20))


## Histograms
str(morley)
attach(morley)
plot(Speed,col=Expt, pch=Expt)

hist(Speed)

hist(Speed, breaks = 20, probability = T, 
     col = 'azure2',xlab='Speed', ylab='Freq',
     main = "Michelson's Experiment")

op <- par(no.readonly = TRUE)
par(mfrow=c(3,2), mar = c(4,4,3,2))
hist(Speed[Expt==1],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 1', cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==2],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 2', cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==3],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 3', cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==4],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 4', cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==5],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 5', cex.axis=1.5, cex.main = 1.5)
hist(Speed,col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'All Experiments', cex.axis=1.5, cex.main = 1.5)
par(op)


op <- par(no.readonly = TRUE)
par(mfrow=c(3,2), mar = c(4,4,3,2))
hist(Speed[Expt==1],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 1',xlim=c(600,1100),ylim = c(0,9), cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==2],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 2',xlim=c(600,1100),ylim = c(0,9), cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==3],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 3',xlim=c(600,1100),ylim = c(0,9), cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==4],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 4',xlim=c(600,1100),ylim = c(0,9), cex.axis=1.5, cex.main = 1.5)
hist(Speed[Expt==5],col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'Experiment 5',xlim=c(600,1100),ylim = c(0,9), cex.axis=1.5, cex.main = 1.5)
hist(Speed,col = 'azure2',xlab='Speed', ylab='Freq',
     main = 'All Experiments',xlim=c(600,1100), cex.axis=1.5, cex.main = 1.5)
par(op)


## Histograms: Number of bins
hist(Speed,col = 'azure2')
hist(Speed, breaks = 8,col = 'azure2')
hist(Speed, breaks = 16,col = 'azure2')


## Histograms: Anchor
par(mfrow=c(2,2),  mar = c(4,4,4,2)+0.1)
hist(Speed,breaks = seq(590,1090,50),ylim = c(0,30),
     col='azure2',main = 'Starts at 590')
hist(Speed,breaks = seq(600,1100,50),ylim = c(0,30),
     col='azure2',main = 'Starts at 600')
par(mar = c(5,4,3,2)+0.1)
hist(Speed,breaks = seq(610,1110,50),ylim = c(0,30),
     col='azure2',main = 'Starts at 610')
hist(Speed,breaks = seq(620,1120,50),ylim = c(0,30),
     col='azure2',main = 'Starts at 620')
par(op)

## Dot Charts
mu <- numeric(5)
std.dev <- numeric(5)
for (i in 1:5) {
  mu[i] <- mean(Speed[Expt==i])
  std.dev[i] <- sd(Speed[Expt==i])
}
par(mfrow=c(2,2),  mar = c(4,4,4,2)+0.1)
dotchart(mu, labels = 1:5, pch=19, main = 'Average')
dotchart(std.dev, labels = 1:5, pch=19, main = 'Standard dev.')
par(mar=c(4.5,3,3,1))
pie(mu, labels = 1:5, main = 'Average', radius = 1)
pie(std.dev, labels = 1:5, main = 'Standard dev.', radius = 1)
par(op)


cols <- hcl.colors(15, "Set 2")
dat1 <- c(19,19.5,20,20.5,21)
dat2 <- rev(dat1)
dat3 <- c(19,20,21,20.5,19.5)
par(mfrow = c(1,3))
pie(dat1,labels = 1:5,radius = 1, col = cols[c(1,6,8,10,14)])
pie(dat2,labels = 1:5,radius = 1, col = cols[c(1,6,8,10,14)])
pie(dat3,labels = 1:5,radius = 1, col = cols[c(1,6,8,10,14)])

par(mfrow = c(1,3))
barplot(dat1, col = cols[c(1,6,8,10,14)], yaxt = 'n',names.arg = 1:5)
barplot(dat2, col = cols[c(1,6,8,10,14)], yaxt = 'n',names.arg = 1:5)
barplot(dat3, col = cols[c(1,6,8,10,14)], yaxt = 'n',names.arg = 1:5)
par(mfrow=c(1,1))


par(xaxt = 'n', mfrow = c(1,3))
dotchart(dat1,1:5, col = cols[c(1,6,8,10,14)], pch = 16,cex = 1.2, xlim = c(18.5,21.5))
dotchart(dat2,1:5, col = cols[c(1,6,8,10,14)], pch = 16,cex = 1.2, xlim = c(18.5,21.5))
dotchart(dat3,1:5, col = cols[c(1,6,8,10,14)], pch = 16,cex = 1.2, xlim = c(18.5,21.5))
par(op)


## \texttt{pairs}
pairs(iris)
pairs( ~ Sepal.Length + Petal.Length +
     Petal.Width, data = iris)
pairs(iris[1:4], main = "Anderson's Iris Data",pch = 21, 
  col = c('red','green3','blue')[iris$Species])



## \texttt{contour}
# contour(x = seq(0, 1, length.out = nrow(z)),
#        y = seq(0, 1, length.out = ncol(z)),
#        z, nlevels = 10, add = FALSE)



library(mvtnorm)
x.points <- y.points <- seq(-3,3,length.out=100)
z <- matrix(0,nrow=100,ncol=100)
mu <- c(1,1)
sigma <- matrix(c(2,1,1,1),nrow=2)
for (i in 1:100) {
  for (j in 1:100) {
  z[i,j] <- dmvnorm(c(x.points[i],y.points[j]),
                  mean=mu,sigma=sigma)
  }
}
contour(x.points,y.points,z)
filled.contour(x.points,y.points,z)
library(mvtnorm)
normal.contour <- function(mu=c(0,0),
          sigma=matrix(rep(1,4),nrow=2)){
  x.points <- y.points <- seq(-3,3,length.out=100)
  z <- matrix(0,nrow=100,ncol=100)
  for (i in 1:100) {
    for (j in 1:100) {
      z[i,j] <- dmvnorm(c(x.points[i],y.points[j]),
                        mean=mu,sigma=sigma)
    }
  }
  contour(x.points,y.points,z, main=paste('Corr = ',
        sigma[1,2]), cex.main = 0.8, drawlabels = FALSE)
  abline(h=0); abline(v=0)
}
mu = c(0,0)
sigma1 <- matrix(c(1,0,0,1),nrow=2)
sigma2 <- matrix(c(1,.25,.25,1),nrow=2)
sigma3 <- matrix(c(1,.5,.5,1),nrow=2)
sigma4 <- matrix(c(1,.75,.75,1),nrow=2)
par(mfrow = c(2,2), mar = c(4,4,2.5,2))
normal.contour(mu,sigma1)
normal.contour(mu,sigma2)
normal.contour(mu,sigma3)
normal.contour(mu,sigma4)
par(mfrow=c(1,1))

## `persp`
y <- x <- seq(-10, 10, length= 30)
f <- function(x, y) { r <- sqrt(x^2+y^2); 10 * sin(r)/r }
z <- outer(x, y, f); z[is.na(z)] <- 1
persp(x, y, z, theta = 30, phi = 30, 
      expand = 0.5, col = "lightblue")



# Other Chart Types
## Other Chart Types

# sunflowerplot(x,y) 
# stripchart(x) 
# matplot(x,y)
# plot.ts(x)
# image(x,y,z)
# stars(x)
 

