

# problem 2
library(MASS)
par(mfrow=c(2,2))
truehist(mix.sample[1:25],h=0.5,xlim=c(-3,3),ymax=1)
lines(density(mix.sample),col="blue")
lines(points.x,points.dens,type='l',col="red")

# problem 3


par(mfrow=c(2,2))
for (i in 1:4) {
  samp1
}

for (i in c(-3,-1,1,3)) {
  dat <- rnorm(30,i);
}

par(mfrow=c(2,2))
for (i in c(0.5,1,2,3)) {
  dat <- rnorm(30,0,i);
  qqnorm(dat,ylim=c(-6,6));
}

par(mfrow=c(2,2))
for (i in 1:4) {
  dat <- rnorm(30,0,i);
  qqnorm(dat,ylim=c(-6,6));
}

#problem 4
# 4 type of distribution, quantile
# Distributions bounded below.
library(EnvStats)
x <- rchisq(50,2)
