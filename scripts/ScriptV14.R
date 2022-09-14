#  Review of Inference I: Pointwise Estimation

# Pointwise estimation
## Example 1.

set.seed(6556)
smpl1 <- rnorm(10,4.5,2)
round(smpl1,2)
(mean.smpl1 <- mean(smpl1))

(var.smpl1 <- var(smpl1))


# Sampling distribution
smpl2 <- rnorm(10,4.5,2)
round(smpl2,2)
mean(smpl2)

## Example 1 revisited
smpl.mat <- matrix(rnorm(25000,4.5,2), ncol = 10)
mean.vec <- apply(smpl.mat,1,mean)
hist(mean.vec, breaks = 20, freq = FALSE, 
     xlab = 'Sample mean',
     main = 'Histogram of 25000 sample means', 
     xlim = c(0,10))
lines(density(mean.vec, adjust = 1.5), 
      col = 'red',lwd = 2)
curve(dnorm(x,4.5,2), 0, 10, add = TRUE, 
      col = 'blue', lwd = 2)


## Asymptotic sampling distribution for the mean
## Simulation Example

set.seed(11)
exp.mean1 <- numeric(10000)
for (i in 1:10000) {
  exp.mean1[i] <- mean(rexp(20))}
## Another way to do this
set.seed(11)
exp.smpls <- matrix(rexp(200000),ncol = 20, byrow = T)
exp.mean2 <- apply(exp.smpls,1,mean)

head(exp.mean1)
head(exp.mean2)
qqplot(exp.mean1,exp.mean2)


hist(exp.mean1, breaks = 20, freq = FALSE, 
     ylim=c(0,2), xlim = c(0.3,2),
     main='Sampling distribution for n=20', 
     xlab='values', cex.main = 0.8)
lines(density(exp.mean1), col='red', lwd=2)
curve(dnorm(x,1,1/sqrt(20)),0,2,add = TRUE, 
      col='blue2', lwd=2)

# We repeat this simulation with samples of size 50.

exp.smpls <- matrix(rexp(500000),ncol = 50, byrow = T)
exp.mean5 <- apply(exp.smpls,1,mean)
hist(exp.mean5, breaks = 20, freq = FALSE, ylim=c(0,3),
     xlim = c(0.3,2),
     main='Sampling distribution for n=50', 
     xlab='values', cex.main = 0.8)
lines(density(exp.mean5), col='red', lwd=2)
curve(dnorm(x,1,1/sqrt(50)),0,2.,add = TRUE, 
      col='blue2', lwd=2)

# Comparison of simulations
par(mfrow = c(2,1))
hist(exp.mean1, breaks = 20, freq = FALSE, 
     ylim=c(0,3), xlim = c(0.3,2),
     main='Sampling distribution for n=20', 
     xlab='values', cex.main = 0.8)
lines(density(exp.mean1), col='red', lwd=2)
curve(dnorm(x,1,1/sqrt(20)),0,2,add = TRUE, 
      col='blue2', lwd=2)
hist(exp.mean5, breaks = 20, freq = FALSE, 
     ylim=c(0,3), xlim = c(0.3,2),
     main='Sampling distribution for n=50', 
     xlab='values', cex.main = 0.8)
lines(density(exp.mean5), col='red', lwd=2)
curve(dnorm(x,1,1/sqrt(50)),0,2.,add = TRUE, 
      col='blue2', lwd=2)
par(mfrow=c(1,1))

cols <- hcl.colors(6, 'terrain')
curve(dnorm(x,sd=1/sqrt(10)),-1,1,ylim=c(0,4),
      lwd=2, cex.main = 0.8, main = 'Sampling density', 
      xlab = 'values', ylab = 'density', col = cols[1])
curve(dnorm(x,sd=1/sqrt(25)),-1,1, add = TRUE, 
      col=cols[2], lwd=2)
curve(dnorm(x,sd=1/sqrt(50)),-1,1, add = TRUE,
      col=cols[3], lwd=2)
curve(dnorm(x,sd=1/sqrt(100)),-1,1, add = TRUE, 
      col=cols[4], lwd=2)
legend('topright',legend = c(10,25,50,100), 
       col=cols[1:4], lwd=rep(2,4))





















