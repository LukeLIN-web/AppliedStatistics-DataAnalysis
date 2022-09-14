
# Review of Inference II: Interval Estimation

set.seed(6556)
smpl1 <- rnorm(10,4.5,2)
mean.smpl1 <- mean(smpl1)
var.smpl1 <- var(smpl1)

# Interval estimation
points.x <- seq(-3,3,length.out = 101)
points.y <- dnorm(points.x)
xw <- points.x[points.x >= 1.74]; 
xw <-c(xw, 3.0,1.74)
yw <- points.y[points.x >= 1.74]; 
yw <- c(yw,0,0)
plot(points.x,points.y,type='l',xlab='',
     main='Gaussian Density',
     ylab='', axes=FALSE, cex.main = 0.8)
abline(h=0);  abline(v=0)
polygon(xw,yw, col='tan1')
axis(1,at=c(1.74),labels = expression(z[alpha]),
     line=0,pos=0,lwd=2)
text(2,.022,expression(alpha))

## Confidence intervals

points.x <- seq(-3,3,length.out = 101)
points.y <- dnorm(points.x)
xw <- points.x[points.x >= 1.74]; xw <-c(xw, 3.0,1.74)
yw <- points.y[points.x >= 1.74]; yw <- c(yw,0,0)
xv <- points.x[points.x <= -1.74]; xv <-c(xv, -1.74, -3)
yv <- points.y[points.x <= -1.74]; yv <- c(yv,0,0)
plot(points.x,points.y,type='l', xlab='',
     main='Gaussian Density',
     ylab='', axes=FALSE)
abline(h=0.0);  abline(v=0)
polygon(xw,yw, col='tan1')
axis(1,at=c(1.74),labels = expression(z[alpha]),
     line=0, pos=0, lwd=2)
text(2,.022,expression(alpha))
polygon(xv,yv, col='tan1')
axis(1,at=c(-1.74),labels = expression(-z[alpha]),
     line=0, pos=0, lwd=2)
text(-2,.022,expression(alpha))


## Example 2
(alp <- qnorm(0.01))

(4*abs(alp))^2



cols <- rainbow(100)
curve(dnorm(x),-3,3,lwd=2, ylab='density',col='grey25')
for (i in c(2,4,6)) {
  curve(dt(x,i),-3,3,lwd=2, add = TRUE, col=cols[55+3*i])}
curve(dt(x,30),-3,3,lwd=2, add = TRUE, col=cols[85])
legend('topright',c('normal','t2','t4','t6','t30'), cex=0.5,
       col=c('grey25',cols[c(61,67,73,85)]),lwd=rep(2,5))
abline(h=0); abline(v=0)

## Example 1 revisited

mean.smpl1

I10 <- matrix(numeric(6), ncol=3)
(zz <- abs(qnorm(0.025)))

(I10[,1] <- c(mean.smpl1 - (zz*2/sqrt(10)),
              mean.smpl1 + (zz*2/sqrt(10))))

(I10[,2] <- c(mean.smpl1 - (zz*sqrt(var.smpl1)/sqrt(10)),
              mean.smpl1 + (zz*sqrt(var.smpl1)/sqrt(10))))

tt9 <- abs(qt(0.025,9))
(I10[,3] <- c(mean.smpl1 - (tt9*sqrt(var.smpl1)/sqrt(10)),
              mean.smpl1 + (tt9*sqrt(var.smpl1)/sqrt(10))))



plot(c(2,2.5,3,3.5),c(2.5,3,5,6), type = 'n', 
     xlab = '', ylab = '',
     axes = FALSE)
for (i in 0:2) {
  segments(2.2+i*0.5,I10[1,i+1],2.2+i*0.5,I10[2,i+1], lwd = 3, col = 'blue')
}
abline(h=4.5, lwd = 2)
abline(h = mean.smpl1, col = 'red', lwd = 2)
axis(2)
text(2.2,2.7,'(a)')
text(2.7,2.7,'(b)')
text(3.2,2.7,'(c)')
legend('top',c('population','sample'), 
       col = 1:2, lwd=c(2,2), cex = 0.7)

## 100 confidence intervals
tt9 <- abs(qt(0.025,9))
smpl.mat <- matrix(rnorm(1000,4.5,2), ncol = 10)
mean.vec <- apply(smpl.mat,1,mean)
sd.vec <- apply(smpl.mat,1,sd)
ints.left <- mean.vec - (tt9*sd.vec/sqrt(10))
ints.right <- mean.vec + (tt9*sd.vec/sqrt(10))

plot(1:100, rep(4.5,100), type = 'l', ylim = c(1,8),
     ylab = 'Confidence interval')
for (i in 1:100) {
  segments(i,ints.left[i],i,ints.right[i])
}
index <- (1:100)[ints.left> 4.5 | ints.right < 4.5]
for (i in index) {
  segments(i,ints.left[i],i,ints.right[i], col = 2)
}


