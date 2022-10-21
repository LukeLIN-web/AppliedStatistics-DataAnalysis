## Analysis of Variance  I


## One-way Analysis of Variance
## Figure 1
set.seed(34567)
samp2 <- samp1 <- numeric(5)
samp1 <- 4.5 + rnorm(5)
samp2 <- 5.5 + rnorm(5)
sampl <- numeric(10)
sampl[c(1,3,5,7,9)] <- samp1
sampl[c(2,4,6,8,10)] <- samp2
plot(sampl, ylab='Response', pch=16, main='SST',
     ylim = c(2,7), cex = 1.5)
abline(h=mean(sampl))
segments(1:10,mean(sampl),1:10,sampl)

## Figure 2
plot(sampl, ylab='Response', pch=16, main='SST',
     ylim = c(2,7))
abline(h=mean(sampl))
points(c(1,3,5,7,9),sampl[c(1,3,5,7,9)], col='blue', pch=16, cex = 1.5)
points(c(2,4,6,8,10),sampl[c(2,4,6,8,10)], col='red', pch=16, cex = 1.5)
segments(1:10,mean(sampl),1:10,sampl)

## Figure 3
plot(sampl, ylab='Response', main='SSE', type = 'n',
     ylim = c(2,7))
points(1:5, samp1, pch=16, col='blue', cex = 1.5)
points(6:10, samp2, pch=16, col='red', cex = 1.5)
segments(1,mean(samp1),5.5,mean(samp1),col='blue', lwd=2)
segments(5.5,mean(samp2),10,mean(samp2),col='red', lwd=2)
segments(1:5,mean(samp1),1:5,samp1, col='blue')
segments(6:10,mean(samp2),6:10,samp2, col='red')

## Figure 4
plot(sampl, ylab='Response', main='SST = SSE', type = 'n',
     ylim = c(2,7))
points(1:5,samp1+0.5, col = 'blue', pch=16, cex = 1.5)
points(6:10, samp2-0.5, pch=16, col='red', cex = 1.5)
segments(1,4.5,5.4,4.5,col='blue', lwd=2)
segments(5.6,4.5,10,4.5,col='red', lwd=2)
segments(1:5,4.5,1:5,samp1+0.5, col='blue')
segments(6:10,4.5,6:10,samp2-0.5, col='red')


## Figure 5
par(mfrow=c(1,2))
plot(1:10,c(rep(4.5,5),rep(5.5,5)),pch=16,
     col=c(rep('blue',5),rep('red',5)),ylim=c(3,7),
     ylab='Response', xlab = 'Index', main = 'SST big')
abline(h=5)
segments(1:5,5,1:5,4.5)
segments(6:10,5,6:10,5.5)

plot(1:10,c(rep(4.5,5),rep(5.5,5)),pch=16,
     col=c(rep('blue',5),rep('red',5)),ylim=c(3,7),
     ylab='Response', xlab = 'Index', main = 'SSE = 0')
segments(1,4.5,5.5,4.5,col='blue',lwd=2)
segments(5.5,5.5,10,5.5,col='red',lwd=2)
par(mfrow=c(1,1))

plot(sampl, ylab='Response', main='SST = SSE', type = 'n',
     ylim = c(2,7))
points(1:5,samp1+0.5, col = 'blue', pch=16)
points(6:10, samp2-0.5, pch=16, col='red')
segments(1,4.5,5.4,4.5,col='blue', lwd=2)
segments(5.6,4.5,10,4.5,col='red', lwd=2)
segments(1:5,4.5,1:5,samp1+0.5, col='blue')
segments(6:10,4.5,6:10,samp2-0.5, col='red')




