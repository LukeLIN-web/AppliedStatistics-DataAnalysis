### | Quantile plots  


  ## Quantile plots
data(morley)
str(morley, vec.length = 1)
qqplot(morley$Speed[morley$Expt==1],
       morley$Speed[morley$Expt==2],
       xlab='Experiment 1', ylab = 'Experiment 2',pch=19)

qqplot(morley$Speed[morley$Expt==4],
       morley$Speed[morley$Expt==2],
       xlab='Experiment 4', ylab = 'Experiment 2',pch=19)



qqnorm(morley$Speed[morley$Expt==2], pch=16)

qqnorm(morley$Speed)

qqline(morley$Speed)

qqline(morley$Speed,probs = c(0.18,0.8), col='red')


## Some Simulations

set.seed(1290)
norm1000 <- rnorm(1000); norm100 <- norm1000[1:100]
norm50 <- norm1000[1:50]; norm20 <- norm1000[1:20]

par(mfrow=c(2,2))
qqnorm(norm20,main='Sample size 20'); qqline(norm20)
qqnorm(norm50,main='Sample size 50'); qqline(norm50)
qqnorm(norm100,main='Sample size 100'); qqline(norm100)
qqnorm(norm1000,main='Sample size 1000'); qqline(norm1000)


set.seed(129021)
norm1000 <- rnorm(1000); norm100 <- norm1000[1:100]
norm50 <- norm1000[1:50]; norm20 <- norm1000[1:20]
par(mfrow=c(2,2))
qqnorm(norm20,main='Sample size 20'); qqline(norm20)
qqnorm(norm50,main='Sample size 50'); qqline(norm50)
qqnorm(norm100,main='Sample size 100'); qqline(norm100)
qqnorm(norm1000,main='Sample size 1000'); qqline(norm1000)


par(mfrow=c(2,2))
qqnorm(norm1000)
qqline(norm1000)
t3.1000 <- rt(1000,3)
qqnorm(t3.1000, main='t')
qqline(t3.1000)
lnorm1000 <- rlnorm(1000)
qqnorm(lnorm1000, main='Lognormal')
qqline(lnorm1000)
chi2.1000 <- rchisq(1000,3)
qqnorm(chi2.1000, main='Chi squared' )
qqline(chi2.1000)


par(mfrow=c(1,1))
curve(dnorm(x),-3,6,ylim=c(0,0.65),lwd=2, ylab='density')
curve(dt(x,3),-3,6,lwd=2,col='blue',add=TRUE)
curve(dlnorm(x),0,6,add=TRUE,col='dodgerblue1',lwd=2)
curve(dchisq(x,3),0,6,add=TRUE,col='cyan3',lwd=2)
legend('topleft',c('normal',expression(t[3]),'lognorm',
                   expression(chi^2)),lty=rep(1,4),lwd=rep(2,4),
       col=c('black','blue','dodgerblue1','cyan3'))

norm1000a <- norm1000
norm1000b <- norm1000
norm1000c <- norm1000
norm1000a[1:2] <- c(10,12)
norm1000b[1:2] <- c(10,-12)
norm1000c[1:2] <- c(-10,-12)
qqnorm(norm1000a)
qqline(norm1000a)
qqnorm(norm1000b)
qqline(norm1000b)
qqnorm(norm1000c)
qqline(norm1000c)

par(mfrow=c(2,2))
norm200 <- norm1000[1:200]
norm161 <- c(norm100,rnorm(100,mean=6))
norm104 <- c(norm100,rnorm(100,sd=2))
norm164 <- c(norm100,rnorm(100,mean=6,sd=2))
qqnorm(norm200,main='N(0,1)')
qqline(norm200)
hist(norm200,nclass=16,main='N(0,1)')
qqnorm(norm161,main='N(0,1) + N(6,1)')
qqline(norm161)
hist(norm161,nclass=16,main='N(0,1)+N(6,1)')
qqnorm(norm104,main='N(0,1)+N(0,4)')
qqline(norm104)
hist(norm104,nclass=16,main='N(0,1)+N(0,4)')
qqnorm(norm164,main='N(0,1)+N(6,4)')
qqline(norm164)
hist(norm164,nclass=16,main='N(0,1)+N(6,4)')







