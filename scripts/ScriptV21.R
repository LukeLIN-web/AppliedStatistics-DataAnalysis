#  Comparing Proportions

# The binomial distribution

par(mfrow=c(1,2))
plot(0:20, dbinom(0:20,20,0.4), type='h', lwd=3,
     ylab='probability',xlab='x',main='p = 0.25')
curve(dnorm(x,8,sqrt(20*0.4*0.6)),0,20, col='red2', 
      add = TRUE, lwd=2)
plot(0:20,dbinom(0:20,20,0.1), type='h', lwd=3, 
     ylim = c(0,0.3),ylab='probability',xlab='x',main='p = 0.1')
curve(dnorm(x,1,sqrt(20*0.1*0.9)),0,20, col='red2', 
      add = TRUE, lwd=2)



## Example
n <- 1050 ; n.A <- 177 
p_0 <- 0.25 ; alpha <- 0.01
SE <- sqrt(p_0 * (1 - p_0) / n)
(pi <- 177/1050)
pnorm(pi,p_0,SE)

prop.test(n.A,n,p_0)
binom.test(n.A,n,p_0)

# Two independent proportions
## Example
car.accidents <- data.frame(survived = c(1781,1443),
                            died=c(135,47))
rownames(car.accidents) <- c('nsb','sb')
car.accidents

prop.test(as.matrix(car.accidents))

# Another syntax:
prop.test(c(1781,1443),c(1916,1490))






