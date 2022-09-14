#  One Sample Problems II

library(MASS)
data(michelson)
mich.exp1 <- michelson[michelson$Expt == 1,1]

# Second Alternative:      Student's $t$ distribution

cols <- rainbow(100)
curve(dnorm(x),-4,4,lwd=2, ylab='density',col='red')
for (i in c(2,4,6)) 
  {
  curve(dt(x,i),-4,4,lwd=2, add = TRUE, col=cols[55+3*i])
  }
curve(dt(x,30),-4,4,lwd=2, add = TRUE)
legend('topright',
       c('normal','t2','t4','t6','t30'),
       cex=0.7,
       col=c('red',cols[c(61,67,73)],'black'),
       lwd=rep(2,5))
abline(h=0); abline(v=0)


sd(mich.exp1)
sd(mich.exp1)/sqrt(20)


2*pt(-3.452,df=19)

t.test(mich.exp1, mu=990)


# Third Alternative: Non-parametric test


(unif.spl <- round(runif(5,-1,1),3))
sort(unif.spl)
rank(unif.spl)


curve(dnorm(x,2,1),-3,7,ylab = 'density', lwd=2)
abline(h=0);abline(v=0)
abline(v=2, col=4, lwd=1.6)
axis(1,at=c(1,3),labels = c('a','b'))
points(c(1,3),c(dnorm(c(1,3),2,1)), col = 'red3', pch = 16)
arrows(1,dnorm(1,2,1),2,dnorm(1,2,1),code = 3,
       length = 0.1, col = 'red3')
text(1.5,dnorm(1,2,1)+.02,expression(d[i]), 
     col = 'red3')


wilcox.test(mich.exp1,mu=990)

mich.dif <- mich.exp1-990
sort(mich.dif)
length((mich.dif)[mich.dif<0])
rank(abs(mich.dif))

mich.signrank <- rank(abs(mich.dif))*sign(mich.dif)
sum(mich.signrank[mich.signrank>0])

