## Experimental Design II


library(PASWR)
mod0 <- lm(StopDist ~ tire, data = Tire)

# Power and the number of replicates

df <- c(10, 30)
curve(df(x,df[1],df[2]),0,5, ylab='density', lwd=2,
      col='slateblue',cex.axis=1.5, cex.lab=1.6)
curve(df(x,df[1],df[2],20),0,5, add=TRUE, col='cyan4')
curve(df(x,df[1],df[2],15),0,5, add=TRUE, col='cyan4')
curve(df(x,df[1],df[2],10),0,5, add=TRUE, col='cyan4')
segments(2,0,2,df(2,df[1],df[2],10), col='red',lwd=2)
abline(h=0)
text(3,0.5,'Non-central F', col='cyan4', cex=2)
text(3.4,0.4,expression(paste('increasing ', lambda, ' ->')), 
     col='cyan4', cex=2)
text(1.7,0.8,'Central F', col='slateblue', cex=2)
sss <- seq(2,3.6,0.2)
polygon(c(sss,2),c(df(sss,df[1],df[2]),0), col='red')
text(2.6,0.1,expression(alpha),col='red', cex=2)


plot(1:4,c(-1,0,0,1), pch=19, axes = FALSE, xlab='',
     ylab='')
abline(h=0); segments(1,0,1,-1); segments(4,0,4,1)
plot(1:4,c(-1,0,0,1), pch=19, axes = FALSE, xlab='',
     ylab='')
#abline(h=0); 
segments(1,0,1,-1); segments(4,0,4,1); segments(1,0,4,0)
text(1.2,-.5, expression(-Delta/2))
text(3.8,.5, expression(Delta/2))
text(1.2,-.95, 'D')
text(3.8,.95, 'A')

library(daewr)
rmin <-2 #smallest number of replicates considered
rmax <-10 # largest number of replicates considered
alpha <- rep(0.05, rmax - rmin +1)
sigma <- 15; nlev <- 4; nreps <- rmin:rmax; Delta <- 30
(power <- Fpower1(alpha,nlev,nreps,Delta,sigma))


# Non-Parametric Tests
sampl1 <- c(286, 251, 325, 313, 309, 302)
sampl2 <- c(249, 324, 289, 303, 310, 318)
sample <- c(rep(1,6), rep(2,6))
wcx <- data.frame(weight=c(sampl1,sampl2), sample)
str(wcx, vec.len = 2)

ord <- order(wcx[,1])
wcx.ord <- cbind(wcx[ord,],rank=1:12)
head(wcx.ord)


plot(wcx[ord,1],rep(.1,12),  axes=FALSE, col=wcx[ord,2], ylim = c(0,.2),
        pch=15+wcx[ord,2],xlim=c(240,340),ylab='', xlab='weight (g)',
     cex = 1.5)
axis(1)


(rank.sum <- c( sum(wcx.ord[,3][wcx.ord[,2]==1]),
                sum(wcx.ord[,3][wcx.ord[,2]==2]) ))

wilcox.test(sampl1,sampl2)

## The Kruskal-Wallis test

results <- read.table('yields.txt',header=T)
frame <- stack(results)
names(frame) <- c('yield','soil')
with(frame,summary(aov(yield~soil)))
with(frame, kruskal.test(yield~soil))

# We get similar $p$-values in this case, $0.025$ in the Anova table and $0.0226$ in the K-W test, so we would reach the same conclusion for the usual values of $\alpha$.

anova(mod0)
with(Tire, kruskal.test(StopDist ~ tire))



