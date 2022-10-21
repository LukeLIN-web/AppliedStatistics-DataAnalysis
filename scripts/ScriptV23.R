## | Analysis of Variance  


## A Worked-Out Example
results <- read.table('yields.txt',header=T)
attach(results)
str(results)
head(results, n=4)


apply(results, 2, mean)
# Another possibility
sapply(list(sand,clay,loam),mean)


frame <- stack(results)
str(frame)

names(frame) <- c('yield','soil')
attach(frame)
head(frame)

plot(yield~soil,col='lightblue3')

sum((yield-mean(yield))^2)

sand-mean(sand)
clay-mean(clay)
loam-mean(loam)

(sums.vec <- c(sum((sand-mean(sand))^2), sum((clay-mean(clay))^2),
  sum((loam-mean(loam))^2)))
sum(sums.vec)

sum(apply(results,2,
           function (x) sum((x-mean(x))^2) ))

sum(sapply(list(sand,clay,loam),
           function (x) sum((x-mean(x))^2) ))


apply(results, 2, var)
mean(apply(results, 2, var))

1-pf(4.24,2,27)


summary(aov(yield~soil))

## Plots
plot(aov(yield~soil), which = 1, cex.lab=0.7, cex.sub=0.7)

plot(aov(yield~soil), which = 2, cex.lab=0.7, cex.sub=0.7)

plot(aov(yield~soil), which = 3, cex.lab=0.7, cex.sub=0.7)

plot(aov(yield~soil), which = 4, cex.lab=0.7, cex.sub=0.7)

