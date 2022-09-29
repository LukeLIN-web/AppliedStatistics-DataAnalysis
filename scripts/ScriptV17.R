## | Comparing Two Populations

# Example 1

getwd()
fish1 = matrix(scan("data/diet1.1"), ncol=1, byrow=T)
fish2 = matrix(scan("data/diet1.2"), ncol=1, byrow=T)

scan("data/diet1.1")

mean(fish1)
mean(fish2)


## Histograms
par(mfrow=c(1,2)) 
hist(fish1, xlab="Composite Score", 
     main="Histogram Diet Group 1", col='azure3')
hist(fish2, xlab="Composite Score", 
     main="Histogram Diet Group 2", col='azure3')

par(mfrow=c(1,2)) 
hist(fish1, xlab="Composite Score", 
     main="Histogram Diet Group 1",xlim=c(35,75), 
     col='azure3')
hist(fish2, xlab="Composite Score", 
     main="Histogram Diet Group 2",xlim=c(35,75), 
     col='azure3')

(minval = 0.80*min(fish1, fish2))  
(maxval = 1.10*max(fish1, fish2)) 

par(mfrow=c(2,1)) 
hist(fish1, xlab="Composite Score", col='azure3', 
     main= 'Histogram Diet Group 1', 
     xlim=c(minval, maxval))
hist(fish2, xlab="Composite Score", col='azure3', 
     main= "Histogram Diet Group 2", 
     xlim=c(minval, maxval))

## Boxplots

par(mfrow=c(1,2)) 
boxplot(fish1,main='Diet 1',col=5) 
boxplot(fish2,main="Diet 2",col=5)


par(mfrow=c(1,1))
fish = c(fish1, fish2) 
n1 = length(fish1); n2 = length(fish2)
group = c(rep(1, n1), rep(2, n2)) 
fish.df <- data.frame(fish=fish, group = group)
boxplot(fish ~ group, data = fish.df,
        main = "Outcomes for the Two Diet Groups",
        col = 'azure3')




## Outliers 

fish1s = sort(fish1); fish2s = sort(fish2)
mean(fish1s); mean(fish2s)

fish1s
fish2s

# Another version
with(fish.df, plot(fish, jitter(group, factor=0.2),
                   xlab='Outcome', ylab='Diet group', 
                   ylim = c(0.5,2.5),
                   pch=19, col = group +2, 
                   yaxp = c(1,2,1)))


fish1x = fish1s
fish2x = fish2s
fish2x[n2] = 1000  # artificial outlier set at 
# the end of the second sample

mean(fish1x)
mean(fish2x)

par(mfrow=c(1,1))
fish1x = c(fish1x, fish2x)
boxplot(fish1x ~ group, 
        main="Outcomes for the Two Diet Groups")


## Example 2
group1 = matrix(scan("data/comp.1"), ncol=2, byrow=T)
group2 = matrix(scan("data/comp.2"), ncol=2, byrow=T)

group1.pre = group1[,1];
group1.post = group1[,2];
group2.pre = group2[,1];
group2.post = group2[,2]


group1.post

sort(group1.post);sort(group2.post)
mean(group1.post);mean(group2.post)


par(mfrow=c(1,2))
hist(group1.post, xlab="Post Score", 
     main="Histogram Group 1 POST", col='azure3')
hist(group2.post, xlab="Post Score", 
     main="Histogram Group 2 POST", col='cornsilk3')

par(mfrow=c(2,1), mar=c(4,4,3,2))
hist(group1.post, xlab="Post Score", 
     main="Histogram Group 1 POST", col='azure3')
hist(group2.post, xlab="Post Score", 
     main="Histogram Group 2 POST", col='cornsilk3')

minval=0.95*min(group1.post,group2.post)
maxval=1.05*max(group1.post,group2.post)
par(mfrow=c(2,1),mar = c(4,4,3,2))
hist(group1.post, xlab="Post Score", 
     main="Histogram Group 1 POST",  
     xlim=c(minval, maxval), col='azure3')
hist(group2.post, xlab="Post Score", 
     main="Histogram Group 2 POST",  
     xlim=c(minval, maxval), col='cornsilk3')


### Numerical Summaries

summary(group1.post)
summary(group2.post)

par(mfrow=c(2,1),mar = c(4,4,3,2))
hist(group1.pre, xlab="Pre Score", 
     main="Histogram Group 1 PRE", col='azure2')
hist(group2.pre, xlab="Pre Score", 
     main="Histogram Group 2 PRE", col='cornsilk2')

  
par(mfrow=c(2,1),mar = c(4,4,3,2))
minval = 0.95*min(group1.pre, group2.pre)
maxval = 1.10*max(group1.pre, group2.pre)
hist(group1.pre, xlab="Pre Score", 
     main="Histogram Group 1 PRE",  
     xlim=c(minval, maxval), col='azure2')
hist(group2.pre, xlab="Pre Score", 
     main="Histogram Group 2 PRE",  
     xlim=c(minval, maxval), col='cornsilk2')

par(mfrow=c(2,2))
minval.pre = 0.95*min(group1.pre, group2.pre)
maxval.pre = 1.10*max(group1.pre, group2.pre)
minval.post = 0.95*min(group1.post, group2.post)
maxval.post= 1.10*max(group1.post, group2.post)

hist(group1.pre, xlab="Pre Score", 
     main="Histogram Group 1 PRE", 
     xlim=c(minval.pre, maxval.pre), col='azure2')
hist(group1.post, xlab="Post Score", 
     main="Histogram Group 1 POST", 
     xlim=c(minval.pre, maxval.pre), col='azure3')

hist(group2.pre, xlab="Pre Score", 
     main="Histogram Group 2 PRE", 
     xlim=c(minval.pre, maxval.pre), col='cornsilk2')
hist(group2.post, xlab="Post Score", 
     main="Histogram Group 2 POST", 
     xlim=c(minval.pre, maxval.pre), col='cornsilk3')

# Another perspective:
  
par(mfrow=c(2,2))
minval.pre = 0.95*min(group1.pre, group2.pre)
maxval.pre = 1.10*max(group1.pre, group2.pre)
minval.post = 0.95*min(group1.post, group2.post)
maxval.post= 1.10*max(group1.post, group2.post)

hist(group1.pre, xlab="Pre Score", 
     main="Histogram Group 1 PRE", 
     xlim=c(minval.pre, maxval.pre), col='azure2')
hist(group2.pre, xlab="Pre Score", 
     main="Histogram Group 2 PRE", 
     xlim=c(minval.pre, maxval.pre), col='cornsilk2')

hist(group1.post, xlab="Post Score", 
     main="Histogram Group 1 POST", 
     xlim=c(minval.pre, maxval.pre), col='azure3')
hist(group2.post, xlab="Post Score", 
     main="Histogram Group 2 POST", 
     xlim=c(minval.pre, maxval.pre), col='cornsilk3')


imp1 = group1.post - group1.pre
imp2 = group2.post - group2.pre
cbind(imp1, imp2)

par(mfrow=c(2,1))
minval.imp = 0.95*min(imp1, imp2)
maxval.imp = 1.10*max(imp1, imp2)
hist(imp1, xlab="Improvement Score", 
     main="Histogram Group 1 IMPROVEMENT", 
     xlim=c(minval.imp, maxval.imp), col='azure3')
hist(imp2, xlab="Improvement Score", 
     main="Histogram Group 2 IMPROVEMENT", 
     xlim=c(minval.imp, maxval.imp), col='cornsilk3')

### Summary statistics
summary(imp1)
summary(imp2)

par(mfrow=c(1,2))
boxplot(imp1, main = "Method 1", col='azure3')
boxplot(imp2, main = "Method 2", col='cornsilk3')

par(mfrow=c(1,2))
boxplot(imp1, main = "Method 1", col='azure3', 
        ylim = c(minval.imp, maxval.imp))
boxplot(imp2, main = "Method 2", col='cornsilk3', 
        ylim = c(minval.imp, maxval.imp))


## Example 3
## **Understanding uncertainty in statistical results 
## arising from random sampling**


n1 = n2 = 5
mu1 = mu2 = 60
sd1 = sd2 = 5
samp1 = rnorm(n1, mu1, sd1)
samp2 = rnorm(n2, mu2, sd2)
mean(samp1)
mean(samp2)
mean(samp1) - mean(samp2)


n1 = n2 = 5
B = 10000
samp1.mat <- matrix(rnorm(n1*B,mu1,sd1), ncol = n1)
samp2.mat <- matrix(rnorm(n1*B,mu1,sd1), ncol = n2)
mean1.vec <- apply(samp1.mat, 1, mean)
mean2.vec <- apply(samp2.mat, 1, mean)
diff.mean.5 <- mean1.vec - mean2.vec

par(mfrow=c(1,1))
hist(diff.mean.5, col='seashell3', 
     main='Differences in the means (n=5)', 
     xlab='values')


mean(diff.mean.5); sd(diff.mean.5)
summary(diff.mean.5)
sort(diff.mean.5)[c(.1*B,.9*B)]

n1 = n2 = 500
samp1.mat <- matrix(rnorm(n1*B,mu1,sd1), ncol = n1)
samp2.mat <- matrix(rnorm(n1*B,mu1,sd1), ncol = n2)
mean1.vec <- apply(samp1.mat, 1, mean)
mean2.vec <- apply(samp2.mat, 1, mean)
diff.mean.500 <- mean1.vec - mean2.vec


par(mfrow=c(1,1))
hist(diff.mean.500, col='powderblue', 
     main = 'Differences in the means (n=500)', 
     xlab='values')


mean(diff.mean.500); sd(diff.mean.500)
summary(diff.mean.500)
sort(diff.mean.500)[c(.1*B,.9*B)]

# Compare the two results
minval = min(diff.mean.5, diff.mean.500)
maxval = max(diff.mean.5, diff.mean.500)
par(mfrow=c(2,1))
hist(diff.mean.5, xlim=c(minval, maxval), 
     main="Sampling Distribution n=5", 
     col='seashell3',xlab='values')
hist(diff.mean.500, xlim=c(minval, maxval), 
     main="Sampling Distribution n=500", 
     col='powderblue',xlab='values')

par(mfrow=c(1,1))
diff.mean = c(diff.mean.5, diff.mean.500)
group = c(rep(5, B), rep(500,B))
boxplot(diff.mean~group, col='seashell3', 
        main="Sampling Distributions")




