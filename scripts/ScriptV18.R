## | Two-Sample Problems

fish1 = matrix(scan("data/diet1.1"), ncol=1, byrow=T)
fish2 = matrix(scan("data/diet1.2"), ncol=1, byrow=T)
fish = c(fish1, fish2) 
n1 = length(fish1); n2 = length(fish2)
group = c(rep(1, n1), rep(2, n2)) 
fish.df <- data.frame(fish=fish, group = group)
group1 = matrix(scan("data/comp.1"), ncol=2, byrow=T)
group2 = matrix(scan("data/comp.2"), ncol=2, byrow=T)
group1.pre = group1[,1];
group1.post = group1[,2];
group2.pre = group2[,1];
group2.post = group2[,2]
imp1 = group1.post - group1.pre
imp2 = group2.post - group2.pre


# Comparing two population means

## Case 1: Equal variances

t.test(fish1,fish2, var.equal = TRUE)
with(fish.df,t.test(fish ~ group, var.equal = TRUE))

t.test(imp1,imp2, var.equal = TRUE)

## Case 2: Unequal variances
t.test(fish1,fish2)

t.test(imp1,imp2)

# Comparison of Variances
c(var(fish1), var(fish2))
(F.ratio <- var(fish1)/var(fish2))
2*(1-pf(F.ratio, length(fish1)-1, length(fish2)-1))
var.test(fish1,fish2)

c(var(imp1),var(imp2))
var.test(imp1, imp2)

# Wilcoxon's test
wilcox.test(fish1,fish2)

wilcox.test(imp1,imp2)
# Tests on paired samples.
t.test(group2.pre, group2.post)

t.test(group2.pre, group2.post,paired = TRUE)
cor(group2.pre,group2.post)
plot(group2.pre,group2.post, pch = 16)


t.test(group1.pre, group1.post)
t.test(group1.pre, group1.post,paired = TRUE)
cor(group1.pre,group1.post)

plot(group1.pre,group1.post, pch = 16)

## A more extreme example

library(MASS)
shoes

with(shoes,t.test(A, B, var.equal = TRUE))
with(shoes,t.test(A, B))
with(shoes,t.test(A, B, paired = TRUE))
with(shoes,cor(A,B))
plot(shoes$A,shoes$B,pch=16)

## The Wilcoxon test for matched pairs.
with(shoes, wilcox.test(A, B, paired = TRUE))









