#  | STAT 210
#| Applied Statistics and Data Analysis
#| Linear Regression VI: 
#  | Outliers and Transformations


# Atypical Points
## Outliers or Atypical Points


tv <- read.csv('TV.csv')
plot(tv)
fittv <- lm(tv$y ~ tv$x)
summary(fittv)


plot(fittv, which=1)
plot(fittv, which=2)
plot(fittv, which=3)
plot(fittv, which=5)

plot(tv)
abline(fittv)
points(tv[27:30,],col='red',bg = 'red', pch = 21)

tv2 <- tv[1:26,]; plot(tv2)
fittv2 <- lm(tv2$y ~ tv2$x)
abline(fittv2)

summary(fittv2)

plot(fittv2, which = 1)



# Transformed Data
## Transformed Data

Bacteria <- read.csv('Bacteria.csv')
plot(Bacteria)

attach(Bacteria)
fitbac <- lm(Count ~ Time)
summary(fitbac)

plot(Bacteria)
abline(fitbac)

plot(fitbac, which=1)
plot(fitbac, which=2)
plot(fitbac, which=3)
plot(fitbac, which=5)


fitlogbac <- lm(log(Count) ~ Time)
summary(fitlogbac)

plot(fitlogbac, which=1)
plot(fitlogbac, which=2)
plot(fitlogbac, which=3)
plot(fitlogbac, which=5)






