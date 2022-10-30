## Hypothesis tests


library(PASWR)
mod0 <- lm(StopDist ~ tire, data = Tire)
mod1 <- aov(StopDist ~ tire, data = Tire)
summary(mod1)

anova(mod0)

names(mod1)
names(anova(mod0))

plot(mod1)


plot(residuals(mod1) ~ sample(1:24,24), xlab='Expt', 
     ylab='resicuals', main='Residuals vs. Exp. Unit', 
     font.main=1)
abline(h=0, col='red')

plot((1:24),residuals(mod1), xlab='Expt', 
     ylab='resicuals', main='Residuals vs. Exp. Unit', 
     font.main=1)
abline(h=0, col='red')

model.tables(mod1, se=T)
model.tables(mod1,'means',  se=T)

summary(mod0)

1 - dbinom(0,100,0.05); 1 - dbinom(0,100,0.01)

with(Tire, pairwise.t.test(StopDist, tire, 
                           p.adjust.method = 'bonferroni'))

(mod1.tky <-TukeyHSD(mod1))
plot(mod1.tky)


