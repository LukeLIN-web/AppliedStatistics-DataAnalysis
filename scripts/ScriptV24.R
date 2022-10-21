## Experimental Design 1

library(PASWR)
str(Tire)
head(Tire, n=4)

with(Tire, oneway.plots(StopDist, tire))

mod0 <- lm(StopDist ~ tire, data = Tire)
summary(mod0)
