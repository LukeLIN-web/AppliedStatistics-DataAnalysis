# | Applied Statistics and Data Analysis
# | Multiple Linear Regression 3    
# | Model Selection 


# Example

library(PASWR)
attach(HSwrestler)
str(HSwrestler)

## Example: Backward elimination

reg.all <- lm(HWFAT ~ AGE + HT + WT + ABS +
                TRICEPS + SUBSCAP)
summary(reg.all)$coefficients

reg.m1 <- lm(HWFAT ~ AGE + WT + ABS + TRICEPS + SUBSCAP)
summary(reg.m1)$coefficients

reg.m2 <- lm(HWFAT ~ AGE + WT + ABS + TRICEPS)
summary(reg.m2)$coefficients

reg.m3 <- lm(HWFAT ~ AGE + ABS + TRICEPS)
summary(reg.m3)$coefficients


## Example: Backward elimination

# Alternately, the function `drop1()` can be used in `R`:
drop1(lm(HWFAT ~ AGE + HT + WT + ABS + TRICEPS + 
           SUBSCAP), test="F")

drop1(lm(HWFAT ~ AGE + WT + ABS + TRICEPS + 
           SUBSCAP), test="F")

drop1(lm(HWFAT ~ AGE + WT + ABS +TRICEPS), 
      test="F")

drop1(lm(HWFAT ~ AGE + ABS +TRICEPS), test="F")



## Example: Forward selection
add1(lm(HWFAT~1), scope=(~.+ AGE + HT + WT + 
                           ABS + TRICEPS + 
                           SUBSCAP), test="F")

add1(lm(HWFAT~ABS), 
     scope=(~.+ AGE + HT + WT + TRICEPS + SUBSCAP),
     test="F")

add1(lm(HWFAT~ABS+TRICEPS),
     scope=(~.+ AGE + HT + WT + SUBSCAP),
     test="F")

add1(lm(HWFAT~ABS+TRICEPS+AGE), 
     scope=(~.+ HT + WT + SUBSCAP), test="F")

summary(lm(HWFAT~ABS+TRICEPS+AGE+HT))$coefficients

summary(lm(HWFAT~ABS+TRICEPS+AGE+WT))$coefficients

summary(lm(HWFAT~ABS+TRICEPS+AGE+SUBSCAP))$coefficients


## Example: Adjusted $R^2$
str(HSwrestler[,-c(8,9)])
library(leaps)
a <- regsubsets(as.matrix(HSwrestler[,-c(7,8,9)]), 
                HSwrestler[,7])

summary(a)
summary(a)$adjr2
max(summary(a)$adjr2)
which.max(summary(a)$adjr2)

## Example Mallows Cp
summary(a)
summary(a)$cp
par(pty="s")
plot(2:7, summary(a)$cp, ylim=c(2,7), xlab="p",
     ylab="Cp", pch=16); abline(a=0, b=1, col='blue');
points(4,summary(a)$cp[3], col='red', pch=16)


## Example: AIC
library(MASS) # For function stepAIC()
reg.all <- lm(HWFAT ~ AGE + HT + WT + ABS + 
                TRICEPS + SUBSCAP)
mod.aic <- stepAIC(reg.all, direction="both",
                   scope=(~.+SUBSCAP+TRICEPS+
                            ABS+WT+HT+AGE), k=2)

mod.aic


## Example: BIC
mod.bic <- stepAIC(reg.all, direction="both",
                   scope=(~.+SUBSCAP+TRICEPS+
                            ABS+WT+HT+AGE),
                   k=log(length(HWFAT)))

mod.bic

