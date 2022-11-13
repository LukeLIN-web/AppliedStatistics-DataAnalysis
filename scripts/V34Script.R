#  | STAT 210
#| Applied Statistics and Data Analysis
#| Multiple Linear Regression 2   
#| Anova and Hypothesis Tests



data.cars <- mtcars[,c('mpg','disp','hp',
                       'drat','wt')]
model1 <- lm(mpg ~ hp + wt + disp 
             + drat, data = data.cars)

(model1.anova <- anova(model1))

(SSE = model1.anova$`Sum Sq`[5])
(SSR = sum(model1.anova$`Sum Sq`[1:4]))

(MSE = SSE/27)
(MSR = SSR/4)

(SST = SSE + SSR)

(Fobs <- MSR/MSE)

summary(model1)

1-pf(Fobs,4,27)


model3 <- lm(mpg ~ hp + wt , data = data.cars)
anova(model3)
model3a <- lm(mpg ~ wt + hp, data=data.cars)
anova(model3a)



## Hypotheses Tests: Example

model1.anova
model4 <- lm(mpg ~ disp + wt, data = data.cars)

(model4.anova <- anova(model4))

(SSRf = sum(model1.anova$`Sum Sq`[1:4])) 
                            # Complete model
(SSRr = sum(model4.anova$`Sum Sq`[1:2])) 
                            # Reduced model
(SSRnv = SSRf-SSRr)
(SSEf = model1.anova$`Sum Sq`[5]) 
                            # Error sum of squares

(Fcomp <- (SSRnv/2)/(SSEf/27))

pf(Fcomp, 2, 27, lower.tail=FALSE)
anova(model4,model1)

                                              
                                              
par(mfrow=c(2,2))
plot(model1)
par(mfrow=c(1,1))

par(mfrow=c(2,2))
plot(model3)
par(mfrow=c(1,1))

