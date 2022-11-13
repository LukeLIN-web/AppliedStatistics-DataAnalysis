#| Applied Statistics and Data Analysis
#| Multiple Linear Regression 1


# Example

str(mtcars, vec.len = 2)
head(mtcars, 3)

data.cars <- mtcars[,c('mpg','disp','hp',
                       'drat','wt')]
head(data.cars)

pairs(data.cars, panel=panel.smooth)

model1 <- lm(mpg ~ hp + wt + disp + drat, 
             data = data.cars)

summary(model1)

with(data.cars, cor(hp,disp))

head(model.matrix(model1),10)

model2 <- lm(mpg ~ hp + wt + drat, data = 
               data.cars)
summary(model2)

round(coef(model1),3); round(coef(model2),3)

model3 <- lm(mpg ~ hp + wt , data = data.cars)
summary(model3)

model4 <- lm(mpg ~ disp + wt, data = data.cars)
summary(model4)

model4a <- lm(mpg ~ wt + disp, data = data.cars)
summary(model4a)

## The hat matrix
X <-model.matrix(model4)
n <-nrow(X)
p <-ncol(X)
H <- X%*%solve(t(X)%*%X)%*%t(X)
hii <-diag(H)

sum(hii)

plot(hii,type = 'h', ylim = c(0,.2))





