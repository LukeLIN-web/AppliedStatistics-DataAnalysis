##   Objects and Data III: Data frames and lists


# The command for creating a data frame is
data.frame (data1, data2, ...)

(dframe1 <- data.frame(int = 2001:2010,
                    real = rnorm(10),
                    lettrs = letters[sample.int(26,10)],
                    logic = (1:10)%%2==0))

dframe1 <- data.frame(int = 2001:2010,
                    real = rnorm(10),
                    lettrs = letters[sample.int(26,10)],
                    logic = (1:10)%%2==0,
                    v = c(1,-1,2))

dframe1$int
dframe1['int']

str(dframe1$int)
str(dframe1['int'])

lettrs <- c('a','e','i','o','u')
attach(dframe1)
int 
real
lettrs

ls()
rm(lettrs)
lettrs

search()

detach(dframe1)
search()


## Lists

(vec1 <- letters[5:10])
(vec2 <- 1:10)
(mat3 <- matrix(rnorm(12),ncol=3))
(mixed.list <- list(item1 = vec1, 
                    item2 = vec2,
                    item3 = mat3, 
                    item4 = cars))

mixed.list$item1
mixed.list[[1]]

mixed.list$item3[,2]
mixed.list[[3]][,2]
mixed.list$item4$dist

length(mixed.list)
mixed.list[[5]] <- c('The', 'new', 'element')

mixed.list[[length(mixed.list)+1]] <-
                c('the last')

mixed.list[[4]] <- -5:10
mixed.list[[2]] <- NULL
mixed.list

names(mixed.list)
names(mixed.list)[4:5] <- c('item.new','item final')
names(mixed.list)

fit1 <- lm(Fertility ~ Education, data=swiss)
str(fit1)










