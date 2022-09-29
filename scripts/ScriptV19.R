# | Tables in R


# Reading Files
## Reading .txt files
options(width=60)
humans1 <- read.table('data/Human_data.txt', 
                      header = TRUE)
dim(humans1)
str(humans1, vec.len = 2)
head(humans1, n=4)

## Reading `.csv` files
humans2 <- read.csv('data/Human_data.csv')
str(humans2, vec.len = 2)
head(humans2)

## Reading `.xlsx` files
library(gdata)
humans <- read.xls('data/Human_data.xlsx')
str(humans, vec.len = 2)
head(humans)


## Selecting data 
names(humans)
humans$Gender[1:5]
fem_h<-humans[humans$Gender=='F', ]
str(fem_h, vec.len = 2)
fem_h<-humans[humans[,2]=="F" , ]
fem_h <- subset(humans,Gender=='F') 
str(fem_h, vec.len = 2)

fem_h_AGE<-humans[humans$Gender=='F' 
                  & humans$age > 35,3:7]
str(fem_h_AGE, vec.len = 2)

fem_h_AGE<- subset(humans, Gender=='F' & age > 35, 
                   select = 3:7)


## Factors 

blood_factor<-factor(humans$blood_type, 
                     labels=c("O", "A", "B", "AB"))
head(blood_factor)

# Tables using cut
Sal_class <-cut(humans$Salary,
      breaks=c(min(humans$Salary),5000,12000, (max(humans$Salary)+1)), 
                labels=c("low","med","high"), right=F )
str(Sal_class)
head(Sal_class)


(freq_gender_sal<- table(Sal_class,humans$Gender))
(f_ocup_age<-table(humans$Ocupation,humans$age))
age_class <- cut(humans$age, 
                 breaks = c(10,20,30,40,50,60,70), 
                 right = FALSE)
(f_ocup_age2<-table(age_class,humans$Ocupation))

## Storing the new information
humans$Salary_class <- Sal_class
humans <- within(humans, Salary_class <-cut(humans$Salary,
              breaks=c(min(humans$Salary),5000,12000, 
              (max(humans$Salary)+1)),
              labels=c("low","med","high"), right=F ))
str(humans, vec.len = 2)

write.csv(freq_gender_sal,file="data/results.csv")

# Summaries of tables 

# Proportion with respect of the total sample
(freq_gender_sal_prop <- freq_gender_sal/
    sum(freq_gender_sal))
prop.table(freq_gender_sal)

### Proportion of females and males
(prop_fem<-sum(freq_gender_sal[,1])/sum(freq_gender_sal))
(prop_male<-1-prop_fem)
prop.table(table(humans$Gender))


### Proportion of salary classes
sum(freq_gender_sal[1,])/sum(freq_gender_sal)
sum(freq_gender_sal[2,])/sum(freq_gender_sal)
sum(freq_gender_sal[3,])/sum(freq_gender_sal)
prop.table(table(Sal_class))

## **Proportions for females and males**
 
freq_gender_sal

(sum.col <- apply(freq_gender_sal,2,sum))
colSums(freq_gender_sal)

# A naive (and wrong!) way to attempt to do this would be to write `freq_gender_sal/sum.com`. 

(mat1 <- matrix(c(1,2,3,4,5,6), ncol=2))
mat1/c(1,2)

# To obtain what we want, we need the function `sweep`. 

sweep(mat1,2,c(1,2),'/')
(freq_gender_sal1 <-sweep(freq_gender_sal,2,sum.col,'/'))

prop.table(freq_gender_sal, margin = 2)


# Graphical representations
plot(1:3, freq_gender_sal1[,1], col=3, 
     ylim=c(0.25,0.42), pch=19, type='b',
     xaxp = c(1,3,2), xlab='Salary Level',
     ylab = 'Proportion', 
     main = 'Proportion of individuals in each income level',
     xaxt='n')
axis(1,at = c(1,2,3),labels = c('Low','Med','High'))
points(1:3, freq_gender_sal1[,2], col=4, 
       type='b', pch=19)
legend('top',c('F','M'), pch=c(16,16),col=c(3,4))

barplot(t(freq_gender_sal1), beside = TRUE,   
        legend.text = TRUE,
        col=c('lightsteelblue','dodgerblue3'), 
        args.legend = list(x='top'))

## Mosaic plots

mosaicplot(freq_gender_sal1, xlab='Salary class', ylab='gender', 
           main = 'Salary Class by Gender', col = c('azure2','darkseagreen3'))

mosaicplot(~ Sal_class + humans$Gender, 
           xlab='Salary class', ylab='gender', 
           main = 'Salary Class by Gender', 
           col = c('azure2','darkseagreen3'))


library(vcd)
mosaic(~ Salary_class + Gender, data = humans, 
       main = 'Salary Class by Gender', 
       highlighting = 'Gender',
       highlighting_fill = c('azure2','darkseagreen3'))


mosaic(~ Gender +  Salary_class, data = humans, 
       main = 'Salary Class by Gender', 
       highlighting = 'Gender',
       highlighting_fill = c('azure2','darkseagreen3'))



# More complicated plots
mosaic(Survived ~ ., data = Titanic)

# example("mosaic")



