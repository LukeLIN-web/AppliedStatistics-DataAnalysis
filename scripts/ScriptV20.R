## | Contingency Tables

# Statistical analysis of contingency tables 
## Contingency tables 

library(vcd)
data("Titanic")
str(Titanic)
class(Titanic)

(titanic.table <- apply(Titanic, c(4, 2), sum))
addmargins(titanic.table)
mosaicplot(titanic.table[1:2,1:2], 
           col = c('dodgerblue','skyblue1'),
           main = 'Survival in the Titanic')


library(knitr)
library(kableExtra)

kable(titanic.table, booktabs = T, 
      caption = 'Observed values') %>%
  kable_styling(bootstrap_options = "hover",
                full_width = F, position = "center",
                latex_options ='striped')

## Expected values
tab.exp <- data.frame(Male = c(1171.89,559.11,1731),
                      Female = c(318.19,151.81,470),
                      Total = c(1490,711,2201))
row.names(tab.exp) <- c('No','Yes','Total')
kable(tab.exp, booktabs = T, escape = FALSE) %>%
  kable_styling(bootstrap_options = "hover",
                full_width = F, position = "center",
                latex_options ='striped') %>%
  column_spec(1,width='7em', background ='white') %>%
  column_spec(2:4, width = '5em') 

## Observed values
kable(titanic.table, booktabs = T, 
      caption = 'Observed values') %>%
  kable_styling(bootstrap_options = "hover",
                full_width = F, position = "center",
                latex_options ='striped')





## Test
(chi.st <- sum((tab.exp[1:2,1:2]
            -titanic.table[1:2,1:2])^2/tab.exp[1:2,1:2]))
1-pchisq(chi.st,1)

chisq.test(titanic.table[1:2,1:2],correct = FALSE)
chisq.test(titanic.table[1:2,1:2])


# Contingency Tables: Independence
library(MASS)
str(survey)
(stdt.tab <- with(survey,table(Smoke,Exer)))
stdt.tot <- cbind(stdt.tab, 
                  Total = apply(stdt.tab, 1, sum))
(stdt.tot <- rbind(stdt.tot, 
                  Total = apply(stdt.tot, 2, sum)))

chisq.test(stdt.tab)

# Fisher's exact test
## Small samples: Fisher's exact test


fisher.test(titanic.table[1:2,1:2])
fisher.test(stdt.tab)



