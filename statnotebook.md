## AppliedStatistics-DataAnalysis

每周写作业， 还要讨论4-5个问题。 然后meeting。 

markdown， 上传 pdf和code

exam   based on R。  开卷 ，两次考试。 

project，  4-5个学生。  10月20日proposal，  11月26日 report， 12月4-8日 展示。 





#### 第一节课

R是大小写敏感的。

考试会考一些函数. 



问题

1. 为啥明明有东西， ls（）显示没有东西。





ts : time series

data frame: 一张表, 允许数据有不同的mode

Subset, 取一个子集.

apply :   在数据上apply函数,   用不同维度来展示数据.

sweep  : 也是apply函数, 比如求和mean的距离. element wise计算.

within, 新增一列, 可以通过别的列计算出来.

#### 常见错误

访问list,  需要 list1[[1]] , [] 会访问到list, 

%*%  矩阵乘法, 

第八题, 搞不懂 reciprocals

#### 数据结构

第一种：向量([vector]

向量可以储存数值型（numeric)、逻辑型（logical)和[字符型](https://www.zhihu.com/search?q=字符型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"28586790"})（character）三种类型的数据，所有向量的值属性要相同

没有oop写法, 写法都是apply(list, args) 而不是 list.apply

```{r}
l =c(3:8)
myFun=function(x){10^x}
lapply(l,myFun)
```

我花了非常多的时间去找怎么append, append不进去很难受.

  results <- append(results, abs(I - result)) 太抽象了这玩意, 这个赋值和append要同时用.



##### list

负下标表示扣除相应的元素后的子集，`x[]`表示取`x`的全部元素作为子集

```
x <- c(1,4,6.25)
x[] <- 999
x
## [1] 999 999 999
x[0]是一种少见的做法， 结果返回类型相同、长度为零的向量， 如numeric(0)。 相当于空集。
设向量x长度为, 则使用正整数下标时下标应在中取值。 如果使用大于的下标， 读取时返回缺失值，并不出错。 给超出的下标元素赋值， 则向量自动变长， 中间没有赋值的元素为缺失值。 
下标可以是与向量等长的逻辑表达式
x <- c(1,4,6.25)
x[x > 3]
## [1] 4.00 6.25
```

mode(list1[[2]]) 这样是数字, mode(list1[2])是list .https://blog.51cto.com/u_15127692/3316141

[] extracts a list, [[]] extracts elements within the list.