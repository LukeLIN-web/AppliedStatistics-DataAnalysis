## AppliedStatistics-DataAnalysis

每周写作业， 还要讨论4-5个问题。 然后meeting。 

markdown， 上传 pdf和code

exam based on R。  开卷 ，两次考试。 

project，  4-5个学生。  10月20日proposal，  11月26日 report， 12月4-8日 展示。 

### 第一节课

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

### lec3

graph in R

```R
boxplot
type = p就是point,  =l就是 lines.
abline, legend, 
axis
par(bg = , col= , col.axis= , ) 
Par( mfrow(c(m,n))
Par( mfcol(c(m,n))
locator (n=512,type ="n")
identify (x,...)
```

problemlist

```
library(Mass)
```

#### Quantiles function
如果F是连续而且严格递增, 那么Q就是F的反函数.
If the quantile is not unique, we take the smallest value
也有可能没有.

Quantiles ,  deciles 分成10份,  quantiles 分成1/4.   就是

P(X <=z) = q

如果x 加F(x) 不加的话, 那就取左边的.  也就是说quantile不唯一, 我们取最小的z.

如果x不变,  Fx突变的话,  可能就没有z满足F(z) = q . 

qqplot

#### Quantiles plots 

可以用来比较一个dataset和分布的.

如果两个分布属于同一个location and scale family, 那么graph 大约是一条直线. 

### lec4

data frame 用$来取列而不是用dot

```R
# using subset function
newdata <- subset(mydata, age >= 20 | age < 10,
select=c(ID, Weight))
```

#### apply

难点在于 margin 维度的区别. 1为行，2为列。

R的apply函数怎么用？ - 李大猫的回答 - 知乎 https://www.zhihu.com/question/39843392/answer/83364326

#### problemlist3

正态分布

工程师总是为这些测试设定98%的信心水平。在这个水平上为测试找到一个拒绝区域。检验统计量的值是在拒绝区域之内还是之外？你的结论是什么？

1-alpha=0.98,  alpha = 0.02  双边拒绝域.  方差未知, 置信区间计算在中文书的164页.

u = 500

6）用R的命令进行这个测试，看看你得到的P值。你的结论是什么？

```
 0.02416
```

7) 由于p值接近她设定的置信度，工程师决定重新抽取一个大小为20的样本，得到的数值如下

95%, a = 平均值 - z *标准差 ,  z = 1.96

https://m.medsci.cn/scale/show.do?id=972b231389 可以帮你算.

https://www.jianshu.com/p/cb53a7dc00e3 讲 的很好.

### lec5

估计

方法1 : 方差不知道的估计,  sn平方来估计  , 有高斯密度估算

方法2 : t分布.估计平均值

1-a 就是置信度.

随机采样

采样分布

power.test. 

有的给定power, 有的给定confidence 

z = qt(0.01,df=9) #degree of freedom就是 n-1

 \* pt()返回值是正态分布的分布函数(probability)
\> * 函数qt()的返回值是给定概率p后的下百分位数(quantitle)
\> * rt()的返回值是n个正态分布随机数构成的向量

### lec6

拒绝了正确的, 就是I类error, 接受了错误的, 是第二类错误.

https://rpsychologist.com/d3/nhst/  可以看一些可视化.

```
var.test(sample1,sample2)
with(example,)
boxplot
```

Var(X-Y) = Var(X) + Var(Y) - 2 Cov(X,Y)

样本数量多, 就用t检验. 

p 太小了, 就拒绝.

R 语言 table可以看各个类型的数量.

#### non-parametric

如果不为正态,那就是 在PPT , V18 , This non-parametric test is useful when the assumption of normality is not justified.

### lec8

cut 函数, 设置break ,可以告诉你high有多少,  right = T就是左开右闭, 

```R
bmi.fac = cut(human$bmi, c(-Inf,20,25,30,Inf),labels = c("underweight","normal","overweight","obese"))
```

Inf来表示低于和高于. 

#### Contingency table 

在统计学中，列联表是一种矩阵形式的表格，显示变量的频率分布。

自由度 v= (m-1)(n-2)

为什么用chisq不用fisher? 什么情况下用chisq?

```R
chisq.test(stdt.tab)
fisher.test(titanic.table[1:2,1:2]) # 通常用在2*2, 样本很少
mytable <- xtabs(~Treatment+Improved, data = Arthritis) # 生成列联表
```

卡方检验属于非参数检验，由于非参检验不存在具体参数和总体正态分布的假设，所以有时被称为**自由分布检验**。

参数和非参数检验**最明显的区别**是它们使用**数据的类型**。

非参检验通常将被试分类，如民主党和共和党，这些分类涉及名义量表或顺序量表，无法计算平均数和方差。

患者接受的治疗和改善水平看上去存在某种关系（ p<0.01 ）这里的**p值表示从总体中抽取的样本行变量与列变量时相互独立的概率**，由于p的概率值很小，所以我们拒绝了治疗类型和治疗结果相互独立的原假设。

#### binomial distribution

The expected value and variance for this distribution are given by E(nA) = np, Var(nA) = np(1 − p).

If n and p are such that np ≥ 5 and n(1 − p) ≥ 5 the binomial distribution can be approximated by the normal distribution.

```R
prop.test(n.A,n,p_0)
prop.test(as.matrix(car.accidents))
```

tapply : tapply() is used **to apply a function over subsets of a vector**. It is primarily used when we have the following circumstances

两边有括号可以赋值后再直接print.

怎么求比例?

Build a table with the proportions with respect to the total number of cases for each gender

reating proportion tables. 

是否两个分布是一样的? 用什么检验? 为什么? 应该满足什么条件?

Chi-square distribution approximation  要求 至少要5个样本, 



### lec9

```R
anova 可以获得一个table
mod0 = lm(stopdis - tire, data = tire)
model.table()
```

SSE   

估计方差  . MSE

one -way Anova 包括一些假设

