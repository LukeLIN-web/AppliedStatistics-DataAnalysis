## AppliedStatistics-DataAnalysis

每周写作业， 还要讨论4-5个问题。 meeting。  markdown， 上传 pdf和code. exam based on R。  开卷 ，两次考试。   project，  4-5个学生。  10月27日proposal，  11月26日 report， 12月4-8日 展示。 

第一场考试将于 10 月 22 日星期六上午 9:00 至中午 12:00 您必须带上电脑，别忘了带上充电器。您可以使用我们在课堂上解决的笔记、演示文稿、书籍和练习

考试将在上午 9:00 在 Blackboard 上的作业/考试下发布，您必须在中午 12:00 通过 Blackboard 提交您的解决方案。您必须提交两份文件，一份包含您的答案的 pdf 和一份包含 R 代码的脚本。该脚本可以是一个 Rmarkdown 文件。确保脚本运行并产生所需的结果。

考试涵盖视频 24，实验设计 1。请记住，仅靠代码不是问题的答案。您对获得的结果的评论和讨论您选择的程序是您回答的基础。 

### 第一节课

R是大小写敏感的。考试会考一些函数. 

 ls（）显示没有东西, why?

ts : time series

data frame: 一张表, 允许数据有不同的mode

Subset, 取一个子集.

apply :   在数据上apply函数,   用不同维度来展示数据.

sweep  : 也是apply函数, 比如求和mean的距离. element wise计算.

within, 新增一列, 可以通过别的列计算出来.

#### 常见错误

访问list中的元素,  需要 list1[[1]] , [] 会访问到list而不是元素, 或者用 list1$item1

%*%  矩阵乘法,  第八题, 搞不懂 reciprocals

#### 数据结构

第一种：向量([vector]

向量可以储存数值型（numeric)、逻辑型（logical)和[字符型](https://www.zhihu.com/search?q=字符型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"28586790"})（character）三种类型的数据，所有向量的值属性要相同

没有oop写法, 写法都是apply(list, args) 而不是 list.apply

```{r}
l=c(3:8)
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

```R
library(Mass)
 human <- read.table('Human_data.txt', header = T) # 写个T就行不用写True
str(human)
```

#### Quantiles function
如果F是连续而且严格递增, 那么Q就是F的反函数.
If the quantile is not unique, we take the smallest value
也有可能没有.

Quantiles ,  deciles 分成10份,  quantiles 分成1/4.  

P(X <=z) = q

如果x 加F(x) 不加的话, 那就取左边的.  也就是说quantile不唯一, 我们取最小的z.

如果x不变,  Fx突变的话,  可能就没有z满足F(z) = q . 

qqplot

#### Quantiles plots 

用来比较一个dataset和分布的.

如果两个分布属于同一个location and scale family, 那么graph 大约是一条直线. 

### lec4

data frame 用$来取列而不是用dot

```R
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

7) 由于p值接近她设定的置信度，工程师决定重新抽取一个大小为20的样本

置信区间为95%, a = 平均值 - z *标准差 ,  z = 1.96

https://m.medsci.cn/scale/show.do?id=972b231389 可以帮你算.

https://www.jianshu.com/p/cb53a7dc00e3 讲很好.

### lec5

估计

方法1 : 方差不知道的估计,  sn平方来估计  , 有高斯密度估算

1-a 就是置信度.

### t检验

 t分布.估计平均值. The t distribution arises as the sampling distribution of the (empirical) mean μˆn when the data come from a normal distribution with unknown variance.  适用于平均值是正太, 方差不知道的情况`pt()`

两列数字平均值大小比较, 也可以用t检验. 因为每列都数据足够多. 所以中心极限定理  central limit theorem 证明这是合理的.  想要知道是否女生更多, 所以就单边alternative. 

假设就是,  假设 sample distribution 是接近Gaussian的, 样本数量够大的时候就是合理假设.  即使不是正态分布也可以.

```
t.test(Theatre[Sex == 0], Theatre[Sex == 1], alternative = 'less')
```

p很小, 就拒绝原null假设.  

同一个subject, 前后两个对比 , 用paired test. 

`t.test(Theatre_ly, Theatre, paired = TRUE)`  或者也可能用 wilcoxon test.

t.test要求是正态分布, 可以先用 qqplot  `shapiro.test(data2$dif)`来检验一下.  

 wilcoxon test. 要求分布是连续而且对称.

power.test

有的给定power, 有的给定confidence 

`z = qt(0.01,df=9) #degree of freedom就是 n-1`

 \* pt()返回值是正态分布的分布函数(probability)
\> * 函数qt()的返回值是给定概率p后的下百分位数(quantitle)
\> * rt()的返回值是n个正态分布随机数构成的向量

### lec6

拒绝了正确的, 就是I类error, 接受了错误的, 是第二类错误.

https://rpsychologist.com/d3/nhst/  可以看一些可视化.

```R
var.test(sample1,sample2)
with(example,)
boxplot
加点可以用points(Hemo ~ jitter(Sulfa, amount = 0.05), data = q3.df, pch = 16, col = 'blue')
```

Var(X-Y) = Var(X) + Var(Y) - 2 Cov(X,Y)

样本数量多, 就用t检验. 

p 太小了, 就拒绝.

 table可以看各个类型的数量.

#### non-parametric

如果不为正态,那就是在PPT , V18 , This non-parametric test is useful when the assumption of normality is not justified.

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

chisq 测出来p 很小 ,  意味着可以拒绝原假设

#### 非参数检验

卡方检验属于非参数检验，由于非参检验不存在具体参数和总体正态分布的假设，所以有时被称为**自由分布检验**。

参数和非参数检验**最明显的区别**是它们使用**数据的类型**。

非参检验通常将被试分类，如民主党和共和党，这些分类涉及名义量表或顺序量表，无法计算平均数和方差。

患者接受的治疗和改善水平看上去存在某种关系（ p<0.01 ）这里的**p值表示从总体中抽取的样本行变量与列变量时相互独立的概率**，由于p的概率值很小，所以我们拒绝了治疗类型和治疗结果相互独立的原假设。

```R
wilcox.test(Culture, mu=216) # 非参数one sample test
wilcox.test(Theatre[Sex == 0],Theatre[Sex == 1], alternative = 'less') # two sample
```

#### binomial distribution

The expected value and variance for this distribution are given by E(nA) = np, Var(nA) = np(1 − p).

If n and p are such that np ≥ 5 and n(1 − p) ≥ 5 the binomial distribution can be approximated by the normal distribution.

proportions test的假设, sample足够大, 

```R
prop.test(n.A,n,p_0)
prop.test(as.matrix(car.accidents))
```

tapply : tapply() is used **to apply a function over subsets of a vector**. It is primarily used when we have the following circumstances

两边有括号可以赋值后再直接print.

```R
q2.df$origin <- factor(q2.df$origin, labels = c('Am','Eu','Jap'))# 可以重命名.
```

怎么求比例?

Build a table with the proportions with respect to the total number of cases for each gender

reating proportion tables. 

```R
prop.table(q2.tbl,1) # 按行分
```

是否两个分布是一样的? 用什么检验? 为什么? 应该满足什么条件? 必考. 

Chi-square distribution approximation  要求 至少要5个样本. 需要做验证. 

### lec9方差分析

基本的idea是decompose.

factors 也经常被叫做 treatments. An-o-va , 全称是 analysis of variance.

anaova, 就是比较多个means, 和多个factors的等级相关.  我们可以分辨多个factors 一起变化的时候的效果. 

比如, 我们有两个level, 第一个level做n1 次实验, 第二个level做n2次实验.  

上面一个bar,就是取平均数, 点, 就是取sum.

SST :  total sum of squres.   不同组的方差

SSE:   error sum of squres.   估计方差. 对平均值的方差之和. 就是组内方差. 不可解释方差, 是SSE. 也叫residual sum of squares

如果treatment没有effct, 那么 SST 就等于SSE .  

如果有, 那么SSE < SST.  如果所有值都等于 treatment 平均值, SSE就是0 . 

```R
anova 可以获得一个table 
mod0 = lm(stopdis - tire, data = tire)
model.table()
Anova model:
model1 <- aov(Hemo ~ fSulfa, data = q3.df) # aov可以fit一个model,p很小的时候, 就认为是有difference的.
 # Mean responses with standard errors
(means <- model.tables(model1, 'means', se = TRUE))
```

   SST和SSE的差叫做  treatment  sum of squares, 叫做SSA.可解释 方差, 就是SSA.

有k个level, 有k-1个df. 自由度

Analysis of Variance (Anova)  one -way Anova 包括一些假设

估计方差, 来自 anova table ,deviation 一个个开根号.

Anova table是啥

Plot the diagnostic charts and comment on them.

diagnostic就是model1 plot的第一张图. 

怎么做 Levene’s test and Shapiro-Wilk. 

写出equation, note anova-pdf中有lm的例子, 但是什么时候是lm拟合? 什么时候不用lm拟合? 

SumSq 就是sum of squre, 

The F value is the ratio MSA/MSE and the last column labeled Pr(>F) is the probability of exceeding the calculated F-value when the null hypothesis is true, i.e. it is the p-value for the F test.

#### Tukey HSD

什么是Tukey’s HSD procedure , 怎么做  pairwise comparisons 

方差分析的3个假设条件是：1、总体服从正态分布，2、个体是相互独立，3、组间方差相等

在PPT V25 

If the result of the F test is to reject the null hypothesis of no treatment effects one is naturally interested in determining where the difference lies. For this, it becomes necessary to compare the individual groups.

```R
with(Tire, pairwise.t.test(a, b , p.adjust.method= 'bonferroni'))
tky= TukeyHSD(mod1)
plot(mod1.tky)
```

 p value higher than 5%, so that they don't have significant difference. p value is smaller than 1%, they have significant difference.

画图 If the intervals include zero,  then the difference is not significant. 

https://wiki.mbalib.com/wiki/HSD%E6%A3%80%E9%AA%8C%E6%B3%95

```R
attach, 就不用加frame 的前缀了, 可以直接引用.
```

