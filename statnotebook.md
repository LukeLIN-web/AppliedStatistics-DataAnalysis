## AppliedStatistics-DataAnalysis

每周写作业， 还要讨论4-5个问题. 作业上传 pdf和code. exam based on R。  开卷 ，两次考试。   project，  4-5个学生。   11月27-30 考试,  11月30日 report， 12月4-8日 展示。   

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

第一种：向量[vector]

向量可以储存数值型（numeric)、逻辑型（logical)和[字符型](https://www.zhihu.com/search?q=字符型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"28586790"})（character）三种类型的数据，所有向量的值属性要相同

没有oop写法, 写法都是apply(list, args) 而不是 list.apply

```{r}
l=c(3:8)
myFun=function(x){10^x}
lapply(l,myFun)
```

我花了非常多的时间去找怎么append, append不进去很难受.

  `results <- append(results, abs(I - result)) `太抽象了这玩意, 这个赋值和append要同时用. 

##### list

负下标表示扣除相应的元素后的子集，`x[]`表示取`x`的全部元素作为子集

```
x <- c(1,4,6.25)
x[] <- 999
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

### lec3 graph in R

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

#### Quantiles plots 

`qqplot` ,用来比较一个dataset分布的.

如果两个分布属于同一个location and scale family, 那么graph 大约是一条直线. 

### lec4

data frame 用$来取列而不是用dot

```R
newdata <- subset(mydata, age >= 20 | age < 10,select=c(ID, Weight))
```

#### apply

难点在于 margin 维度的区别. 1为行，2为列。

R的apply函数怎么用？ - 李大猫的回答 - 知乎 https://www.zhihu.com/question/39843392/answer/83364326

#### problemlist3

工程师总是为这些测试设定98%的信心水平。在这个水平上为测试找到一个拒绝区域。检验统计量的值是在拒绝区域之内还是之外？你的结论是什么？

1-alpha=0.98,  alpha = 0.02  双边拒绝域.  方差未知, 置信区间计算在中文书的164页.

6）用R的命令进行这个测试，看看你得到的P值。你的结论是什么？

7) 由于p值接近她设定的置信度，工程师决定重新抽取一个大小为20的样本

置信区间为95%, a = 平均值 - z *标准差 ,  z = 1.96

https://m.medsci.cn/scale/show.do?id=972b231389 可以帮你算.

https://www.jianshu.com/p/cb53a7dc00e3 讲很好.

### lec5

估计   方法1 : 方差不知道的估计,  sn平方来估计, 有高斯密度估算

1-a 就是置信度.

### t检验

 t分布.估计平均值. The t distribution arises as the sampling distribution of the (empirical) mean μˆn when the data come from a normal distribution with unknown variance.  适用于平均值是正太, 方差不知道的情况`pt()`

两列数字平均值大小比较, 也可以用t检验. 因为每列都数据足够多. 所以中心极限定理  central limit theorem 证明这是合理的.  想要知道是否女生更多, 所以就单边alternative. 

假设就是,  假设 sample distribution 是接近Gaussian的, 样本数量够大的时候就是合理假设.  即使不是正态分布也可以.

如果只有29个点的话, 要检查normality.

```
t.test(Theatre[Sex == 0], Theatre[Sex == 1], alternative = 'less')
power.test
```

p很小, 就拒绝原null假设.  

同一个subject, 前后两个对比 , 用paired test. 

`t.test(Theatre_ly, Theatre, paired = TRUE)`  或者也可能用 wilcoxon test.

t.test要求是正态分布, 可以先用 qqplot 和  `shapiro.test(data2$dif)`来检验一下.  

 wilcoxon test. 要求分布是连续而且对称.

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
boxplot(Critic_Score~Platform, data = sale) #直接plot也可以
加点可以用points(Hemo ~ jitter(Sulfa, amount = 0.05), data = q3.df, pch = 16, col = 'blue')
```

Var(X-Y) = Var(X) + Var(Y) - 2 Cov(X,Y)

样本数量多, 就用t检验.   p 太小了, 就拒绝.

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

proportions test的假设, sample足够多

```R
prop.test(n.A,n,p_0)
prop.test(as.matrix(car.accidents))
```

tapply : tapply() is used **to apply a function over subsets of a vector**. 

两边加上括号可以赋值后直接print.

```R
q2.df$origin <- factor(q2.df$origin, labels = c('Am','Eu','Jap'))# 可以重命名.
```

怎么求比例?

Build a table with the proportions with respect to the total number of cases for each gender.  reating proportion tables. 

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
anova 可以获得一个table , Anova 会获得一个 (Type II tests)
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

#### 均方误差

就是anova table中的Mean Sq  

The estimated variance for the errors is the mean square error

#### diagnostic plots

Plot the diagnostic charts and comment on them.  diagnostic是四张图. 

```r
par(mfrow=c(2,2))
plot(model5)
par(mfrow=c(1,1)) # 是为了不影响后面, 后面不会变成4张图. 
```

可以看一下是怎么分析Draw diagnostic plots and discuss the results.的

评论

1. All the plots look reasonable. In the first plot, the distribution of the residuals looks random and approximately symmetric. The quantile plot shows some departures at the tails, but in general seems reasonable. We can confirm this using the Shapiro-Wilk test on the standardized residuals:

2. In general, the plots look good. The quantile plot is partivularly good, so there are no doubts about normality. The only point that may raise cause for concern is the assumption of homoscedasticity, since the scale-location lot shows a small increasing tendency. We can check this with a test.

Do the diagnostic plots for this model and comment ,  normal性质 好或者不好.  

3. The third plot also looks reasonable although a slight decreasing pattern can be seen in the local regression line. To confirm whether this is significant, we use the ncv test.   Finally, the fourth plot shows one point with high leverage and large value for Cook’s distance (close to the contour line), which is point 29. This point should be checked in a more thorough study of the regression model.

##### 如果不好

要学会怎么用英语答题.    所有的诊断图都有问题。

In this case all the diagnostic plots have issues. 

qqplot:  

1. The quantile plot has some very large values on the right tail.  QQ图的右尾部有一些非常大的数值。

residuals against fitted values: 

1.  the majority of the residuals are negative, the red line is far from 0 and is not horizontal, and the residuals are not homogeneously spread in the plot. 在残差与拟合值的对比中，大部分的残差是负的，红线离0很远，而且不是水平的，残差在图中的分布也不均匀。
2. shows a quadratic pattern and the residuals are not symmetrically distributed. The model is not adequate.残差与拟合图显示了一个二次方模式，残差不是对称分布的。该模型是不充分的。
3.  unequal dispersion and more positive values at the center 

The scale-location plot:

1.  shows an increasing pattern for the dispersion of the data 标度-位置图显示了数据的分散性在增加。

The residuals vs leverage plot:

1.  has one point with a very large value for leverage and high residual. This would not be an acceptable model. 残差与杠杆的关系图有一个点的杠杆值非常大，而残差却很高。这将不是一个可接受的模型。

##### 寻找outlier

怎么找:  所有的图中都标记出来.  它是残差最大的一个点，在量化图中，它离其他的点和参考线都很远。在比例-位置图中，该点对应的数值大于2。

例子: 一个点是一个离群点，但不是回归中的一个影响点，因为当该点被排除时，回归方程没有实质性substantially的改变。然而，当该点被纳入时，残差的正态性假设没有得到验证。

```r
influenceIndexPlot(q4.mod) # 找outlier
influencePlot(q4.mod) #查找 influential points
```

##### 如果改进了

All the plots have improved considerably.

#### Levene’s test

`library(car) 然后可以leveneTest(model1)`   检查 homoscedasticy

This test has a large p-value, saying that hypothesis of homoscedasticy is not rejected. 

同质变异数（Homoscedasticity）指的是一系列的[随机变量](https://zh.m.wikipedia.org/wiki/随机变量)间的方差相同

 **异质变异数**（英语：Heteroscedasticity），又称**分散不均一性**，指的是一系列的[随机变量](https://zh.m.wikipedia.org/wiki/随机变量)间的方差不相同，

#### 方差uniform

The assumption of uniform variance is not so clear from the plots, particularly from the Scale-Location graph. The test we used for analysis of variance does not work here, because we do not have grouped data. A test that can be used in this situation is the Score Test

```{r}
ncvTest(model4)
```

#### Shapiro-Wilk

`shapiro.test(df1$sp1)` 检查是不是正态分布, 做t-test之前要检查!  也可以同时做一个qqnorm qqline检查.    `shapiro.test(resid(mod1)) `如果p比0.05大, 那么 This shows that at the 5% level (or lower levels) we cannot reject the null hypothesis of Gaussianity.

SumSq 就是sum of squre

The F value is the ratio MSA/MSE and the last column labeled Pr(>F) is the probability of exceeding the calculated F-value when the null hypothesis is true, i.e. it is the p-value for the F test.

#### Tukey HSD

什么是Tukey’s HSD procedure , 怎么做  pairwise comparisons 

We can do pairwise comparisons using TukeyHSD.

方差分析的3个假设条件是：1、总体服从正态分布，2、个体是相互独立，3、组间方差相等

在PPT V25 

如果F检验的结果是拒绝没有治疗效果的无效假设，人们自然会对确定差异在哪里感兴趣。为此，有必要对各组进行比较。

```R
with(Tire, pairwise.t.test(a, b , p.adjust.method= 'bonferroni'))
(mod1.tky = TukeyHSD(mod1,conf.level = 0.95))
plot(mod1.tky)
```

 p值高于5%，说明它们没有显著差异。p值小于1%，它们有显著差异。

画图,  If the intervals include zero,  then the difference is not significant. 

https://wiki.mbalib.com/wiki/HSD%E6%A3%80%E9%AA%8C%E6%B3%95

```R
attach(xxx)之后 就不用加frame 的前缀了, 可以直接引用xxx
```

confidence band for 回归线

```R
predict(mode,new.data,interval='p')
abline(lm1) 
```

  SST = SSE  +SSR  , 

determination 系数, R^2,  = SSR/SST

### V32 简单线性模型6

画图要画出anova. 画出Diagnostic charts  `plot(model2)` 

#### 模型等式

yik =μ+τi +εik

with1≤i≤5,1≤k≤5. μ is the overall mean总的平均,τi represents the effect of treatment i,and εik are the experimental errors, which are assumed to be independent Gaussian random variables with mean 0 and common variance σ2. 

influential and atypical points

#### summary的解读

estimated standard deviation越小, 说明fit的越好. 

Multiple R-squared.多重 R 平方也称为决定系数，这是经常被引用的衡量模型与数据拟合程度的指标, 可以读出来. 越大越好. 

summary一个线性模型, 如果一个变量,estimate = 1.44×10-3, p值却很大，这说明截距与零没有显著区别。

系数的p值都很小，而R2几乎是88%。说明模型比较充分adequate. 

**Residuals**：模型预测的值与 y 的实际值之间的差异

一个p value会显示在Fstatic , 最后是 F 统计量。包括 t 检验，这是汇总函数为 lm 模型生成的第二个“检验”。F 统计量是一种“全局”测试，用于检查您的系数中是否至少有一个非零。

#### problemlist8

```R
scatterplot(City.fc~Weight, data=City) 需要 car库.
scatterplotMatrix(birds[,2:4]) # 可以取部分变量. 
abline(modela) # 画出这个模型的线
```

#### 写出等式

写出equation, note anova-pdf中有lm的例子, 但是什么时候是lm拟合? 什么时候不用lm拟合?  目前讲的都是lm拟合. 

Write down the equation for the regression line and interpret the parameters

summary出来, 看  ( estimate std 这一列)  ,   y =  sepal.length x - intercept 

##### exercise4

summary , p 很小, 拒绝假设, 两个参数和0不同. 

Give a confidence interval at a confidence level of 98% for the parameters of the regression.

```r
confint(wt.lm,level = 0.98)
 # It is equal to the square of the correlation between wt and ht:
cor(ais$wt,ais$ht)ˆ2
```

#### boxcox

Box-Cox变换是一个变换系列，试图纠正数据的非正态性。 该图显示了参数λ的对数似然。和一个大约95%的置信区间。通常情况下，如果这个置信区间包括零，我们应该试着进行对数变换，也就是λ=0时的变换。就看最高的时候的x坐标. 

### lec14 多元线性回归

```R
获得一个hat matrix,regressors hp and wt
model.matrix
nrow
ncol
diag
sum(hii)可以
rstandards
查看相关性
library(psych)
describe(dataB)
describeBy(birds[,1:3],group = birds$Aquatic)
```

#### test on individual parameters

#### properties of residuals

### V35 multiplereg3

[多重线性回归](https://www.zhihu.com/search?q=多重线性回归&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"1677930134"})是研究一个因变量和多个自变量的关系

变量回归中的p, 当回归中的 p 值大于显着性水平时，表明您的样本中没有足够的证据得出存在非零相关性的结论。

#### Backward Elimination

αcrit is sometimes referred to as the ‘p-to-remove’ and is typically set to 15 or 20%.

每次选最大p的去掉, 慢慢减掉一些变量, 直到所有的p都小于0.01

```r
model1 <- lm(pemax ~ ., data = cystfibr) # 一开始如果变量很多, 可以用点, 不用一个个复制名字. 
lm2 <- update(lm1, ~. - wcc) #去掉wcc
```

For AIC use stepAIC in the MASS package, 慢慢去掉AIC最小的.直到所有的AIC都比none大.  AIC是啥?  

AIC, 是 线性模型的衡量

BIC,  是log的衡量

或者用For R^2作为指标, 用regsubsets

```r
library(MASS) 
stepAIC(lm1)
library(leaps)
a <- regsubsets(bmi ~ ., data = ex3.df)
which.max(summary(a)$adjr2) # 那些带星号的就是适合的变量.
```

#### problemlist

第一题, leverage 的x太长, 去掉intercept可以变短.  只要standard  residuals不大就不会影响. 

 +0 就是截距设置为0.

怎么获得scatter plot matrices

```R
pairs(iris[,1:4], pch = 19)
```

怎么获得 graphical representation of the correlation matrix. Comment on your results?

```R
 cor.ex3 <- cor(ex3.df) 
corrplot::corrplot.mixed(cor.ex3)#可以看图的correlation matrix
#round(res, 2)可以看数字的correlation matrix
 library(GGally) 
ggcorr(cor.ex3) #也是图的
```

What is the p-value for the overall significance test for the regression?

怎么预测predict? Predict the `res` value for a subject with covariates `(var1,var2,var3,var4,var5) = (65,100,50,0.02,3)`. Add a confidence interval at level 98%.

```{r}
newdata = data.frame(var1=65,var2=100,var3=50,var4=0.02,var5=3)
答案: predict(lm3,newdata, level = 0.98, interval = 'confidence') 
```

Check for large leverage points:

```R
plot(hatvalues(q4.mod), type = 'h', lwd=2, ylab='Leverage') 
abline(h=0.2, col='red')
然后你可以查找大于阈值的
high.lev <- (1:50)[hatvalues(q4.mod)>0.2] 
dimnames(sat)[[1]][high.lev]
```

#### problemlist 9

#### residualPlots

问题: Add a quadratic term to the initial regression model. Print the summary table, and interpret the results. 

function residualPlots in the car package. 这个函数将残差与所有回归因子和拟合值作图，并增加一个二次项。它还测试了新增项的显著性，并列出了p值。

总体意义检验的P值 在summary表的底部。

画曲线要用curve

```
curve(21.862962 + 1.849283*x + 0.051399*x^2, add=T, col='blue')
```

#### 采样分布

The sampling distribution for the estimated parameters is normal

where the covariance matrix **V** = σ2(**X**′**X**)−1 , 

estimated covariance matrix for **β****ˆ** can be obtained with 可以用 `vcov(lm3)` 来求.  也可以用`(summary(wt.lm)$sigmaˆ2)*invXtX `

Describe the sampling distribution for the estimated parameters in the previous regression.

The estimated parameters are **β****ˆ** = (βˆ0 , βˆ1 ), which have a normal distribution: 

**β****ˆ** = N  ( β 0 , β 1 ) ′ , σ 2 ( **X** ′ **X** ) − 1 

The matrix (**X**′**X**)−1 is obtained in R with 

`(invXtX <- summary(wt.lm)$cov.unscaled)` 参考problem list 8 , question4. 

### lec15

#### 相关性

scatterplotMatrix来检查 , 如果最后一行是直线, 那么没有relation, 如果是45度, 那么shows a strong linear relation with positive slope.  普通角度就是 a (moderate) linear relation with res with positive slope, but there is more variability in these cases. could not find function "scatterplotMatrix"是为什么?

变量var3和var4有适度大的负相关，可能造成多重共线性multicollinearity问题。变量var5与res有一个重要的正相关关系。这一点在上一张图中已有评论。变量var2和var3与res有一个中等程度的正相关。

poly  ,polynomial 回归. 

大写的S 也是summary命令

```
influencePlot
```

要会解释回归分析的系数是什么意思

#### ncvTest

显著性, 可以用ncvTest, 为了Check the constant variance assumption, 全称是Non-constant Variance Score Test . p 很大, 就不能拒绝 homogeneity of variances.  p 小, 就拒绝 homogeneity of variances. 

#### 删除部分列

```R
drops <- c("svi","gleason")
prostate = prostate[ , !(names(prostate) %in% drops)]
```

怎么Compute and plot the correlation coefficients for the regressors.  

#### collinearity

vif是什么? variance inflation factors 

应用:  Variables 2, 3 and 4 appear significant(模型summary中pr很小.) but we need to check for collinearity

```r
vif(mod1)
library(corrplot)
cor.q2 <- cor(q2.df) 
corrplot.mixed(cor.q2)
# We see the Variance Inflation Factors for variables 4, 3 and 2 are large and the correlation matrix also shows large values. 
#因为vif 最大 for variable 4, we try dropping it from the model
 mod2 <- update(mod1, .~. - var4)
```

#### AIC

stepAIC in package MASS to get a reduced model

`modelAIC <- stepAIC(mod1)`   可以计算AIC,   AIC 越小模型越好. 

BIC criterion是什么?     就是BIC(mod1)

categorical regressor  怎么写? 

estimatation kernels.

```
compare(x.group  = , est.group=  )
x = Pima()
Kda(x= x, x.group= groups)
compare()
```

考试 after experiment design, 不包括 experiment design.

and the interaction between the two是什么意思? 
