#   | Graphics in R
# | Low level commands and graphical windows


# Low level commands
## Legends

## legend(x, y = NULL, legend, fill, col, bg)


cols <- hcl.colors(15, "Set 2")
with(iris, plot(Petal.Length,Petal.Width, 
              pch=16+as.numeric(Species),
              col = cols[1+2*as.numeric(iris$Species)]))
legend('topleft',
       legend=c('setosa','versicolor','virginica'),
       col=cols[c(3,5,7)], pch=16+(1:3))     


attach(mtcars)
boxplot(mpg~cyl, xlab="Cylinders", ylab="Miles/gallon", 
        col=topo.colors(3))
 
legend("bottomleft", inset=.02, 
       title="Number of Cylinders",
       c("4","6","8"), fill=topo.colors(3), 
       horiz=TRUE, cex=0.8)
 
detach(mtcars)

## `points` 
attach(iris)
boxplot(Petal.Length ~ Species, data = iris)
points(jitter(as.numeric(Species), factor = 0.2), 
       Petal.Length, 
       col = cols[c(6,8,10)][as.numeric(Species)])
detach(iris)

## `lines`
plot(conc[Subject == 1] ~ time[Subject == 1], 
     data = Indometh,col=cols[2],
     type = 'o', ylim = c(0,2.7), xlab = 'time (hr)',
     ylab = 'concentration (mcg/ml)',
     main='Concentration of indometacin')
for(i in 2:6){
  lines(conc[Subject == i] ~ time[Subject == i], 
      data = Indometh, type = 'o', col = cols[2*i])
  }
legend('topright', legend = c('1','2','3','4','5','6'), 
      col = cols[2*(1:6)], pch = rep(1,6), 
      lwd = rep(1,6))




## `abline`

plot(cars) 
abline(h=60) 
abline(v=15)  

plot(cars)
abline(reg = lm(dist ~ speed, data = cars))


## Other Commands
with(mtcars,plot(disp, mpg))

par(mar = c(5,4,4,4.5)) 
attach(mtcars)
plot(disp, mpg, type='n',axes=F, ylim = c(10,35), 
     xlim = c(50,500), xlab = 'displacement', ylab = '' )
points(disp[cyl==4], mpg[cyl==4], pch=16, col=2) 
points(disp[cyl==6], mpg[cyl==6], pch=17, col=3) 
points(disp[cyl==8], mpg[cyl==8], pch=18, col=4)
axis(1)
axis(4)
mtext('mpg',4, line = 2)
title('Fuel Consumption', 'Data from 1974') 
arrows(470, 17, 470, 12, code = 2, length = 0.15) 
text(485,20,'worst\nyield',font=3,adj=1) 
arrows(100,34, 150,34, code=1, length = 0.2) 
text(220, 34, 'best yield', font=2)
leg.txt <- c('4 cyl.', '6 cyl.', '8 cyl.') 
legend(400,35, leg.txt,col=2:4, pch=16:18)




## Graphical Parameters 
par()

?par

old.par <- par(no.readonly = TRUE) 
par(bg=7, bty='u', cex=1.5, col='blue', col.axis=4,
    font=2,lty='dashed', lwd=3, pch=3, las=2, tck =1)



## Graphical Parameters 

par(bg=7, bty='u', cex=1.5, col='blue', col.axis=4,
    font=2,lty='dashed', lwd=3, pch=3, las=2, tck =1)
plot(cars)
plot(iris)
par(old.par) 
plot(cars)


## Graphical Windows
split.screen(c(2, 1))     # split display into two screens
split.screen(c(1, 2), 2)  # split bottom half in two
plot(1:10)                # screen 3 is active, draw plot
erase.screen()            # erase and redraw
plot(1:10, ylab = "ylab 3")
screen(1)                 # prepare screen 1 for output
plot(1:10)
screen(4)                 # prepare screen 4 for output
plot(1:10, ylab = "ylab 4")
screen(1, FALSE)    # return to screen 1, but do not clear
plot(10:1, axes = FALSE, 
     lty = 2, ylab = "")  # overlay second plot
axis(4)                # add tic marks to right-hand axis
title("Plot 1")
close.screen(all = TRUE)    # exit split-screen mode

plot(1:10) 


## \texttt{layout}
(mat1 <- matrix(1:4,2,2))
layout(mat1)
layout.show(4) 

layout(matrix(1:6,3,2))
layout.show(6)
layout(matrix(1:6,3,2,byrow=TRUE))
layout.show(6)
layout(matrix(1:6,2,3))
layout.show(6)
layout(matrix(1:6,3,2,byrow=TRUE))
layout.show(6)
(m <- matrix(c(1:3,3), 2, 2))
layout(m)
layout.show(3)


m <- matrix(1:4, 2, 2)
layout(m, widths=c(1,3), heights=c(3,1))
layout.show(4)
m <- matrix(c(1,1,2,1), 2, 2)
layout(m, widths=c(2,1), heights=c(1,2))
layout.show(2)

m <- matrix(0:3, 2, 2)
layout(m, widths=c(1,3), heights=c(1,3))
layout.show(3)


nf <- layout(matrix(c(2,0,1,3),2,2,byrow=TRUE),
             c(3,1), c(1,3), TRUE)
layout.show(nf)
par(mar=c(2,2,2,2)) 
x <- pmin(3, pmax(-3, rnorm(50)))
y <- pmin(3, pmax(-3, rnorm(50)))
xrange <- c(-3,3)
yrange <- c(-3,3)
xhist <- hist(x,breaks=seq(-3,3,0.5),plot=FALSE)
yhist <- hist(y,breaks=seq(-3,3,0.5),plot=FALSE)
top <- max(c(xhist$counts, yhist$counts))

plot(x, y, xlim=xrange, ylim=yrange, xlab='',
     ylab='')
barplot(xhist$counts, axes=FALSE,
        ylim=c(0, top), space=0)
barplot(yhist$counts, axes=FALSE,
        xlim=c(0, top), space=0, horiz=TRUE)
par(old.par)

## \texttt{locator}

# Syntax
## locator(n,type)

plot(cars)
locator(3,type='n')
locator(2,type='l')
text(locator(1), 'Point', adj=0)



## \texttt{identify}
## identify(x, y, labels).
plot(cars)
identify(cars)
