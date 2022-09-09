##  Graphics in R
##  Basics



# Graphics in `R`
  
example(plot)
# example(par)
# example("palette")

demo(graphics)
# demo(persp)
# demo(image)
# demo(colors)


# plot
par(ask = FALSE)
attach(iris)
plot(Sepal.Length, Sepal.Width)
plot( Sepal.Width ~ Sepal.Length)
plot(~ Sepal.Length + Sepal.Width)

plot(dist ~ speed, data = cars)

plot(Species)

plot(Species, Sepal.Length)

plot(iris)
plot(Petal.Length ~ Sepal.Width + Sepal.Length)
detach(iris)

## type 
str(cars)
plot(cars, type='p')
plot(cars, type='l')
plot(cars, type='b')
plot(cars, type='o')
plot(cars, type='h')
plot(cars, type='n')


par(mfrow=c(1,2))
plot(pressure,type='s')
points(pressure)
plot(pressure,type='S')
points(pressure)
par(mfrow=c(1,1))



  # `xlab` and `ylab`

plot(cars, xlab = 'Speed (mph)', 
     ylab = 'Breaking distance (ft)')


  # `main` and `sub`

plot(cars, xlab = 'Speed (mph)', ylab = 'Distance (ft)',
     main = 'Breaking distance', sub = 'Data from 1920')

  # xlim and ylim 
attach(mtcars); 
par(mfrow=c(1,2))
plot(mpg[am==0] ~ disp[am==0], 
     xlab = 'displacement (cu in)',
     ylab = 'fuel consumption (mpg)')
plot(mpg[am==1] ~ disp[am==1], 
     xlab = 'displacement (cu in)',
     ylab = 'fuel consumption (mpg)')
par(mfrow=c(1,1))

(r1 <- range(mpg))
(r2 <- range(disp))

par(mfrow=c(1,2))
plot(mpg[am==0] ~ disp[am==0], 
     xlab = 'displacement (cu in)',
     ylab = 'fuel consumption (mpg)', 
     ylim = c(10, 35),xlim = c(70, 480))
plot(mpg[am==1] ~ disp[am==1], 
     xlab = 'displacement (cu in)',
     ylab = 'fuel consumption (mpg)', 
     ylim = c(10, 35), xlim = c(70, 480))
par(mfrow=c(1,1))


  # asp
plot(cars,type='l',xlab='Speed',ylab='Distance',
     main='Braking Distance',sub='Data from 1920')
plot(cars,type='l',xlab='Speed',ylab='Distance',
     main='Braking Distance',sub='Data from 1920', 
     asp=0.5)
plot(cars,type='l',xlab='Speed',ylab='Distance',
     main='Braking Distance',sub='Data from 1920', 
     asp=0.1)


  ### `lty` and `lwd`
showLty <- function(ltys, xoff = 0, ...) {
  stopifnot((n <- length(ltys)) >= 1)
  op <- par(mar = rep(.5,4)); on.exit(par(op))
  plot(0:1, 0:1, type = "n", axes = FALSE, ann = FALSE)
  y <- (n:1)/(n+1)
  clty <- as.character(ltys)
  mytext <- function(x, y, txt)
    text(x, y, txt, adj = c(0, -.3), cex = 0.8, ...)
  abline(h = y, lty = ltys, ...); mytext(xoff, y, clty)
  y <- y - 1/(3*(n+1))
  abline(h = y, lty = ltys, lwd = 2, ...)
  mytext(1/8+xoff, y, paste(clty," lwd = 2"))
}
showLty(c("solid", "dashed", "dotted",
          "dotdash", "longdash", "twodash"))

  # points
x <- rep(1:5,5)
y <- rep(6:2,rep(5,5))
plot(x,y,type = 'n', axes = FALSE, ylab = '', 
     xlab = '', ylim=c(0,7))
points(x, y, pch = 0:24,  bg='green', col = 'red')
points(3,1,pch = 25, bg = 'green', col = 'red')
text(x, y, labels = 0:24, pos = 3, offset = 0.6)
text(3,1,labels = 25, pos = 3, offset = 0.6)


  # Color
colors()

palette()
xx <- 1:8
yy <- rep(1, 8)
plot(xx, yy/2, type = 'n', axes = FALSE, 
     ylab = '', xlab = '')
text(xx,yy/2,c(' 1 black', '2 red', '3 green3', 
               '4 blue', '5 cyan',
               '6 magenta', '7 yellow', 
               '8 grey '), col = 1:8)

pie(rep(1, 50), col = rainbow(50), labels = '')















