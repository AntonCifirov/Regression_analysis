library(boot)
library(ggpubr)
library(ggplot2)
library(car)
library(statmod)
library(corrplot)

# Duomenu nuskaitymas
dt <- read.table('auto-mpg.data')
colnames(dt) <- c('mpg', 'cyl', 'dis', 'hp', 'wgh', 'acc', 'yr', 'orig', 'name')

apply(dt,MARGIN = 2, FUN = min)
apply(dt,MARGIN = 2, FUN = max)

dt

# Netinkamu reiksmiu naikinimas
nrow(dt)
# [1] 398
dt <- dt[dt$hp != "?",]
nrow(dt)
# [1] 392

# Atsako kintamojo pasiskirstymo patikrinimas
m <- mean(dt$mpg)
v <- var(dt$mpg)
scale <- v/m
shape <- m*m/v
shape
nu = 23.4459184099745
lam = 193.383166845373
# 9.023767
scale
# 2.598241

hist(dt$mpg, freq = FALSE, ylab = "Tankis", xlab = "Kuro sąnaudos, mpg", main = "Atsako kintamojo histograma", ylim = c(0, 0.056))
lines(x=0:50, y = dgamma(0:50, shape = shape, scale = scale ), col = 'red')
lines(x=0:50, y = dinvgauss(0:50, mean = nu, shape = lam), col = 'blue')

paste(scale,shape,m,v)

# KS-Test

ks.test(x = dt$mpg, function(x) pgamma(x, shape = shape, scale = scale ))
ks.test(x = dt$mpg, function(x) pinvgauss(x, mean = nu, shape = lam))

# 	Asymptotic one-sample Kolmogorov-Smirnov test
#
# data:  dt$mpg
# D = 0.057656, p-value = 0.1476
# alternative hypothesis: two-sided

# Pradine duomenu analize

dt$cyl <- as.character(dt$cyl)
dt$orig <- as.character(dt$orig)
dt$hp <- as.integer(dt$hp)
plt_1 <- ggplot(data=dt, aes(x = cyl, y = mpg)) + geom_boxplot(size = 0.25)
plt_2 <- ggplot(data=dt, aes(x = orig, y = mpg)) + geom_boxplot(size = 0.25)
plt_3 <- ggplot(data=dt, aes(x = dis, y = mpg)) + geom_point(size = 0.25)
plt_4 <- ggplot(data=dt, aes(x = hp, y = mpg)) + geom_point(size = 0.25)
plt_5 <- ggplot(data=dt, aes(x = wgh, y = mpg)) + geom_point(size = 0.25)
plt_6 <- ggplot(data=dt, aes(x = acc, y = mpg)) + geom_point(size = 0.25)
plt_7 <- ggplot(data=dt, aes(x = yr, y = mpg)) + geom_point(size = 0.25)



ggarrange(plt_1, plt_2, plt_3, plt_4, plt_5, plt_6, plt_7, ncol=4, nrow = 2)


# Daugiau grafiku

plt_1 <- ggplot(data=dt, aes(x = dis, y = mpg)) + geom_point(size = 0.25)
plt_2 <- ggplot(data=dt, aes(x = wgh, y = mpg)) + geom_point(size = 0.25)
plt_3 <- ggplot(data=dt, aes(x = acc, y = mpg)) + geom_point(size = 0.25)
plt_4 <- ggplot(data=dt, aes(x = hp, y = mpg)) + geom_point(size = 0.25)
plt_5 <- ggplot(data=dt, aes(x = cyl, y = mpg)) + geom_boxplot(size = 0.25)
plt_6 <- ggplot(data=dt, aes(x = orig, y = mpg)) + geom_boxplot(size = 0.25)

plt_1l <- ggplot(data=dt, aes(x = log(dis), y = log(mpg))) + geom_point(size = 0.25)
plt_2l <- ggplot(data=dt, aes(x = log(wgh), y = log(mpg))) + geom_point(size = 0.25)
plt_3l <- ggplot(data=dt, aes(x = log(acc), y = log(mpg))) + geom_point(size = 0.25)
plt_4l <- ggplot(data=dt, aes(x = log(hp), y = log(mpg))) + geom_point(size = 0.25)
plt_5l <- ggplot(data=dt, aes(x = cyl, y = log(mpg))) + geom_boxplot(size = 0.25)
plt_6l <- ggplot(data=dt, aes(x = orig, y = log(mpg))) + geom_boxplot(size = 0.25)

plt_1i <- ggplot(data=dt, aes(x = 1/(dis), y = 1/(mpg))) + geom_point(size = 0.25)
plt_2i <- ggplot(data=dt, aes(x = 1/(wgh), y = 1/(mpg))) + geom_point(size = 0.25)
plt_3i <- ggplot(data=dt, aes(x = 1/(acc), y = 1/(mpg))) + geom_point(size = 0.25)
plt_4i <- ggplot(data=dt, aes(x = 1/(hp), y = 1/(mpg))) + geom_point(size = 0.25)
plt_5i <- ggplot(data=dt, aes(x = cyl, y = 1/(mpg))) + geom_boxplot(size = 0.25)
plt_6i <- ggplot(data=dt, aes(x = orig, y = 1/(mpg))) + geom_boxplot(size = 0.25)

ggarrange(plt_1, plt_2, plt_3, plt_4, plt_5, plt_6, plt_1l, plt_2l, plt_3l, plt_4l, plt_5l, plt_6l, plt_1i, plt_2i, plt_3i, plt_4i, plt_5i, plt_6i,ncol=6, nrow = 3)

# Multikolinearumas
dt$cyl <- as.numeric(dt$cyl)
dt$orig_1 <- as.integer(dt$orig == 1)
dt$orig_2 <- as.integer(dt$orig == 2)
corrplot.mixed(cor(dt[c("cyl", "dis", "hp", "wgh", "acc", "orig_1", "orig_2")]))

# Išimčių analizė

model <- glm(mpg ~ orig + wgh + acc,family = Gamma(link = 'identity'), data = dt)
eta.log <- model$linear.predictor
plt_a1 <- ggplot(data=dt, aes(x = fitted(model), y = rstandard(model))) + geom_point(size = 1) + xlab("Fitted values") + ylab("Standardized residuals") + ggtitle("Log link")
plt_a2 <- ggplot(data=dt, aes(x = eta.log, y = (resid(model, type="working") + eta.log))) + geom_point(size = 1) + xlab("Linear predictor, eta") + ylab("Working residuals") + ggtitle("Linear predictor plot")
plt_a3 <- ggplot(data=dt, aes(sample = qresid(model))) + stat_qq() + stat_qq_line(col = "red") + xlab("Theoretical Quantiles") + ylab("Sample Quantiles") + ggtitle("Normal Q-Q plot")
plt_a4 <- ggplot(data = dt, aes(x = 1:nrow(dt), y = cooks.distance(model))) + geom_bar(stat="identity") + xlab("Index") + ylab("Cook's distance") + ggtitle("Cook's distance plot")
ggarrange(plt_a1, plt_a2, plt_a3, plt_a4, ncol=2, nrow = 2)

# Cross-Validation

dt$folds <- 1:nrow(dt) %% 10


# Gamma log
se_ga_log <- numeric(10)
for (i in 1:10){
  dt_test <- dt[which(dt$folds == i-1),]
  dt_train <- dt[which(dt$folds != i-1),]
  model_ga_log <- glm(mpg ~ orig + log(wgh) + log(acc),family = Gamma(link = 'log'), data = dt_train)
  res <- exp(predict(model_ga_log, newdata = dt_test))
  se_ga_log[i] <- mean((res - dt_test$mpg)^2)
}
model_ga_log <- glm(mpg ~ orig + log(wgh) + log(acc),family = Gamma(link = 'log'), data = dt)
AIC(model_ga_log)
se_ga_log

c(AIC(model_ga_log), mean(se_ga_log))
# [1] 2122.93357   17.62845


# Gamma identity
se_ga_id <- numeric(10)
for (i in 1:10){
  dt_test <- dt[which(dt$folds == i-1),]
  dt_train <- dt[which(dt$folds != i-1),]
  model_ga_id <- glm(mpg ~ orig_1 + orig_2 + wgh + acc,family = Gamma(link = 'identity'), data = dt_train)
  res <- (predict(model_ga_id, newdata = dt_test))
  se_ga_id[i] <- mean((res - dt_test$mpg)^2)
}
model_ga_id <- glm(mpg ~ orig + wgh + acc,family = Gamma(link = 'identity'), data = dt)
AIC(model_ga_id)
se_ga_id

c(AIC(model_ga_id), mean(se_ga_id))
# 2159.61730   18.29898


# Gamma inv
se_ga_in <- numeric(10)
for (i in 1:10){
  dt_test <- dt[which(dt$folds == i-1),]
  dt_train <- dt[which(dt$folds != i-1),]
  model_ga_in <-  glm(mpg ~ orig + I(1/(wgh)) + I(1/(acc)),family = Gamma(link = 'inverse'), data = dt_train)
  res <- 1/(predict(model_ga_in, newdata = dt_test))
  se_ga_in[i] <- mean((res - dt_test$mpg)^2)
}

model_ga_in <- glm(mpg ~ orig + I(1/(wgh)) + I(1/(acc)),family = Gamma(link = 'inverse'), data = dt)
AIC(model_ga_in)
se_ga_in

c(AIC(model_ga_in), mean(se_ga_in))
# 2241.54267   25.28852


# Inv Gaussian log
se_inv_log <- numeric(10)
for (i in 1:10){
  dt_test <- dt[which(dt$folds == i-1),]
  dt_train <- dt[which(dt$folds != i-1),]
  model_inv_log <- glm(mpg ~ orig + log(wgh) + log(acc),family = inverse.gaussian(link = 'log'), data = dt_train)
  res <- exp(predict(model_inv_log, newdata = dt_test))
  se_inv_log[i] <- mean((res - dt_test$mpg)^2)
}

model_inv_log <- glm(mpg ~ orig + log(wgh) + log(acc),family = inverse.gaussian(link = 'log'), data = dt)
AIC(model_inv_log)
se_inv_log

c(AIC(model_inv_log), mean(se_inv_log))
# 2110.82949   17.73566


# Inv Gaussian identity
se_inv_id <- numeric(10)
for (i in 1:10){
  dt_test <- dt[which(dt$folds == i-1),]
  dt_train <- dt[which(dt$folds != i-1),]
  model_inv_id <- glm(mpg ~ orig + wgh + acc,family = inverse.gaussian(link = 'identity'), data = dt_train)
  res <- (predict(model_inv_id, newdata = dt_test))
  se_inv_id[i] <- mean((res - dt_test$mpg)^2)
}

model_inv_id <- glm(mpg ~ orig + wgh + acc,family = inverse.gaussian(link = 'identity'), data = dt)
AIC(model_inv_id)
se_inv_id

c(AIC(model_inv_id), mean(se_inv_id))
# [1] 2157.22996   18.71493


# Inv Gaussian inverse

# aic_inv_in <- numeric(10)
se_inv_in <- numeric(10)
for (i in 1:10){
  dt_test <- dt[which(dt$folds == i-1),]
  dt_train <- dt[which(dt$folds != i-1),]
  model_inv_in <-  glm(mpg ~ orig + I(1/(wgh)) + I(1/(acc)),family = inverse.gaussian(link = 'inverse'), data = dt_train)
  res <- 1/(predict(model_inv_in, newdata = dt_test))
  se_inv_in[i] <- mean((res - dt_test$mpg)^2)
}
model_inv_in <-  glm(mpg ~ orig + I(1/(wgh)) + I(1/(acc)),family = inverse.gaussian(link = 'inverse'), data = dt)
AIC(model_inv_in)
se_inv_in

c(AIC(model_inv_in), mean(se_inv_in))
# [1] 2227.44286   31.19571

res <- data.frame(names = c("Gamma log", "Gamma identity", "Gamma inverse", "Inverse Gaussian log", "Inverse Gaussian identity", "Inverse Gaussian inverse"),
                  AIC = c(AIC(model_ga_log),AIC(model_ga_id), AIC(model_ga_in), AIC(model_inv_log), AIC(model_inv_id), AIC(model_inv_in)),
                  MSE = c(mean(se_ga_log), mean(se_ga_id), mean(se_ga_in), mean(se_inv_log), mean(se_inv_id), mean(se_inv_in))
)

summary(model_inv_log)
#
# Call:
# glm(formula = mpg ~ orig + log(wgh) + log(acc), family = inverse.gaussian(link = "log"),
#     data = dt)
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 10.27276    0.39063  26.298  < 2e-16 ***
# orig2        0.02204    0.02678   0.823   0.4109
# orig3        0.04873    0.02827   1.724   0.0855 .
# log(wgh)    -0.98043    0.03981 -24.626  < 2e-16 ***
# log(acc)     0.23065    0.04760   4.846 1.83e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# (Dispersion parameter for inverse.gaussian family taken to be 0.001173433)
#
#     Null deviance: 2.02706  on 391  degrees of freedom
# Residual deviance: 0.44602  on 387  degrees of freedom
# AIC: 2110.8
#
# Number of Fisher Scoring iterations: 5

summary(model_ga_log)

#
# Call:
# glm(formula = mpg ~ orig + log(wgh) + log(acc), family = Gamma(link = "log"),
#     data = dt)
#
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 10.11691    0.39837  25.396  < 2e-16 ***
# orig2        0.01654    0.02558   0.647   0.5182
# orig3        0.04579    0.02615   1.751   0.0808 .
# log(wgh)    -0.96301    0.04060 -23.721  < 2e-16 ***
# log(acc)     0.23722    0.05102   4.650 4.57e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# (Dispersion parameter for Gamma family taken to be 0.02704548)
#
#     Null deviance: 44.205  on 391  degrees of freedom
# Residual deviance: 10.150  on 387  degrees of freedom
# AIC: 2122.9
#
# Number of Fisher Scoring iterations: 4




