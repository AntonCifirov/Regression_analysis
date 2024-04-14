library(ggplot2)
library(ggpubr)
library(QuantPsyc)
library(boot)
library(pROC)
library(regclass)
    # Duomenu nuskaitymas
    set.seed("12345")
    dat <- read.csv("diabetes.csv")
    is.numeric(dat$Outcome)
    dat$Outcome <- ifelse(dat$Outcome == 1,"Sick","Healthy")
    dat$Outcome <- factor(dat$Outcome)
    
    #Pasidalinam duomenys į apmokymo ir testavimo dalys
    test <- sample(1:nrow(dat), size = 0.2*nrow(dat), replace = FALSE)
    dat_train <- dat[-test,]
    dat_test <- dat[test,]


# Grafiku piesimas
sick <- sum(dat$Outcome=="Sick")
healthy <- sum(dat$Outcome=="Healthy")
df = data.frame(cond = c("Healthy","Sick"), num = c(sick,healthy))
ggplot(data = df, aes(x = cond, y = num)) +
  geom_bar(stat = "identity") + theme(axis.title.x=element_blank(), axis.title.y=element_blank())
plt_1 <- ggplot(data=dat,aes(x=Outcome,y = Pregnancies))+ geom_boxplot()
plt_2 <- ggplot(data=dat,aes(x=Outcome,y = Glucose))+ geom_boxplot()
plt_3 <- ggplot(data=dat,aes(x=Outcome,y = BloodPressure))+ geom_boxplot()
plt_4 <- ggplot(data=dat,aes(x=Outcome,y = SkinThickness))+ geom_boxplot()
plt_5 <- ggplot(data=dat,aes(x=Outcome,y = Insulin))+ geom_boxplot()
plt_6 <- ggplot(data=dat,aes(x=Outcome,y = BMI))+ geom_boxplot()
plt_7 <- ggplot(data=dat,aes(x=Outcome,y = DiabetesPedigreeFunction))+ geom_boxplot()
plt_8 <- ggplot(data=dat,aes(x=Outcome,y = Age))+ geom_boxplot()

ggarrange(plt_1, plt_2, plt_3, plt_4, plt_5, plt_6, plt_7, plt_8,ncol=4, nrow = 2)


# Pirminis modelis
diabetes_logit <- glm(formula = Outcome ~ Pregnancies + Glucose + Age + BloodPressure + SkinThickness + Insulin + BMI +DiabetesPedigreeFunction,
family = binomial(logit), data = dat_train)
summary(diabetes_logit)

VIF(diabetes_logit)
# Ismetami kintamieji
diabetes_logit <- glm(formula = Outcome ~ Pregnancies + Glucose + BMI,
family = binomial(logit), data = dat_train)
summary(diabetes_logit)

# Isimtys

pearson_residuals <- glm.diag(diabetes_logit)$rp
plot(pearson_residuals, pch="*", cex=2, main="Pearson
residuals")

# Nuliniu reiksmiu naikinimas
sum(abs(pearson_residuals) > 3)
dat[abs(pearson_residuals) > 3,]
dat1_test <- dat_test[dat_test$Glucose != 0 & dat_test$BMI != 0 & dat_test$BloodPressure != 0,]
dat1_train <- dat_train[dat_train$Glucose != 0 & dat_train$BMI != 0 & dat_train$BloodPressure != 0,]


# Galutinio modelio kurimas
diabetes_logit_final <- glm(formula = Outcome ~ Pregnancies + Glucose + BMI,
family = binomial(logit), data = dat1_train)
summary(diabetes_logit_final)

pearson_residuals1 <- glm.diag(diabetes_logit_final)$rp
plot(pearson_residuals1, pch="*", cex=2, main="Pearson
residuals")

VIF(diabetes_logit_final)


# Modelio testavimas
      #Antras modelis
      slenkstis <- 0.5
      
      
      model_probs1 <- predict(diabetes_logit_final, dat1_test,
                              type = "response")
      model_pred1 <- rep("Healthy", length(model_probs1))
      model_pred1[model_probs1 > slenkstis] <- "Sick"
      conf_m1 <- table(model_pred1, dat1_test$Outcome)
      test_acc1 <- mean(model_pred1 == dat1_test$Outcome)
      
      #Ivertinimas false negative & false positive klaidu tikimybes
      tn1 <- conf_m1[1,1]
      fn1 <- conf_m1[1,2]
      fp1 <- conf_m1[2,1]
      tp1 <- conf_m1[2,2]
      fn_prob1 <- fn1/(fn1+tp1) 
      fp_prob1 <- fp1/(fp1+tn1) 
      
      specificity1 <- tn1/(tn1+fp1)
      sensitivity1 <- tp1/(tp1+fn1) #reikia padidint sensitivity
      
      roc_curve <- roc(dat1_test$Outcome, model_probs1)
      plot(roc_curve ,main ="ROC curve -- Logistic Regression ")
      points(specificity1, sensitivity1, lwd = 3, cex = 1,, col = "red" ) 
      text(x = specificity1-0.15, y = sensitivity1-0.02, labels = "slenkstis = 0.5", col = "red", cex = 1)
      abline(v=specificity1, lty = 2, col = "red")
      abline(h=sensitivity1, lty = 2, col = "red")
      
      #Slenkscio parinkimas sumazinti false negative
      #cutpoint <- coords(roc_curve, x = 'best', input = 'threshold', best.method = 'youden')[1,1]
      # opt_slenkstis <- coords(roc_curve, "best", ret = "threshold")[1,1]
      
      rc <-  data.frame(roc_curve$sensitivities, roc_curve$specificities, roc_curve$thresholds,
                        "tf"=abs(roc_curve$sensitivities - (roc_curve$specificities)))
      rc <- rc[order(rc$tf),]
      rownames(rc) <- 1:length(rc$tf)
      opt_slenkstis <- rc[1,]
      
   
      model_pred2 <- rep("Healthy", length(model_probs1))
      model_pred2[model_probs1 > opt_slenkstis$roc_curve.thresholds] <- "Sick"
      conf_m2 <- table(model_pred2, dat1_test$Outcome)
      test_acc2 <- mean(model_pred2 == dat1_test$Outcome)
      test_acc2
      
      #Ivertinimas false negative & false positive klaidu tikimybes
      tn2 <- conf_m2[1,1]
      fn2 <- conf_m2[1,2]
      fp2 <- conf_m2[2,1]
      tp2 <- conf_m2[2,2]
      fn_prob2 <- fn2/(fn2+tp2) 
      fp_prob2 <- fp2/(fp2+tn2) 
      
      specificity2 <- tn2/(tn2+fp2)
      sensitivity2 <- tp2/(tp2+fn2) 
      
      #pridedame taska grafike atitinkanti musu parinkta slenksti
      points(specificity2, sensitivity2, lwd = 3, cex = 1,, col = "darkgreen")
      points(opt_slenkstis$roc_curve.specificities, opt_slenkstis$roc_curve.sensitivities, lwd = 3, cex = 1,, col = "darkgreen")
      text(x = opt_slenkstis$roc_curve.specificities-0.15, y = opt_slenkstis$roc_curve.sensitivities-0.02, labels = paste("slenkstis ≈ ", round(opt_slenkstis$roc_curve.thresholds,2)), col = "darkgreen", cex = 1)
      abline(v=opt_slenkstis$roc_curve.specificities, lty = 2, col = "darkgreen")
      abline(h=opt_slenkstis$roc_curve.sensitivities, lty = 2, col = "darkgreen")



# Testavimas, slenkstis = 0.5
ClassLog(diabetes_logit_final, dat1_train$Outcome)
res <- predict(diabetes_logit_final, newdata = dat1_test, type = "response")
c50 <- ifelse(res > 0.5,"Sick","Healthy")
c50 <- ordered(c50, levels = c("Healthy" , "Sick"))
dat1_test$Outcome <- ordered(dat1_test$Outcome, levels = c("Healthy" , "Sick"))

# Klasifikavimo lentele slenkstis = 0.5



# Testavimas, slenkstis = 0.1995
ClassLog(diabetes_logit_final, dat1_train$Outcome, cut = opt_slenkstis$roc_curve.thresholds)
res <- predict(diabetes_logit_final, newdata = dat1_test, type = "response")
c50_ <- ifelse(res > opt_slenkstis$roc_curve.thresholds,"Sick","Healthy")
c50_ <- ordered(c50_, levels = c("Sick","Healthy"))
dat1_test$Outcome <- ordered(dat1_test$Outcome, levels = c("Healthy" , "Sick"))


# Klasifikavimo lentele slenkstis = 0.1995
cm_ <- table(Predicted = c50_, Actual = dat1_test$Outcome)
cm1_ <- cm_
cm1_[,1] <- cm1_[,1] / sum(cm_[,1])
cm1_[,2] <- cm1_[,2] / sum(cm_[,2])
cm_
cm1_
(cm_[2] + cm_[3] )/ sum(cm_)





