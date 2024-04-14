import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Duomenu nuskaitymas
random.seed("123")
dat = pd.read_csv("diabetes.csv")
n = len(dat["Outcome"])
sample = random.sample(range(0,n), int(n * 0.2),)
smpl_test = [(k in sample) for k in range(0,n)]
smpl_train = [(k not in sample) for k in range(0,n)]
dat_test = dat[smpl_test]
dat_train = dat[smpl_train]
print(len(dat_train["Outcome"]))
print(len(dat_test["Outcome"]))


# Grafiku piesimas

print(sum(dat["Outcome"] == 1)/ len(dat["Outcome"]))
plt.title("Atsako kintamojo reikšmių pasiskirstymas")
plt.bar(x = ["Yra diabetas", "Nėra diabeto"], height = [sum(dat["Outcome"] == 1), sum(dat["Outcome"] == 0)])
plt.savefig("Barplot.png")
plt.clf()


plt.title("Palyginamosios stačiakampės diagramos")
fig = plt.figure(figsize=(14,7))
grid = gridspec.GridSpec(2, 4)


p = 1
for i in range(0,2):
    for k in range(0,4):
        exec("plt_{} = fig.add_subplot(grid[{},{}])".format(p, i, k))
        p += 1

vrs = ["Pregnancies", "Glucose" , "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
i = 1
for v in vrs:
    exec("plt_{}.boxplot( [dat[\"{}\"][dat[\"Outcome\"] == 0], dat[\"{}\"][dat[\"Outcome\"] == 1]], widths = 0.75)".format(i,v,v))
    exec("plt_{}.set_title(\"{}\")".format(i, v))
    i += 1

fig.suptitle("Playginamosios stačiakampės diagramos")
plt.savefig("boxplot.png")
plt.clf()

# Pirminis modelis
diabetes_logit = smf.glm('Outcome ~ Pregnancies + Glucose + Age + BloodPressure + SkinThickness + Insulin + BMI +'
                         'DiabetesPedigreeFunction', dat_train, family=sm.families.Binomial()).fit()
print(diabetes_logit.summary())

# Ismetami kintamieji
diabetes_logit = smf.glm('Outcome ~ Pregnancies + Glucose + BMI', dat_train, family=sm.families.Binomial()).fit()
print(diabetes_logit.summary())

# Isimtys

plt.title("Binarinio atsako modelio Pearson'o liekanos")
plt.ylabel("Pearson'o liekanos")
plt.xlabel("Indeksai")
plt.scatter(x = range(0,len(diabetes_logit.resid_pearson)), y = diabetes_logit.resid_pearson,
            c = "black", marker = "x")
plt.savefig("Pearson_resiguals.png")
plt.clf()

# Nuliniu reiksmiu naikinimas

print(sum(abs(diabetes_logit.resid_pearson > 3)))
print(dat_train[diabetes_logit.resid_pearson > 3])
dat1_test = dat_test[(dat_test["Glucose"] != 0) & (dat_test["BMI"] != 0) & (dat_test["BloodPressure"] != 0)]
dat1_train = dat_train[(dat_train["Glucose"] != 0) & (dat_train["BMI"] != 0) & (dat_train["BloodPressure"] != 0)]

# Galutinio modelio kurimas

diabetes_logit_final = smf.glm('Outcome ~ Pregnancies + Glucose + BMI', dat1_train, family=sm.families.Binomial()).fit()
print(diabetes_logit_final.summary())


# Testavimas
pred = diabetes_logit_final.predict(dat1_test)
cutoff = 0.5
prd = [0 if k < cutoff else 1 for k in pred]

con_m = confusion_matrix(y_true = dat1_test["Outcome"], y_pred= prd)
print("Klasifikavimo lentele\n" + "      0    " + "1\n"
      + "FALSE " + str(con_m[0][0]) + "   " + str(con_m[1][0]) +"\n" + "TRUE  " + str(con_m[0][1]) + "   " + str(con_m[1][1])+ "\n")
print("Klasifikavimo lentele\n" + "          0      " + "1\n" + "FALSE  " + str(round(con_m[0][0] / sum(con_m[0]) , 6)) + " " +
      str(round(con_m[1][0] / sum(con_m[1]), 6)) +"\n" + "TRUE   " + str(round(con_m[0][1] / sum(con_m[0]), 6)) + " "
      + str(round(con_m[1][1] / sum(con_m[1]), 6))+ "\n")
print("Patikimuno lygis = ", accuracy_score(y_true = dat1_test["Outcome"], y_pred= prd))
print( "False Negative rate = ", con_m[0][1] / sum(con_m[0]), "\nFalse Positive rate = ", con_m[1][0] / sum(con_m[1]))

# Optimaliojo slenkscio parinkimas

fpr, tpr, thresholds = roc_curve(y_true = dat1_test["Outcome"], y_score = pred)\

tf = tpr - (1 - fpr)
rc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "tf" : abs(tpr - (1 - fpr)), "threshold" : thresholds })
op_cutoff = rc.sort_values(by = "tf").iloc[0]
print(op_cutoff)

# ROC grafikas
fig = plt.figure(figsize=(10,10))
plt.title("Binarinio atsako modelio ROC kreivė")
plt.ylabel("True positive rate")
plt.xlabel("False positive rate")
plt.plot(fpr, tpr)
plt.annotate(text = "Slenkstis = {}".format(round(op_cutoff["threshold"],3)), xy = (op_cutoff["fpr"] + 0.01, op_cutoff["tpr"] - 0.02), c = "g")
plt.scatter(x = op_cutoff["fpr"], y = op_cutoff["tpr"], s = 100, c = "g")
plt.vlines(x = op_cutoff["fpr"],ymax=1, ymin=0, color = 'g', ls = "--")
plt.hlines(y = op_cutoff["tpr"],xmax=1, xmin=0, color = 'g', ls = "--")
rc["th05"] = abs(rc["threshold"] - 0.5)
cutoff_old = rc.sort_values(by = "th05").iloc[0]
plt.annotate(text = "Slenkstis = 0.5", xy = (cutoff_old["fpr"] + 0.01, cutoff_old["tpr"] - 0.02), c = "r")
plt.scatter(x = cutoff_old["fpr"], y = cutoff_old["tpr"], s = 100, c = "r")
plt.vlines(x = cutoff_old["fpr"],ymax=1, ymin=0, color = 'r', ls = "--")
plt.hlines(y = cutoff_old["tpr"],xmax=1, xmin=0, color = 'r', ls = "--")
plt.savefig("ROC.png")
plt.clf()

print("AUC = ", roc_auc_score(y_true = dat1_test["Outcome"], y_score = pred))

print("Optimaliausias slekstis = ",op_cutoff["threshold"])

# Galutinis tikslumo patikrinimas

prd1 = [0 if k < op_cutoff["threshold"] else 1 for k in pred]

con_m1 = confusion_matrix(y_true = dat1_test["Outcome"], y_pred= prd1)
print("Klasifikavimo lentele\n" + "      0    " + "1\n"
      + "FALSE " + str(con_m1[0][0]) + "   " + str(con_m1[1][0]) +"\n" + "TRUE  " + str(con_m1[0][1]) + "   " + str(con_m1[1][1])+ "\n")
print("Klasifikavimo lentele\n" + "          0      " + "1\n" + "FALSE  " + str(round(con_m1[0][0] / sum(con_m1[0]) , 6)) + " " +
      str(round(con_m1[1][0] / sum(con_m1[1]), 6)) +"\n" + "TRUE   " + str(round(con_m1[0][1] / sum(con_m1[0]), 6)) + " "
      + str(round(con_m1[1][1] / sum(con_m1[1]), 6))+ "\n")
print("Patikimuno lygis = ", accuracy_score(y_true = dat1_test["Outcome"], y_pred= prd1))
print( "False Negative rate = ", con_m1[0][1] / sum(con_m1[0]), "\nFalse Positive rate = ", con_m1[1][0] / sum(con_m1[1]))


