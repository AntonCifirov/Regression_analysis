FILENAME REFFILE '/home/u63803163/diabetes.csv';

*Importuojame duomenys;
PROC IMPORT DATAFILE=REFFILE
	DBMS=CSV
	OUT=diabetes;
	GETNAMES=YES;

*Santikys 1 ir 0;
proc freq data=diabetes;
tables Outcome;
run;

*Sukuriame boxplotus;

Filename odsout '/home/u63803163/';
ods PDF body= "/home/u63803163/Relative.PDF" ;
ods layout Start columns=4 rows=2;

ods region row=1 column=1;
ods graphics / width= 3.5 in height=4 in;
proc sgplot data=diabetes;
vbox Pregnancies/ category=Outcome;
run;

ods region row=1 column=2;
proc sgplot data=diabetes;
vbox SkinThickness/ category=Outcome;
run;

ods region row=1 column=3;
proc sgplot data=diabetes;
vbox DiabetesPedigreeFunction/ category=Outcome;
run;

ods region row=1 column=4;
proc sgplot data=diabetes;
vbox Insulin/ category=Outcome;
run;

ods region row=2 column=1;
proc sgplot data=diabetes;
vbox Glucose/ category=Outcome;
run;

ods region row=2 column=2;
proc sgplot data=diabetes;
vbox BloodPressure/ category=Outcome;
run;

ods region row=2 column=3;
proc sgplot data=diabetes;
vbox Age/ category=Outcome;
run;

ods region row=2 column=4;
proc sgplot data=diabetes;
vbox BMI/ category=Outcome;
run;

quit;
ods layout end;
ods PDF close;



*Daliname duomenys i apmokymo ir testaviimo;
proc surveyselect data=diabetes rate=0.8 outall out=diabetes2 seed=1234;
run;
data train test; 
set diabetes2; 
if selected =1 then output train; 
else output test; 
drop selected;
run;


*Pirminis modelis;
PROC LOGISTIC DATA=train DESCENDING;
MODEL Outcome = Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age/
RSQUARE CTABLE PPROB=0.5;
  
*Nulines reiksmes;
DATA train_filtered;
    set train;
    where Glucose > 0 or BMI > 0 or BloodPressure > 0;

DATA test_filtered;
	set test;
	where Glucose > 0 or BMI > 0 or BloodPressure > 0;
run;

*Ismetami kintamieji;
PROC LOGISTIC DATA=train_filtered DESCENDING;
MODEL Outcome = Pregnancies Glucose BMI/
RSQUARE CTABLE PPROB=0.5;


*Pearson residuals vizualizacija

ODS GRAPHICS ON;
ods graphics / width= 6 in height=6 in;
PROC LOGISTIC DATA=train_filtered PLOTS(only)=influence;
MODEL Outcome = Pregnancies Glucose BMI;
run;



*Klasifikacijos matrica;

PROC LOGISTIC DATA=train_filtered outmodel=logmod;
MODEL Outcome = Pregnancies Glucose BMI;
run;

PROC LOGISTIC INMODEL=logmod;
    score data=test_filtered out = Predictions;
run;

PROC FREQ DATA = Predictions;
	tables F_Outcome * I_Outcome / ncol nrow;
run;


PROC FREQ DATA = Predictions;
	tables F_Outcome * I_Outcome / ncol nrow out = con_m;
run;
DATA out_0;
	set con_m;
	where F_Outcome = "0";
DATA out_1;
	set con_m;
	where F_Outcome = "1";

proc sql;
create table out_o as
select COUNT as Negative, COUNT / sum(COUNT) as Tikimybes
from out_0
quit;
proc sql;
create table out_i as
select COUNT as Positive, COUNT / sum(COUNT) as Tikimybes
from out_1;
quit;




*Roc kreive;
PROC LOGISTIC DATA=train_filtered DESCENDING plots(only)=roc;
MODEL Outcome = Pregnancies Glucose BMI;


*Optimalaus slenkscio parinkimas naudojant Youden indeksa;
PROC LOGISTIC DATA=train_filtered DESCENDING plots(only)=roc;
MODEL Outcome = Pregnancies Glucose BMI /scale=none
aggregate expb rsquare clparm=wald outroc=roc_kreive;
run;quit;
data roc_kreive;
set roc_kreive;
specificity=1 - _1MSPEC_;
Youden=_SENSIT_+specificity -1;
run;
proc sql;
select max(Youden) into :Youden from roc_kreive;
select _PROB_ into :slenkstis from roc_kreive where
Youden=(select max(Youden) from roc_kreive);
quit;
%put &=Youden &=slenkstis;

PROC LOGISTIC DATA=train_filtered ;
MODEL Outcome = Pregnancies Glucose BMI/
scale=none aggregate expb rsquare clparm=wald
outroc=roc_mokymo;
score data=test out=testine_progn outroc=roc_testine;
roc; roccontrast;
run;quit;

PROC LOGISTIC DATA=train_filtered DESCENDING outmodel = logmodel;
MODEL Outcome = Pregnancies Glucose BMI;
run;

proc logistic inmodel=logmodel;
    score data=test_filtered out=predictions;
run;

proc freq data=predictions;
	tables F_Outcome*I_Outcome/nocol norow;
run;


*Slenkstis = 0.350993;
data predictions1;
	set predictions;
	if P_1 > 0.350993 then I_Outcome = 1;
	else I_Outcome = 0;

proc freq data=predictions1;
	tables F_Outcome*I_Outcome/nocol norow;
run;

PROC LOGISTIC DATA=train_filtered ;
MODEL Outcome = Pregnancies Glucose BMI;
score data=test_filtered out = Predictions;
run;

proc sql;
create table Predictions_n as
select * , P_1 > 0.350993 as I_Outcome_n 
from Predictions
quit;


PROC FREQ DATA = Predictions_n;
	tables F_Outcome * I_Outcome_n / ncol nrow out = con_m;
run;

DATA out_0;
	set con_m;
	where F_Outcome = "0";
DATA out_1;
	set con_m;
	where F_Outcome = "1";

proc sql;
create table out_o as
select COUNT as Negative, COUNT / sum(COUNT) as Tikimybes
from out_0
quit;
proc sql;
create table out_i as
select COUNT as Positive, COUNT / sum(COUNT) as Tikimybes
from out_1;
quit;

