FILENAME REFFILE '/home/u58687055/reglab/autompg.csv'; 

  

*Importuojame duomenys; 

PROC IMPORT DATAFILE=REFFILE 

DBMS=CSV 

OUT=data; 

GETNAMES=YES; 

/* Netinkamu reiksmiu naikinimas */
proc print data = data;
DATA data;
    set data;
    where horsepower > 0 ;  




/* Atsako kintamojo pasiskirstymo patikrinimas */
proc univariate data=data; 

    histogram mpg / gamma;  /*fit gamma distribution */ 

run; 


ods graphics / width= 8 in height=8 in;
proc univariate data=data; 

histogram mpg / gamma(shape = 9.031298 scale =2.596074);  /* Create histogram of data*/ 

run; 

proc univariate data=data; 

    histogram mpg / igauss;  /*fit gamma distribution */ 

run; 
ods graphics / width= 8 in height=8 in;
proc univariate data=data; 

histogram mpg / igauss(mu = 23.44592 lambda =193.3832);  /* Create histogram of data*/ 

run; 

*Pradine duomenu analize; 

Filename odsout '/home/u58687055/reglab/';
ods PDF body= "/home/u58687055/reglab/Relativ.PDF" ;
ods layout Start columns=4 rows=2;
ods graphics / width= 3.5 in height=4 in;


ods region row=1 column=1; 

proc sgplot data=data; 

vbox mpg /category= cylinders; 

run; 

ods region row=1 column=2; 
proc sgplot data=data; 

vbox mpg/ Category = origin; 

run; 

ods region row=1 column=3; 

proc sgplot data=data; 

scatter x = displacement y = mpg/; 

run; 

ods region row=1 column=4; 

proc sgplot data=data; 

scatter x = horsepower y = mpg/; 

run; 

ods region row=2 column=1; 

proc sgplot data=data; 

scatter x = weight y = mpg/; 

run; 

ods region row=2 column=2; 

proc sgplot data=data; 

scatter x = acceleration y = mpg/; 

run; 

ods region row=2 column=3; 

proc sgplot data=data; 

scatter x = model_year y = mpg/; 

run; 

quit; 

ods layout end; 

ods PDF close; 

* Papildome duomenys modifikuotais prediktoriais; 

data data; 

    set data;  /* Read data from existing dataset */ 

    log_dis = log(displacement); 

    log_hp = log(horsepower); 

    log_wg = log(weight); 

    log_acc = log(acceleration); 

    log_mpg = log(mpg); 
    

     

    in_dis = 1/(displacement); 

    in_hp = 1/(horsepower); 

    in_wg = 1/(weight); 

    in_acc = 1/(acceleration); 

    in_mpg = 1/(mpg); 

run; 


/* Papildomi grafikai */

Filename odsout '/home/u58687055/reglab/';
ods PDF body= "/home/u58687055/reglab/Relativ1.PDF" ;
ods layout Start columns=6 rows=3;
ods graphics / width= 2.4 in height= 2.5 in;

/* Be pakeitimu */
ods region row=1 column=1; 

proc sgplot data=data; 

vbox mpg /category= cylinders; 

run; 

ods region row=1 column=2; 

proc sgplot data=data; 

vbox mpg /category= origin; 

run; 


ods region row=1 column=3; 

proc sgplot data=data; 

scatter x = displacement y = mpg/; 

run; 

ods region row=1 column=4; 

proc sgplot data=data; 

scatter x = horsepower y = mpg/; 

run; 

ods region row=1 column=5; 

proc sgplot data=data; 

scatter x = weight y = mpg/; 

run; 

ods region row=1 column=6; 

proc sgplot data=data; 

scatter x = acceleration y = mpg/; 

run; 



/* log */

ods region row=2 column=1; 

proc sgplot data=data; 

vbox log_mpg /category= cylinders; 

run; 

ods region row=2 column=2; 

proc sgplot data=data; 

vbox log_mpg /category= origin; 

run; 


ods region row=2 column=3; 

proc sgplot data=data; 

scatter x = log_dis y = log_mpg/; 

run; 

ods region row=2 column=4; 

proc sgplot data=data; 

scatter x = log_hp y = log_mpg/; 

run; 

ods region row=2 column=5; 

proc sgplot data=data; 

scatter x = log_wg y = log_mpg/; 

run; 

ods region row=2 column=6; 

proc sgplot data=data; 

scatter x = log_acc y = log_mpg/; 

run; 


/* inv */

ods region row=3 column=1; 

proc sgplot data=data; 

vbox in_mpg /category= cylinders; 

run; 

ods region row=3 column=2; 

proc sgplot data=data; 

vbox in_mpg /category= origin; 

run; 


ods region row=3 column=3; 

proc sgplot data=data; 

scatter x = in_dis y = in_mpg/; 

run; 

ods region row=3 column=4; 

proc sgplot data=data; 

scatter x = in_hp y = in_mpg/; 

run; 

ods region row=3 column=5; 

proc sgplot data=data; 

scatter x = in_wg y = in_mpg/; 

run; 

ods region row=3 column=6; 

proc sgplot data=data; 

scatter x = in_acc y = in_mpg/; 

run; 

quit; 

ods layout end; 

ods PDF close;







/* Koreliaciju lentele */


data data;
    set data;
    orig_1 = (origin = 1); /* Sets orig_1 to 1 if origin is 1, else sets it to 0 */
    orig_2 = (origin = 2); /* Sets orig_2 to 1 if origin is 2, else sets it to 0 */
run;

proc corr data=data;
    var orig_1 orig_2 cylinders displacement horsepower weight acceleration;
run;



/* Išimčių analizė */
data data;
    set data;
    origin_cat =   put(origin, 8.); /* Apply the custom format */
run;


ods graphics / width= 7 in height=7 in;
proc genmod data=data plots=all;
	CLASS origin_cat;
    model mpg = origin_cat weight acceleration / DIST = GAMMA LINK = identity;
    output out=residuals stdresdev=std_residual  p=predicted cooksd=cooksd;
run;

Filename odsout '/home/u58687055/reglab/';
ods PDF body= "/home/u58687055/reglab/Relativ1.PDF" ;
ods layout Start columns=2 rows=2;
ods graphics / width= 4 in height= 4 in;

/* Scatterplot of standardized residuals vs fitted values */
ods region row=1 column=1; 
data residuals;
    set residuals;
    w = 1;
    ID = _N_;
    res = (mpg - predicted);
    rs = (res + predicted);
run;
proc sgplot data=residuals;
    scatter x=predicted y=std_residual / markerattrs=(symbol=circlefilled);
    xaxis label="Fitted Values";
    yaxis label="Standardized Residuals";
    refline 0 / lineattrs=(color=red);
run;
/* Working residuals */
ods region row=1 column=2; 
proc sgplot data=residuals;
    scatter x=predicted y=rs/ markerattrs=(symbol=circlefilled);
    xaxis label="Linear predictor, eta";
    yaxis label="Working residuals";
    refline 0 / lineattrs=(color=red);
run;
/* Q-Q plot of qresiduals */
proc sql;
create table residuals as
select *, sum((w * ((mpg - predicted) / predicted) ** 2) / 387) as dispersion
from residuals;
quit;

data residuals;
    set residuals;
    p = (cdf("Gamma", ((w * mpg) / predicted / dispersion), ( w / dispersion)));
    rand_qresidual = quantile("Normal", p);
run;

ods region row=2 column=1; 
proc univariate data=residuals normal;
    qqplot rand_qresidual / square;
    ods select QQPlot;
run;
/* Cook's distance plot */

ods region row=2 column=2; 
proc sgplot data=residuals;
    vbar ID / response=cooksd fillattrs=(color=blue);
    xaxis DISPLAY=NONE;
    yaxis label="Cook's Distance";
run;
quit; 

ods layout end; 

ods PDF close; 

 
 

data folds;
    set data;
    fold = mod(_N_, 10) + 1; /* Assign fold number */
run;

proc sql;
CREATE TABLE Results
    (NAME VARCHAR,
    AIC FLOAT,
     MSE FLOAT);

/* Gama identity */
 
proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
run; 


proc sql;
CREATE TABLE SE
    (SE FLOAT);
 
/*  i = 1    */
data train test;
    set folds;
    if fold ne 1 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 2    */
data train test;
    set folds;
    if fold ne 2 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 3    */
data train test;
    set folds;
    if fold ne 3 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 4    */
data train test;
    set folds;
    if fold ne 4 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 5    */
data train test;
    set folds;
    if fold ne 5 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;

/*  i = 6    */
data train test;
    set folds;
    if fold ne 6 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 7    */
data train test;
    set folds;
    if fold ne 7 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 8    */
data train test;
    set folds;
    if fold ne 8 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 9    */
data train test;
    set folds;
    if fold ne 9 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 10    */
data train test;
    set folds;
    if fold ne 10 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=gamma link=identity;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;

proc print data=MSE;
run;

proc sql;
   insert into Results
   select "Gamma_ide" as NAME, 2159.6129 as AIC, mean(SE) as MSE from SE;
   
/* Gama log */
 
proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
run; 




proc sql;
CREATE TABLE SE
    (SE FLOAT);
 
/*  i = 1    */
data train test;
    set folds;
    if fold ne 1 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;


proc sql;
   insert into SE
   select SE from testout;



/*  i = 2    */
data train test;
    set folds;
    if fold ne 2 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 3    */
data train test;
    set folds;
    if fold ne 3 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
	model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 4    */
data train test;
    set folds;
    if fold ne 4 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 5    */
data train test;
    set folds;
    if fold ne 5 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;

/*  i = 6    */
data train test;
    set folds;
    if fold ne 6 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 7    */
data train test;
    set folds;
    if fold ne 7 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 8    */
data train test;
    set folds;
    if fold ne 8 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 9    */
data train test;
    set folds;
    if fold ne 9 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 10    */
data train test;
    set folds;
    if fold ne 10 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;

proc print data=MSE;
run;


proc sql;
   insert into Results
   select "Gamma_log" as NAME, 2122.9299 as AIC, mean(SE) as MSE from SE;
   
/* Inv Gaussian id */
 
proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=IGAUSSIAN link=identity;
    output out=model_fit predicted=predicted;
run; 



proc sql;
CREATE TABLE SE
    (SE FLOAT);

/*  i = 1    */
data train test;
    set folds;
    if fold ne 1 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=Igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 2    */
data train test;
    set folds;
    if fold ne 2 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 3    */
data train test;
    set folds;
    if fold ne 3 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 4    */
data train test;
    set folds;
    if fold ne 4 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 5    */
data train test;
    set folds;
    if fold ne 5 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;

/*  i = 6    */
data train test;
    set folds;
    if fold ne 6 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;


/*  i = 7    */
data train test;
    set folds;
    if fold ne 7 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 8    */
data train test;
    set folds;
    if fold ne 8 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 9    */
data train test;
    set folds;
    if fold ne 9 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 10    */
data train test;
    set folds;
    if fold ne 10 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat weight acceleration / dist=igaussian link=identity;
    output out=model_fit predicted=predicted;
    store inv_id;
run;


proc plm restore=inv_id;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - predicted)**2;
run;


proc sql;
   insert into SE
   select SE from testout;

proc sql;
   insert into Results
   select "iGaus_ide" as NAME, 2157.2300 as AIC, mean(SE) as MSE from SE;
   
         
   
/* Inv Gaussian log */
 
 
proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=IGAUSSIAN link=log;
    output out=model_fit predicted=predicted;
run; 




proc sql;
CREATE TABLE SE
    (SE FLOAT);
 
/*  i = 1    */
data train test;
    set folds;
    if fold ne 1 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;


proc sql;
   insert into SE
   select SE from testout;



/*  i = 2    */
data train test;
    set folds;
    if fold ne 2 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 3    */
data train test;
    set folds;
    if fold ne 3 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
	model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 4    */
data train test;
    set folds;
    if fold ne 4 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 5    */
data train test;
    set folds;
    if fold ne 5 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;

/*  i = 6    */
data train test;
    set folds;
    if fold ne 6 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 7    */
data train test;
    set folds;
    if fold ne 7 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 8    */
data train test;
    set folds;
    if fold ne 8 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 9    */
data train test;
    set folds;
    if fold ne 9 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;



/*  i = 10    */
data train test;
    set folds;
    if fold ne 10 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;

proc sql;
   insert into SE
   select SE from testout;

proc print data=MSE;
run;


proc sql;
   insert into Results
   select "iGaus_log" as NAME, 2110.8295 as AIC, mean(SE) as MSE from SE;
   
/* Inv Gaussian inverse */
 
 
proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=IGAUSSIAN link=log;
    output out=model_fit predicted=predicted;
run; 




proc sql;
CREATE TABLE SE
    (SE FLOAT);
 
/*  i = 1    */
data train test;
    set folds;
    if fold ne 1 then output train;
    else output test;
run;

proc genmod data=train;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=igaussian link=log;
    output out=model_fit predicted=predicted;
    store ga_log;
run;


proc plm restore=ga_log;
   score data=test out=testout predicted;
run;

data testout;
    set testout;
    SE = (mpg - exp(predicted))**2;
run;


proc sql;
   insert into SE
   select SE from testout;   
   
 proc print data=results;
run;









/* Inv Gaussian log */

proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=IGAUSSIAN link=log;
    output out=model_fit predicted=predicted;
run; 

/* Gama log */
 
proc genmod data=data;
	class origin_cat;
    model mpg = origin_cat log_wg log_acc / dist=gamma link=log;
    output out=model_fit predicted=predicted;
run; 
