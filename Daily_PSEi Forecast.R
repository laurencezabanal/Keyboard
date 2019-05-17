#Prophet
#install.packages('Prophet')
# Importing the dataset

dataset = read.csv('PSEi_Prices.csv')

asDate <- as.Date(as.character(dataset$Date),format="%m/%d/%Y")
dataset[1] = asDate
cutoffDate = '2019-05-01'
training_set <- subset(dataset[dataset$Date< cutoffDate,])
testset <- subset(dataset[dataset$Date>= cutoffDate,])
ds <- as.Date((training_set$Date),format="%m/%d/%Y")
test.ds <- as.Date((testset$Date),format="%m/%d/%Y")
training_set2 <- rbind(training_set,testset)

library(prophet)

#1. TimeSeries for Volume
y_actual = testset[,2]
library(ggplot2)
qplot(Date,Last.Price,data=training_set)
training_set$Last.Price[training_set$Last.Price==0] <- NA
#y <- log(dataset$Volume)
y <- (training_set$Last.Price)
df <- data.frame(ds,y)
#df$CrudeOil_Price <- training_set$Stock_Price
#df$Stock_Price <- training_set$Stock_Price

qplot(ds,y,data=df)

#HolidaySetting
library(dplyr)
PH_holidays <- data.frame(
  holiday = 'PH Holiday',
  ds = as.Date(c('2017-01-01', '2017-12-25', 
                 '2018-01-01', '2018-12-25',
                 '2019-01-01','2019-05-01')),
  lower_window = -3,
  upper_window = 3
)
PH_Otherholidays <- data.frame(
  holiday = 'US Holiday',
  ds = as.Date(c('2019-01-01')),
  lower_window = -1,
  upper_window = 0
)
EOM<- data.frame(
  holiday = 'US Holiday',
  ds = as.Date(c('2017-01-31','2017-02-28','2017-03-31','2017-04-30',
                 '2017-05-31','2017-06-30','2017-07-31','2017-08-31',
                 '2017-09-30','2017-10-31','2017-11-30','2017-12-31',
                 '2018-01-31','2018-02-28','2018-03-31','2018-04-30',
                 '2018-05-31','2018-06-30','2018-07-31','2018-08-31',
                 '2018-09-30','2018-10-31','2018-11-30','2018-12-31',
                 '2019-01-31','2019-02-28')),
  lower_window = -7,
  upper_window = 7
)

ALL_Holidays <- rbind(PH_holidays)

#Forecasting with Facebook's Prophet
#Grid

changepoint.prior.scale <- seq(0.07,0.1, by=0.01)
seasonality.prior.scale <- seq(15,20, by=1)
n.changepoints <- seq(20,25, by=1)
daily.seasonality <- FALSE
yearly.seasonality <- FALSE
weekly.seasonality <- FALSE
monthly.seasonality <- FALSE

yearly.fourier_order<- 10
monthly.fourier_order<- 7
weekly.fourier_order <- 3

prophetGrid <- expand.grid(changepoint.prior.scale,seasonality.prior.scale,n.changepoints,daily.seasonality,yearly.seasonality,monthly.seasonality,
                           weekly.seasonality,yearly.fourier_order,monthly.fourier_order,weekly.fourier_order)
colnames(prophetGrid) <- c("changepoint.prior.scale","seasonality.prior.scale","n.changepoints","daily.seasonality","yearly.seasonality","month.seasonality",
                           "weekly.seasonality","yearly.fourier_order","monthly.fourier_order","weekly.fourier_order")
prophetGrid$MAPE <- 0

library(doParallel)
optimal <-
  foreach(i = 1:nrow(prophetGrid), .errorhandling = "stop",.combine='rbind', .packages = 'prophet') %dopar% {
    parameters = prophetGrid[i,]
    model = prophet(n.changepoints = parameters$n.changepoints, changepoint.prior.scale = parameters$changepoint.prior.scale, 
                    seasonality.prior.scale = parameters$seasonality.prior.scale, yearly.seasonality = parameters$yearly.seasonality,
                    weekly.seasonality =parameters$weekly.seasonality,daily.seasonality=parameters$daily.seasonality,monthly.seasonality=parameters$monthly.seasonality,holidays = ALL_Holidays)
    model<- add_seasonality(model,name='weekly',period=7,fourier.order =parameters$weekly.fourier_order)
    model<- add_seasonality(model,name='monthly',period=30.5, fourier.order = parameters$monthly.fourier_order)
    model<- add_seasonality(model,name='yearly',period=365.25,fourier.order =10)
    model <- fit.prophet(model, df) 
    
    future<- make_future_dataframe(model,periods = max(dataset$Date)-max(training_set$Date),freq ='day' )
    forecast <- predict(model,future)
    tail_train <- tail(forecast[c('ds','yhat','yhat_lower','yhat_upper')],n=as.numeric(max(dataset$Date)-max(training_set$Date)))
    tail_train[,1] <- as.Date((tail_train$ds),format="%m/%d/%Y")
    y_pred = tail_train[,2]
    y_actual <- testset[,2]
    
    Test_Eval<-data.frame(testset$Date,y_pred,y_actual,weekdays(as.Date(testset$Date)),testset$Weekend.Holiday)
    Test_Eval<- filter(Test_Eval,Test_Eval[,5] %in% c('0'))
    Test_Eval$Error <- 0
    counter=0
    for (row in 1:length(Test_Eval$y_actual)){
      counter= counter+1
      MAPE<- (abs(Test_Eval[row,3]-Test_Eval[row,2])/Test_Eval[row,3])*100
      Test_Eval[row,6]<- MAPE
    }
    
    MAPE<- sum(Test_Eval$Error)/counter

    print(i)
    print(MAPE)
    print(i/nrow(prophetGrid)*100)
    prophetGrid[i,12]<- MAPE
  }

prophetGrid$MAPE

optimized_hyperparameter<- prophetGrid[which.min(prophetGrid$MAPE), ]
optimized_hyperparameter

#Optimal Values
changepoint.prior.scale = 0.1
seasonality.prior.scale = 15
n.changepoints = 23
daily.seasonality=0
yearly.seasonality=0



#Optimal Values
changepoint.prior.scale = 0.09
seasonality.prior.scale = 17
n.changepoints = 23



#write.csv(prophetGrid, "ProphetTuning.csv")

model = prophet(n.changepoints = n.changepoints, changepoint.prior.scale = changepoint.prior.scale, 
                seasonality.prior.scale = seasonality.prior.scale, yearly.seasonality = FALSE,
                weekly.seasonality = FALSE,daily.seasonality=FALSE,monthly.seasonality=FALSE,holidays = ALL_Holidays)
model<- add_seasonality(model,name='weekly',period=7,fourier.order =3)
model<- add_seasonality(model,name='monthly',period=30.5, fourier.order = 7)
model<- add_seasonality(model,name='yearly',period=365.25,fourier.order =10)
model <- fit.prophet(model, df) 
#model<- add_seasonality(model,name='monthly',period=30.5,fourier.order = parameters$monthly.fourier_order)
#model<- add_seasonality(model,name='daily',period=1,fourier.order = parameters$daily.fourier_order)


future<- make_future_dataframe(model,periods = max(dataset$Date)-max(training_set$Date),freq ='day' )

tail(future,n=as.numeric(max(dataset$Date)-max(training_set$Date)))
forecast <- predict(model,future)
tail_train <- tail(forecast[c('ds','yhat','yhat_lower','yhat_upper')],n=as.numeric(max(dataset$Date)-max(training_set$Date)))
tail_train[,1] <- as.Date((tail_train$ds),format="%m/%d/%Y")
#tail_train2 <- filter(tail_train,weekdays(tail_train[,1]) %in% c('Saturday','Sunday'))
#tail_train <- filter(tail_train,weekdays(tail_train[,1]) %in% c('Monday','Tuesday','Wednesday','Thursday','Friday'))
#tail_train <- filter(tail_train,
y_pred = tail_train[,2]
y_actual <- testset[,2]

#Training vs testset
#accuracy(forecast(model, h = nrow(testset)), testset$Volume)

Test_Eval<-data.frame(testset$Date,y_pred,y_actual,weekdays(as.Date(testset$Date)),testset$Weekend.Holiday)
Test_Eval<- filter(Test_Eval,Test_Eval[,5] %in% c('0'))
Test_Eval$Error <- 0
counter=0
  for (row in 1:length(Test_Eval$y_actual)){
    counter= counter+1
    MAPE<- (abs(Test_Eval[row,3]-Test_Eval[row,2])/Test_Eval[row,3])*100
    Test_Eval[row,6]<- MAPE
  }

MAPE<- sum(Test_Eval$Error)/counter
#write.csv(Test_Eval,"PSEi Test.csv")
#CROSS VALIDATION

Horizon=90
cv_results = cross_validation(model,horizon=Horizon, units="days", period=1,initial=550)
cv_results$ds
cv_results$Weekend.Holiday <- 0
cv_results$ds <- as.Date((cv_results$ds),format="%m/%d/%Y")
Weekend_Holiday<- 
  for(row in 1:length(cv_results$ds)){
    for(row2 in 1:length(training_set2$Date)){
      if(training_set2[row2,1]==cv_results[row,1]){
        cv_results[row,7]=training_set2[row2,5]
      }else{
        row2=row2+1
      }
    }
  }

cv_results$MAPE<- 0
cv_counter=0
CV_MAPE_Computation <- 
  for (row in 1:length(cv_results$ds)){
    if(cv_results[row,7]==0){
      cv_counter= cv_counter+1
      MAPE<- (abs(cv_results[row,2]-cv_results[row,3])/cv_results[row,2])*100
      cv_results[row,8]<- MAPE
    }else{counter=counter
    }
  }

CV_MAPE<- sum(cv_results$MAPE)/cv_counter
CV_MAPE
cv_results1 = cross_validation(model,horizon=Horizon, units="days", period=1,initial=550)
performance<- performance_metrics(cv_results1)
plot_cross_validation_metric(cv_results1,metric='mape')
MSE <- mean((y_pred-y_actual)^2)
RMSE <- sqrt(MSE)

tail_train <- tail(forecast[c('ds','yhat','yhat_lower','yhat_upper')],n=789)
y_pred = tail_train[,2]
Forecast_SundayAdjustedToMonday <- data.frame(tail_train[,1],tail_train[,2],dataset[,4])

#write.csv(Forecast_SundayAdjustedToMonday,"Forecast_SundayAdjustedToMonday_Final.csv")

#Plot Forecast
plot(model,forecast)
dyplot.prophet(model, forecast)
prophet_plot_components(model, forecast)


#2019 Forecast
library(ggplot2)
qplot(Date,Last.Price,data=dataset)
dataset$Last.Price[dataset$Last.Price==0] <- NA
ds <- as.Date((dataset$Date),format="%m/%d/%Y")
y <- (dataset$Last.Price)
df <- data.frame(ds,y)
qplot(ds,y,data=df)

model = prophet(n.changepoints = n.changepoints, changepoint.prior.scale = changepoint.prior.scale, 
                seasonality.prior.scale = seasonality.prior.scale, yearly.seasonality = FALSE,
                weekly.seasonality = FALSE,daily.seasonality=FALSE,monthly.seasonality=FALSE,holidays = ALL_Holidays)
model<- add_seasonality(model,name='weekly',period=7,fourier.order =3)
model<- add_seasonality(model,name='monthly',period=30.5, fourier.order = 7)
model<- add_seasonality(model,name='yearly',period=365.25,fourier.order =10)
model <- fit.prophet(model, df) 

Target_Date <- as.Date('2019-12-31')
Forecast_period= Target_Date-max(dataset$Date)
future<- make_future_dataframe(model,periods = Forecast_period,freq ='day' )
forecast <- predict(model,future)
tail_train <- tail(forecast[c('ds','yhat','yhat_lower','yhat_upper')],n=as.numeric(Forecast_period)+length(dataset$Date))
y_pred = tail_train[,2]

plot(model,forecast)
dyplot.prophet(model, forecast)
#write.csv(tail_train, "Daily_PSEi_Forecast_v2.csv")
#write.csv(tail_train, "Daily_Prophet_Tuned_March-May2019.csv")
