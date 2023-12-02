library(seasonal)
library(FNN)
library(forecast)
library(tseries)
library(timeSeries)
library(xts)
library(class)
library(data.table)
library(readr)
library(quantmod)
library(ggplot2)
library(stats)
library(caret)
library(tsfknn)
library(tsibble)
library(gridExtra)
library(lmtest)

#get the EUR/MXN exchange rate dataset from yahoo finance
exchange_rate_eur <- getSymbols("EURMXN=X", from = "2020-09-01", to = "2023-08-31", src = "yahoo",
                            auto.assign = F)[,6]
#For some reason the getSymbols command from the quantmod package sometimes shifts the dates up. The imported prices from
#2020-09-01 to 2023-08-31 are indeed correct. However, the dates are shifted 1 day up. This represents no problem as I always represent the 
# data as days and not dates
autoplot(exchange_rate_eur, main = 'Daily EUR/MXN', ylab ='Price') + theme_bw()

#Compute the test for stationarity.
adf.test(exchange_rate_eur) #It didn't pass the test.


#Show correlation between prices
acf(exchange_rate_eur, main = 'ACF of EUR/MXN') #The time series is not stationary because the p-Value 
#is far greater than 5%. This is clearly the case since the exchange rates represent a random walk.

#Take the first Difference of the time series
exchange_rate_eur_diff <- diff(exchange_rate_eur, lag =1)
exchange_rate_eur_diff <- na.omit(exchange_rate_eur_diff)
autoplot(exchange_rate_eur_diff, main = 'First Difference Order EUR/MXN') + theme_bw()

#ACF and PACF of first drift
acf_diff <- acf(exchange_rate_eur_diff, main = 'ACF of First Difference')
pacf_diff <- pacf(exchange_rate_eur_diff, main = 'PACF of First Difference')



adf.test(exchange_rate_eur_diff) #after applying the first order difference, the data becomes stationary, giving us green light to apply ARIMA.

#PLOT PRICES AND DIFF PRICES TOGETHER
# Plot 1: Daily EUR/MXN
plot_prices <- autoplot(exchange_rate_eur) +
  ggtitle('Daily EUR/MXN') + theme_bw() +
  labs(x = 'Days', y = 'EUR/MXN')

# Plot 2: First Difference Order EUR/MXN
plot_diff_prices <- autoplot(exchange_rate_eur_diff) +
  ggtitle('First Difference Order EUR/MXN') + theme_bw() +
  labs(x = 'Days', y = 'EUR/MXN')

# Combine plots side by side
grid.arrange(plot_prices, plot_diff_prices, ncol = 2)


#split data into train and test. I used this splittig ratio is at is widely used un numerous forecast research papers.
n <- round(0.20*length(exchange_rate_eur))
train_eur <- head((exchange_rate_eur), length((exchange_rate_eur))-n)
test_eur <- tail(exchange_rate_eur, n)
length(train_eur)
length(test_eur)


#train the ARIMA model and choose the best one with the smallest AIC
model_eur <- auto.arima(train_eur, trace = TRUE, seasonal = FALSE, approximation=FALSE)
plot(model_eur)
summary(model_eur)
#forecast test data after having trained the model
forecast_test_eur <- forecast(model_eur,h=n)
#Plot the complete exchange rate data with actuals and predicted test data
autoplot(forecast_test_eur, xlab='Days',ylab='EUR/MXN',main = 'ARIMA(1,1,1)+c Forecast on Test Set')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
print(forecast_test_eur)
accuracy(forecast_test_eur,test_eur)
summary(forecast_test_eur)
coeftest(model_eur) #Parameters Test ARIMA(1,1,1) with drift

##### MAIN ARIMA CANDIDATES 
#Candidate ARIMA (1,1,1) with drift
model_eur_arima_111_drift <- Arima(train_eur, order=c(1,1,1), include.drift = TRUE) #fit ARIMA(1,1,1) without drift (straight line is then predicted)
summary(model_eur_arima_111_drift)
forecast_arima_111_drift <-forecast(model_eur_arima_111_drift, h=n)
print(forecast_arima_111_drift)
autoplot(forecast_arima_111_drift, xlab='Days',ylab='EUR/MXN',main = 'ARIMA(1,1,1) Forecast on Test Set')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
print(forecast_test_eur)
coeftest(model_eur_arima_111_drift) #Parameters Test ARIMA(1,1,1) with drift

#Candidate ARIMA (1,1,1)
model_eur_arima_111 <- Arima(train_eur, order=c(1,1,1), include.drift = FALSE) 
plot(model_eur_arima_111)
summary(model_eur_arima_111)
forecast_arima_111 <-forecast(model_eur_arima_111, h=n)
print(forecast_arima_111)
autoplot(forecast_arima_111, xlab='Days',ylab='EUR/MXN')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
accuracy(forecast_arima_111,test_eur)
coeftest(model_eur_arima_111) #Parameters Test ARIMA(1,1,1)

#### OTHER ARIMA CANDIDATES
#Candidate ARIMA(2,1,1) with drift
model_2_1_1 <- Arima(train_eur, order = c(2, 1, 1), include.drift = TRUE)
summary(model_2_1_1)
forecast_test_arima_2_1_1 <- forecast(model_2_1_1,h=n)
autoplot(forecast_test_arima_2_1_1, xlab='Days',ylab='EUR/MXN')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
accuracy(forecast_test_arima_2_1_1,test_eur)
print(forecast_test_arima_2_1_1)
accuracy(forecast_test_arima_2_1_1,test_eur)

#Candidate ARIMA(1,1,2) with drift
model_1_1_2 <- Arima(train_eur, order = c(1, 1, 2), include.drift = TRUE)
summary(model_1_1_2)
forecast_test_arima_1_1_2 <- forecast(model_1_1_2,h=n)
autoplot(forecast_test_arima_1_1_2, xlab='Days',ylab='EUR/MXN')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
accuracy(forecast_test_arima_1_1_2,test_eur)
print(forecast_test_arima_1_1_2)

#Candidate ARIMA(0,1,0) WHITE NOISE WITH DRIFT 
model_0_1_0_drift <- Arima(train_eur, order = c(0, 1, 0), include.drift = TRUE)
summary(model_0_1_0_drift)
forecast_test_arima_0_1_0_drift <- forecast(model_0_1_0_drift,h=n)
autoplot(forecast_test_arima_0_1_0_drift, xlab='Days',ylab='EUR/MXN')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
accuracy(forecast_test_arima_0_1_0_drift,test_eur)
print(forecast_test_arima_0_1_0_drift)

#Candidate ARIMA(0,1,0) WHITE NOISE NO DRIFT
model_0_1_0 <- Arima(train_eur, order = c(0, 1, 0), include.drift = FALSE)
summary(model_0_1_0)
forecast_test_arima_0_1_0 <- forecast(model_0_1_0,h=n)
autoplot(forecast_test_arima_0_1_0, xlab='Days',ylab='EUR/MXN')+
  autolayer(ts(test_eur, start= length(train_eur)), series="Test Data") +theme_bw() +theme(legend.position = c(0.9, 0.9), legend.justification = c(1, 1))
accuracy(forecast_test_arima_0_1_0,test_eur)
print(forecast_test_arima_0_1_0)


#ARIMA(1,1,1) with drift correctly predicts the downward trend.
#measure training and test set accuracy
accuracy(forecast_test_eur, test_eur) #accuracy of ARIMA(1,1,1) with drift on test 
accuracy(forecast_arima_111, test_eur) #accuracy of ARIMA(1,1,1) on test set




#predictions as data frame to inspect forecasts and confidence intervals
arima_predictions <- as.data.frame(forecast_test_eur)
#create vector of only predicted prices
arima_predicted_prices <- arima_predictions$`Point Forecast`
#create vector of forecast error
forecast_errors <- test_eur-arima_predicted_prices
#table with test, predictions and errors
arima_test_predicted_error <- cbind(test_eur,arima_predicted_prices,forecast_errors)


#Residual analysis with Ljung Box test. Our models passed the test so they are suitable for forecasting.
checkresiduals(forecast_test_eur) #ARIMA (1,1,1) drift
checkresiduals(forecast_arima_111) #ARIMA(1,1,1)


#make DAYS AHEAD forecast with the preferred ARIMA(1,1,1) with drift. "_da means" days ahead
model_eur_da <- Arima(exchange_rate_eur, order= c(1,1,1), include.drift = TRUE, include.mean = FALSE) #fit arima (1,1,1) to the whole data
summary(model_eur_da)
forecast_model_eur_da <- forecast(model_eur_da)
print(forecast_model_eur_da)
plot(forecast_model_eur_da)

#make DAYS AHEAD forecast with ARIMA(1,1,1) without drift
arima_111_da <- Arima(exchange_rate_eur, order= c(1,1,1), include.drift = FALSE, include.mean = FALSE)
summary(arima_111_da)
forecast_arima_111_da <- forecast(arima_111_da)
print(forecast_arima_111_da)




#KNN analysis
#use first difference exchange rate to make predictions and convert it into time series format
autoplot(exchange_rate_eur_diff, main="First Difference Order EUR/MXN") + theme_bw()

#The predictors of the data, in other words x, will be the price yesterday, and the values to be predicted y, which is the price on the next day
#I therefore created another column with the first lag of the price
lag_1 <- lag(exchange_rate_eur_diff, 1)
combined_data <- cbind(exchange_rate_eur_diff, lag_1)
#split training data (train_X) 1-640 and testing data (test_X) from 641-800. 
train_X <- head((combined_data$EURMXN.X.Adjusted.1), length((combined_data$EURMXN.X.Adjusted.1))-n) #Train lags from differenced prices
#eliminate the NA from the first observation in the train.X data (which is the lag from the first value, being of course 0)
train_X[is.na(train_X)] <- 0
test_X <- tail(combined_data$EURMXN.X.Adjusted.1,n) #out of sample lags from differenced prices
train_Y <- head(exchange_rate_eur_diff, length(exchange_rate_eur_diff)-n) #Train differenced prices
test_Y <- tail(exchange_rate_eur_diff,n) #out of sample differenced prices

#set seed so that the knn model does not break the tie
set.seed(123)
#train the knn_model
knn_model <- knn.reg(train_X, test_X, y=train_Y, k=21)
#make a variable where all the the predicted values of the test set are going to be stored
knn_predictions_diff <- knn_model$pred
print(knn_predictions_diff)


#make them numeric to calculate error metrics
knn_predictions_diff_numeric <- knn_model$pred
test_X_numeric <- as.numeric(test_X)
#Since making predictions on the stationary series, whose values are extremely close to zero, the MAPE gets extremely biased.
mape_knn_diff <- mean(abs((test_X_numeric - knn_predictions_diff_numeric) / test_X_numeric)) * 100
cat("The KNN MAPE on the differenced series is",mape_knn_diff) #To this end, the predicted values of the differenced series are going to be added to the lags of the non differenced series.
#In other words, I need to do a iteration, where I add the first predicted difference of the test price (the direction) plus the actual price of the last training set.
#Then, that calculated price of the test set will be added by the second predicted difference of the test price, and so on.

#create a new empty vector to store the KNN Predictions without difference (The prices)
knn_predictions <- rep(NA,n)
#Specify that the first observation of this new vector will 
#be the forecasted KNN first difference + the last observation of the training set, which in other words is the 
#last seen observed price before making predictions
last_observation_train <- tail(train_eur,1)
first_observation_test <- head(knn_predictions_diff,1) #This is the very first predicted differenced price of the knn model
x1 <- first_observation_test+last_observation_train #The first prediction of the actual price is the prediction of the differenced price 
#in time t + the price last day (in other words the lag of the first test observation which is the last training observation)
knn_predictions[1] <- x1 #The vector of knn predicted prices will always start with this calculation above

# Get the length of the knn_predictions vector
n_knn <- length(knn_predictions)

# Loop through the knn_predictions to fill the missing values
for (i in 2:n_knn) {
  knn_predictions[i] <- knn_predictions_diff[i] + knn_predictions[i-1]
}
knn_predictions <- ts(knn_predictions)
autoplot(knn_predictions)

# Combine the predictions and actual values into a single time series object to plot them
combined_ts <- ts.union(test_eur = test_eur, knn_predictions = knn_predictions)

# Plot the combined time series
autoplot(combined_ts, xlab ='Days', ylab = 'EUR/MXN', main = '21-NN Forecast on Test Set') +
  scale_color_manual(name = "series", values = c("black", "blue"),
                     labels = c("Test", "21NN Predictions")) + theme_bw()+ theme(legend.position = c(0.6, 0.6), legend.justification = c(2.8,2.8))


# Make knn predicted values and test values as numeric vectors to calculate error metrics
test_eur_numeric <- as.numeric(test_eur)
knn_predictions_numeric <- as.numeric(knn_predictions)
#calculate RMSE
rmse_knn <- sqrt(mean((test_eur_numeric - knn_predictions_numeric)^2))
# Calculate MAPE
mape_knn <- mean(abs((test_eur_numeric - knn_predictions_numeric) / test_eur_numeric)) * 100
print(knn_predictions)
cat("The KNN Test MAPE is",mape_knn) 


#Grid search process to find the best k by getting the smallest RMSE
k_values <- 1:100 #search parameter

# create a vector that will store all rmse of all k values from the model
rmse_values <- numeric(length(k_values))

# Loop every k value (The whole code above will be repeated)
for (k in k_values) {
  # Train the knn_model with current k
  knn_model <- knn.reg(train_X, test_X, y=train_Y, k=k)
  
  # predictions
  knn_predictions_diff <- knn_model$pred
  
  # Create a new empty vector to store the KNN Predictions without difference
  knn_predictions <- rep(NA, n)
  
  # Specify that the FIRST observation of this new vector will 
  # be the forecasted KNN first difference + the last observation of the training set
  last_observation_train <- tail(train_eur, 1)
  first_observation_test <- head(knn_predictions_diff, 1)
  x1 <- first_observation_test + last_observation_train
  knn_predictions[1] <- x1
  
  # Loop through the knn_predictions to fill the missing values
  for (i in 2:n) {
    knn_predictions[i] <- knn_predictions_diff[i] + knn_predictions[i-1]
  }
  knn_predictions <- ts(knn_predictions)
  
  # Calculate RMSE
  knn_predictions_numeric <- as.numeric(knn_predictions)
  rmse_knn <- sqrt(mean((test_eur_numeric - knn_predictions_numeric)^2))
  
  # Store RMSE value for this k
  rmse_values[k] <- rmse_knn
}

# Find the k value with the lowest RMSE
best_k <- which.min(rmse_values)
best_rmse <- min(rmse_values)

cat("Best k:", best_k, "\n")
cat("Best RMSE:", best_rmse, "\n")

# print all values from every k
cat("RMSE values for each k:\n")
cat(rmse_values, "\n")

rmse_knn <- best_rmse

#create a plot of k values in the x axis and rmse in the y axis.
plot(k_values,rmse_values, type= "o",col="blue", pch=3, xlab= 'k', ylab= 'RMSE', main='RMSE of each k')

#Now perform "days ahead forecast" with the optimal k=21. "_da" means day ahead.
#I basically repeat the whole code above but with the only difference that the variables are for "day ahead"
#purposes. 
lag_1_da <- lag(exchange_rate_eur_diff, 1)
combined_data_da <- cbind(exchange_rate_eur_diff, lag_1_da)
#use all observations as training set
train_X_da <- combined_data_da$EURMXN.X.Adjusted.1  #This is the column of the predictors (lagged prices)
train_X_da[is.na(train_X_da)] <- 0 #eliminate the NA from the first observation in the train.X data (which is the lag from the very first value that should be 0)
train_Y_da <- exchange_rate_eur_diff # Targets of lagged prices


######NOTE: To predict tomorrows test response Y, we need to use as test X the predictor of tomorrow. We know, that the predictor tomorrow is our price today.
#train_Y_da contains the responses of the whole exchange rate data set and train_X_da the lags of it. 
#The predictor for tomorrow will be the response yesterday, meaning, that we will always use as test denoted as in "y", the last response in the variable train_Y_da.
#Intuitively, the predictor is the last price of the set and the training set should be the whole data set, so that knn looks at the resemblance between the price at hand
#with all the other prices in the exchange rate.

# Predict 1 day ahead. Train the KNN model using the extended dataset
y_t= tail(train_Y_da,1) #We will use as predictor the last differenced price of our data set, that is 0.5534
y_t1 <- knn.reg(train_X_da, test=y_t, y = train_Y_da, k = 21)$pred  #predict one day ahead by using as test the last differenced price of our data 
y_t1 <- as.numeric(y_t1) #make it numeric so that we can add it to our data set


train_X_da <- as.numeric(train_X_da)
#Predict 2 days ahead
train_X_da2 <- c(train_X_da,y_t) #y_t will be a lagged value of our prediction y_t+1. That is why we add y_t to the lagged prices column. 
#The predicted differenced price y_t1 will now be the last response observation of our set and therefore the predictor for tomorrow.
train_Y_da <- as.numeric(train_Y_da)
train_Y_da2 <- c(train_Y_da,y_t1) #we create a new colum of response to add our last predicted y_t1
y_t2 <- knn.reg(train_X_da2, test=y_t1, y=train_Y_da2, k=21)$pred #to predict 2 days ahead, we use the whole data set and the predicted price y_t1=0.4727 as test set.

###REPEAT THE WHOLE PROCESS
#Predict 3 days ahead
train_X_da3 <- c(train_X_da2,y_t1)
train_Y_da3 <- c(train_Y_da2,y_t2)
y_t3 <- knn.reg(train_X_da3, test=y_t2, y=train_Y_da3,k=21)$pred
#Predict 4 days ahead
train_X_da4 <- c(train_X_da3,y_t2)
train_Y_da4 <- c(train_Y_da3,y_t3)
y_t4 <- knn.reg(train_X_da4, test=y_t3, y=train_Y_da4,k=21)$pred
#Predict 5 days ahead
train_X_da5 <- c(train_X_da4,y_t3)
train_Y_da5 <- c(train_Y_da4,y_t4)
y_t5 <- knn.reg(train_X_da5, test=y_t4, y=train_Y_da5,k=21)$pred

#Predict 6 days ahead
train_X_da6 <- c(train_X_da5,y_t4)
train_Y_da6 <- c(train_Y_da5,y_t5)
y_t6 <- knn.reg(train_X_da6, test=y_t5, y=train_Y_da6,k=21)$pred

#Predict 7 days ahead
train_X_da7 <- c(train_X_da6,y_t5)
train_Y_da7 <- c(train_Y_da6,y_t6)
y_t7 <- knn.reg(train_X_da7, test=y_t6, y=train_Y_da7,k=21)$pred

#Predict 8 days ahead
train_X_da8 <- c(train_X_da7,y_t6)
train_Y_da8 <- c(train_Y_da7,y_t7)
y_t8 <- knn.reg(train_X_da8, test=y_t7, y=train_Y_da8,k=21)$pred

#Predict 9 days ahead
train_X_da9 <- c(train_X_da8,y_t7)
train_Y_da9 <- c(train_Y_da8,y_t8)
y_t9 <- knn.reg(train_X_da9, test=y_t8, y=train_Y_da9,k=21)$pred

#Predict 10 days ahead
train_X_da10 <- c(train_X_da9,y_t8)
train_Y_da10 <- c(train_Y_da9,y_t9)
y_t10 <- knn.reg(train_X_da10, test=y_t9, y=train_Y_da10,k=21)$pred


knn_da <- ts(c(y_t1,y_t2,y_t3,y_t4,y_t5,y_t6,y_t7,y_t8,y_t9,y_t10)) #knn_da means knn days ahead (prediction) of differenced prices
print(knn_da)
plot(knn_da)

#I will now iterate to show the predictions in prices and not predicted prices. Same logic as before.

#Vector for predicted prices
knn_da <- numeric(10)

# First predicted price
knn_da[1] <- tail(exchange_rate_eur, 1) + y_t1 #The last observed perice of our whole data set + the predicted differenced price (predicted direction)

# Loop
for (i in 2:10) {
  knn_da[i] <- knn_da[i - 1] + get(paste0("y_t", i))
}
plot(knn_da, type='l')
# Compare 10 days ARIMA and KNN Predictions


arima_111_da_drift <- as.data.frame(forecast_model_eur_da)[,1] #take only forecast values from arima with drift
arima_111_da <- as.data.frame(forecast_arima_111_da)[,1] #take only forecast values with arima without drift
#Comparison predictions 10 days ahead
cbind(arima_111_da_drift,arima_111_da,knn_da)


actual_prices_next_days <- getSymbols("EURMXN=X", from = "2023-09-01", to = "2023-09-14", src = "yahoo",
                                      auto.assign = F)[,6]
actual_prices_next_days <- as.ts(actual_prices_next_days)

cbind(arima_111_da_drift,arima_111_da, knn_da, actual_prices_next_days)

#compare out of sample confidence intervals ARIMA(1,1,1) with drift with actual prices
forecast_drift_intervals <- cbind(forecast_model_eur_da,actual_prices_next_days)
colnames(forecast_drift_intervals)[1:6] <- c("Point Forecast", "Low 80%", "High 80%", "Low 95%", "High 95%", "Market Prices")
colnames(forecast_drift_intervals)
forecast_drift_intervals <- as.data.frame(forecast_drift_intervals)

#compare out of sample confidence intervals ARIMA(1,1,1) with actual prices
forecast_intervals <- cbind(forecast_arima_111_da,actual_prices_next_days)
colnames(forecast_intervals)[1:6] <- c("Point Forecast", "Low 80%", "High 80%", "Low 95%", "High 95%", "Market Prices")
colnames(forecast_intervals)
forecast_intervals <- as.data.frame(forecast_intervals)


#PLOT OUT OF SAMPLE FORECASTS
matplot(cbind(arima_111_da_drift, arima_111_da, knn_da, actual_prices_next_days), 
        type = "l", col = c("blue", "red", "green", "black"),
        lwd = 2,
        xlab = "Days Ahead", ylab = "EUR/MXN")

# Adjusting plot margins
par(mar = c(4, 4, 4, 2))  # Set the margins for bottom, left, top, and right

# Adjusting legend position and size
legend("topleft", legend = c("ARIMA + c", "ARIMA", "KNN", "Actual Prices"),
       col = c("blue", "red", "green", "black"), lty = 1, cex = 0.6)



arima_111_da <- as.numeric(arima_111_da)
actual_prices_next_days <- as.numeric(actual_prices_next_days)

rmse_arima_da <- sqrt(mean((actual_prices_next_days - arima_111_da)^2))
rmse_knn_da <- sqrt(mean((actual_prices_next_days - knn_da)^2))

rmse_arima_da_drift_i <- numeric(10)
rmse_arima_da_i <- numeric(10)
rmse_knn_da_i <- numeric(10)

# Loop through i values from 1 to 10
for (i in 1:10) {
  rmse_arima_da_drift_i[i] <- sqrt(mean((actual_prices_next_days[1:i] - arima_111_da_drift[1:i])^2))
  rmse_arima_da_i[i] <- sqrt(mean((actual_prices_next_days[1:i] - arima_111_da[1:i])^2))
  rmse_knn_da_i[i] <- sqrt(mean((actual_prices_next_days[1:i] - knn_da[1:i])^2))
}

all_rmse_horizons <- cbind(rmse_arima_da_drift_i,rmse_arima_da_i,rmse_knn_da_i)

#PLOT RMSE OF OUT OF SAMPLE FORECASTS
matplot(all_rmse_horizons, 
        type = "l", col = c("blue", "red", "green"),
        lwd = 2,
        xlab = "Days Ahead", ylab = "EUR/MXN")

# Adjusting plot margins
par(mar = c(4, 4, 4, 2))  # Set the margins for bottom, left, top, and right

# Adjusting legend position and size
legend("topleft", legend = c("ARIMA + c", "ARIMA", "KNN"),
       col = c("blue", "red", "green"), lty = 1, cex = 0.6)


#Actual price of the mexican peso on the 01.09.2023 was 18.43 pesos per EURO. 
#ARIMA and KNN predicted 18.31742 pesos and 18.36102 pesos respectively.
#Both models forecasted that the price would go up next day, which is already a good sign, since in real life it actually increased,
#but arima had a higher forecast error. However, ARIMA(1,1,1) forecasted an upward trend for the next 10 days, which actually happened.
#KNN on the other side predicted a downward trend. Given that ARIMA predicted very well the downward trend of the test set
#and the upward trend on days ahead, it is more suitable for longer time horizons than KNN.
#Though, it is better to make one day ahead predictions due to the sensible nature of exchange rates.

#PLOT OUT OF SAMPLE FORECAST AND RMSE TOGETHER
par(mfrow = c(1, 2))  # 1 row, 2 columns

# Plot 1: Out of sample forecasts
matplot(cbind(arima_111_da_drift, arima_111_da, knn_da, actual_prices_next_days), 
        type = "l", col = c("green", "red", "blue", "black"),
        lwd = 2,
        main = 'Out of Sample Forecast',
        xlab = "Days Ahead", ylab = "EUR/MXN")

# Adjusting legend position and size for Plot 1 with shorter lines (e.g., lty = 2)
legend("topleft", legend = c("ARIMA(1,1,1) + c", "ARIMA(1,1,1)", "21-NN", "Actual Prices"),
       col = c("green", "red", "blue", "black"), lty = 2, cex = 0.56)

# Plot 2: RMSE of out-of-sample forecasts
matplot(all_rmse_horizons, 
        type = "l", col = c("green", "red", "blue"),
        lwd = 2,
        main = 'RMSE Out of Sample Forecast',
        xlab = "Days Ahead", ylab = "RMSE")

# Adjusting legend position and size for Plot 2 with shorter lines (e.g., lty = 2)
legend("topleft", legend = c("ARIMA(1,1,1) + c", "ARIMA(1,1,1)", "21-NN"),
       col = c("green", "red", "blue"), lty = 2, cex = 0.56)

# Reset the layout to default (1 plot per page) when done
par(mfrow = c(1, 1))


