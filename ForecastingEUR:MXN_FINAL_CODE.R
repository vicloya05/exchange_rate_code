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

#get the EUR/MXN exchange rate dataset from yahoo finance. 
exchange_rate_eur <- getSymbols("EURMXN=X", from = "2020-09-01", to = "2023-08-31", src = "yahoo",
                            auto.assign = F)[,6]

length(exchange_rate_eur)

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

#The predictors of the data, in other words x, will be the price yesterday, and the values to be predicted y are the targets
lag_1 <- lag(exchange_rate_eur_diff, 1) #create a column with lagged prices 
combined_data <- cbind(exchange_rate_eur_diff, lag_1) #combine differenced prices with its lags

#split training data (train_X) 1-640 and testing data (test_X) from 641-800. The training data consist on lags
#and is going to be tested on (today's prices).
train_X <- head((combined_data$EURMXN.X.Adjusted.1), length((combined_data$EURMXN.X.Adjusted.1))-n) #Train lags from differenced prices
train_X[is.na(train_X)] <- 0 #eliminate the NA from the first observation in the train.X data (which is the lag from the first value, being of course 0)
train_Y <- head(exchange_rate_eur_diff, length(exchange_rate_eur_diff)-n) #Train differenced prices (response of the lags in train_X)
test_X <- tail(combined_data$EURMXN.X.Adjusted.1,n) #test differenced lag prices
test_Y <- tail(exchange_rate_eur_diff,n) #test differenced prices (response of the lags in test_X)

#set seed so that the knn model does not break the tie
set.seed(123)
#train the knn_model
knn_model <- knn.reg(train_X, test_X, y=train_Y, k=21)
#make a variable where all the the predicted values of the test set are going to be stored
knn_predictions_diff <- knn_model$pred # predicted prices of test_Y
print(knn_predictions_diff)


#create a new empty vector to store the KNN Predictions without difference (We will now implement the reversal of the differentiation technique)
knn_predictions <- rep(NA,n)

#Specify that the first observation of this new vector will be the first KNN forecast test differenced price + the last observation of the training set.
last_observation_train <- tail(train_eur,1)
first_observation_test <- head(knn_predictions_diff,1) #This is the very first predicted test differenced price of the knn model
x1 <- first_observation_test+last_observation_train #The first prediction of the actual price is the prediction of the differenced price 
#in time t + the price last day.
knn_predictions[1] <- x1 #The vector of knn predicted prices will always start with this calculation above

# Get the length of the knn_predictions vector
n_knn <- length(knn_predictions)

# Loop through the knn_predictions to fill the missing values
for (i in 2:n_knn) {
  knn_predictions[i] <- knn_predictions_diff[i] + knn_predictions[i-1] #The last predicted price + the predicted direction for the next day
}
knn_predictions <- ts(knn_predictions) #make it a time series
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
print(knn_predictions)

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
#use all observations of the EUR/MXN data as training set
train_X_da1 <- combined_data_da$EURMXN.X.Adjusted.1  #This is the column of the predictors (lagged prices)
train_X_da1[is.na(train_X_da1)] <- 0 #eliminate the NA from the first observation in the train_X_da1 data (which is the lag from the very first value that should be 0)
train_Y_da1 <- exchange_rate_eur_diff # Targets of lagged prices in train_X_da1


######NOTE: To predict tomorrows test response Y, we need to use as test X the predictor of tomorrow.
#The predictor for tomorrow will be the response yesterday, meaning, that we will always use as test_X denoted as in "y", the last response in the variable train_Y_da.
#Intuitively, the predictor is the last price of the set and the training set (of the whole data set).

# Predict 1 day ahead. Train the KNN model using the extended dataset
y_t= tail(train_Y_da1 , 1) #We will use as test predictor the last differenced price of our data set, that is the last response of train_Y_da1 which has all the observations in the data set.
y_t1 <- knn.reg(train_X_da1 , test=y_t, y = train_Y_da1 , k = 21)$pred  #predict one day ahead by using as test the last differenced price of our data 
y_t1 <- as.numeric(y_t1) #make it numeric so that we can add it to our our response column. This at the same time will be used to predict the second day ahead as it is the last differenced price at hand.

train_X_da1 <- as.numeric(train_X_da1) #make this vector numeric to add the response y_t to the the lag column, since we are now going to predict 2 days ahead.
train_Y_da1 <- as.numeric(train_Y_da1)

#Predict 2 days ahead
train_X_da2 <- c(train_X_da1,y_t) #we add the response y_t which at the same time was used as a predictor, to the column of lag prices since we are now 2 days ahead
#The predicted differenced price y_t1 will now be the last response observation of our set and therefore the predictor for tomorrow.
train_Y_da2 <- c(train_Y_da1 , y_t1) #we add the predicted y_t1 to the column of response
y_t2 <- knn.reg(train_X_da2, test=y_t1, y=train_Y_da2, k=21)$pred #to predict 2 days ahead, we use the new whole data set and the predicted price y_t1=0.4727 as test set.

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
print(knn_da) #see predicted days ahead price differences 
plot(knn_da) 

#I will now iterate to show the predictions in prices and not predicted prices. Same logic as before.

#Vector for predicted prices
knn_da <- numeric(10)

# First predicted price
knn_da[1] <- tail(exchange_rate_eur, 1) + y_t1 #The last observed price of the exchange rate + the day ahead predicted differenced price (predicted direction)

# Loop
for (i in 2:10) {
  knn_da[i] <- knn_da[i - 1] + get(paste0("y_t", i))
}
print(knn_da)
plot(knn_da, type='l')
# Compare 10 days ARIMA and KNN Predictions


arima_111_da_drift <- as.data.frame(forecast_model_eur_da)[,1] #take only forecast values from arima with drift
arima_111_da <- as.data.frame(forecast_arima_111_da)[,1] #take only forecast values with arima without drift
#Comparison predictions 10 days ahead
cbind(arima_111_da_drift,arima_111_da,knn_da)


actual_prices_next_days <- getSymbols("EURMXN=X", from = "2023-09-01", to = "2023-09-14", src = "yahoo",
                                      auto.assign = F)[,6]
actual_prices_next_days <- as.ts(actual_prices_next_days)

comparison_predictions <- cbind(arima_111_da_drift,arima_111_da, knn_da, actual_prices_next_days)
colnames(comparison_predictions)[1:4] <- c("ARIMA(1,1,1) + c", "ARIMA(1,1,1)", "21-NN", "ACTUALS")
print(comparison_predictions)

#compare out of sample confidence intervals ARIMA(1,1,1) with drift with actual prices
forecast_drift_intervals <- cbind(forecast_model_eur_da,actual_prices_next_days)
colnames(forecast_drift_intervals)[1:6] <- c("Point Forecast", "Low 80%", "High 80%", "Low 95%", "High 95%", "Market Prices")
colnames(forecast_drift_intervals)
forecast_drift_intervals <- as.data.frame(forecast_drift_intervals)
print(forecast_drift_intervals)

#compare out of sample confidence intervals ARIMA(1,1,1) with actual prices
forecast_intervals <- cbind(forecast_arima_111_da,actual_prices_next_days)
colnames(forecast_intervals)[1:6] <- c("Point Forecast", "Low 80%", "High 80%", "Low 95%", "High 95%", "Market Prices")
colnames(forecast_intervals)
forecast_intervals <- as.data.frame(forecast_intervals)
print(forecast_intervals)


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


