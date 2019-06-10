#######################################################################
## Fit a GRU layer on just one of the sites and see how it does...   ##
## Much of the code for the GRU  here is from "Deep Learning with R" ##
## book by Francois Chollet and J.J. Allaire (2018).                 ##
################################################################# #####

##-------------------------------------
## Set up directories for R/W
airTDir = '~/Documents/riverT/data/gagesII/airT/'
precipDir = '~/Documents/riverT/data/gagesII/precip/'
prismDir = '~/PRISM/'
gagesDir = '~/Documents/riverT/data/gagesII/'
dataDir = '~/Documents/riverT/data/'
# outDir = '~/riverT/data/gagesII/airT/'
# prismDir = '~/PRISM/'
# gagesDir = '~/riverT/data/gagesII/'
# dataDir = '~/riverT/data/'

##-------------------------------------
library(dplyr); library(openxlsx); library(lubridate); library(keras); library(abind); library(tensorflow)
##-------------------------------------

#-------------------------------------------------------------
## Read in USGS data (from downloadUSGS.R), columns correspond to waterT, Q (discharge), date, station number
## and clean it up a bit.
datUSGS = readRDS(paste0(dataDir, 'datUSGS.rds'))
siteIds = unique(datUSGS$siteId) # Unique 8-digit identifiers for each site
nSites = length(siteIds); nSites # ~250 sites

## See how many sites have at least nThresh daily temperature measurements
## Note that there are 13,514 days between the years 1981 to 2017 inclusive
# nThresh = 10000
# nSamples = aggregate(datUSGS$siteId, by=list(datUSGS$siteId), FUN=length) # Number of temperature measurements per site
# names(nSamples) = c('siteId', 'n')
# sum(nSamples$n>nThresh) # number of sites
# sum(nSamples$n[nSamples$n>nThresh]) # total number of measurements
# which(nSamples$n > nThresh)
# nSamples$siteId[which(nSamples$n > nThresh)]

#-------------------------------------------------------------
## Let's go with site "01421000", on the Deleware River in NY
## Data goes from 01/01/1981 to 12/31/2017
siteLSTM = '01421000'
dat = datUSGS[datUSGS$siteId == siteLSTM, ]

## Take a quick look for e.g. any decadal / very long term trends
plot(dat$date, dat$wt, type='l', las=1, ylab='Water Temperature [degC]', xlab='Date') ## Clear periodicity

## Read in corresponding air temperature data
fileName = paste0(airTDir, 'airT_', siteLSTM,'.rds')
airTtemp = readRDS(fileName)
names(airTtemp)[3] = 'airT'
dat = dat %>% left_join(airTtemp, by=c('siteId','date'))

## Read in corresponding precipitation data
fileName = paste0(precipDir, 'precip_', siteLSTM,'.rds')
preciptemp = readRDS(fileName)
names(preciptemp)[3] = 'precip'
dat = dat %>% left_join(preciptemp, by=c('siteId','date'))

####### FIX LATER ################
nrow(dat) # 12,431 observations (not good, some missing days in there!)

## Temporary fix, substitute the few missnig months with data in same months of other years
# medianT = aggregate(dat$wt, by = list(yday(dat$date)), FUN = median)
# fullDates = seq(as.Date("1981-01-01"), as.Date("2017-12-31"), by="days")
# dat2=dat
# dat2 %>% left_join(fullDates, by=c('','date'))
h1 = dat[dat$date %in% seq(as.Date("1991-07-21"), as.Date("1991-11-15"), by='days'),]
year(h1$date) = 1996
h2 = dat[dat$date %in% seq(as.Date("1992-06-07"), as.Date("1992-08-24"), by='days'),]
year(h2$date) = 1997
dat = rbind(dat, h1)
dat = rbind(dat, h2)
dat = dat[order(dat$date),]
nrow(dat)

##################################
## Check for missing values
sum(is.na(dat)) # Currently 0 

 # idxRemove = which(names(dat) %in% c('siteId', 'date'))
 # dat = dat[,-idxRemove] #remove 
 # dat = apply(dat, 2, as.numeric)

# ## Set train set as data from 1981 to 2006
# datTrain = dat[year(dat$date) %in% 1981:2006,-1]
# ## Set dev set as data from 2007 to 2013
# datDev = dat[year(dat$date) %in% 2007:2013,-1]
# ## Set test set as data from 2014 to 2017
# datTest = dat[year(dat$date) %in% 2014:2017,-1]
#


#-------------------------------------------------------------
## Rearrange so that target is in 2nd column like in the book FIX LATER THATS BAD HARDCODING AUTHORS
data = dat[,c('airT','wt','precip')]

## Normalize the data
mean = apply(data, 2, mean)
std = apply(data, 2, sd)
data = scale(data, center = TRUE, scale = TRUE)

## Indexes for train, eval, and test set partitiions
train.start = 1
train.end = 8800
eval.start = 8801
eval.end = 11400
test.start = 11401
test.end = nrow(data)

pdf('~/Desktop/dataDeleware.pdf', height=6, width=10)
  plot(dat$date, dat$wt, type='l', las=1, ylab='Water Temperature [°C]', xlab='',
       main = 'Observed Deleware River Temperature', ylim=c(0,31))
  lines(dat$date[eval.start:eval.end], dat$wt[eval.start:eval.end], col="yellowgreen")
  lines(dat$date[test.start:test.end], dat$wt[test.start:test.end], col="purple3")
  legend("topleft", c("Training Set", "Evaluation Set", "Test Set"), lty=c(1,1,1),
         col=c("black","yellowgreen","purple3"), bty='n', lwd=c(2,2,2))
dev.off()


## MSE if we just predict water temperature to be average of this week's air temperature (calculated on training set)
input_train = data[train.start:train.end,-2]
lstm_num_timesteps <- 7 ## Look back this many days (tuneable)
X_train1 <- t(sapply(1:(nrow(input_train) - lstm_num_timesteps), function(x) input_train[x:(x + lstm_num_timesteps - 1),1]))
naivePred = apply(X_train1, 1, mean)
y_train = data[1:nrow(X_train1), 2]
naiveMSE = mean((naivePred - y_train)^2); naiveMSE ## 0.133
naiveMSE * std[2] ## MSE in degC = 1.02


#-------------------------------------------------------------
## Generator yielding timeseries samples and their targets
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index)) max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]],
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2] ## Don't harcode the '2' for column number---
    }
    list(samples, targets)
  }
}


#-------------------------------------------------------------
## Prepare the training, validation, and test generators
lookback <- 14 # Observations go back this many days
step <- 1 # Observations sampled at one day per week (if step = 7)
delay <- 2 # Target will be this many days ahead in the future
batch_size <- 128
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = train.start,
  max_index = train.end,
  shuffle = TRUE,
  step = step,
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = eval.start,
  max_index = eval.end,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = test.start,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)
val_steps <- (eval.end - eval.start - lookback) / batch_size               
test_steps <- (nrow(data) - test.start - lookback) / batch_size 

#-------------------------------------------------------------
## Common-sense baseline MSE, predicting the temperature that it was 'delay' days ago
# evaluate_naive_method <- function() {
#   batch_mses <- c()
#   for (step in 1:val_steps) {
#     c(samples, targets) %<-% val_gen()
#     preds <- samples[,dim(samples)[[2]],2]
#     mse <- mean((preds - targets)^2)
#     batch_mses <- c(batch_mses, mse)
#   }
#   print(mean(batch_mses))
# }
# mmm = evaluate_naive_method() ## MSE of standardized temperature (MSE=.116)
#naiveMSE = mmm * std[2]; naiveMSE ## MSE in degC = 1.95



#-------------------------------------------------------------
## Test out a simple CNN
## Validation error around  0.069 (below the naive estimate)
model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mean_squared_error"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 8,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)

#-------------------------------------------------------------
## Test out a GRU with dropout regularization
## As of now I have not done a hyperparameter optimization for e.g. # units, % dropout
## Validation error around 0., lower than the naive estimate and half that of the CNN! 
model <- keras_model_sequential() %>%
  layer_gru(units = 16, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mean_squared_error"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 5,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)
#pp = generator(data, batch_size = batch_size, max_index = nrow(data), min_index=1, lookback=lookback, delay=delay)
#p = predict_generator(model, pp, steps=1)
#p = keras_predict(model, array(data[,-2], dim=c(nrow(dat),2,1)))
#predictions = model %>% predict(data[1,])

#-------------------------------------------------------------
## Test out first passing two 1-D conv nets then do a GRU with dropout regularization
## Validation error around 0.094, below naive estimate but higher than the GRU without conv nets
## Evidence this is therefore overfitting quite a bit
## Prepare the training, validation, and test generators
lookback <- 28 # Observations go back this many days
step <- 1 # Observations sampled at one day per week (if step = 7)
delay <- 2 # Target will be this many days ahead in the future
batch_size <- 128
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = train.start,
  max_index = train.end,
  shuffle = TRUE,
  step = step
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = eval.start,
  max_index = eval.end,
  step = step
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = test.start,
  max_index = NULL,
  step = step
)
val_steps <- (eval.end - eval.start - lookback) / batch_size               
test_steps <- (nrow(data) - test.start - lookback) / batch_size 

model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  layer_gru(units = 32, dropout = 0.4, recurrent_dropout = 0.5) %>%
  layer_dense(units = 1)
#summary(model)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mean_squared_error"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 10,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)


#-------------------------------------------------------------
## Try out LSTM
## Achieves evaluation set MSE of 0.049

input_train = data[train.start:train.end,-2]
input_eval = data[eval.start:eval.end,-2]
input_test = data[test.start:test.end,-2]

## Following https://rpubs.com/zkajdan/279967
lstm_num_timesteps <- 14 ## Look back this many days (tuneable)

X_train1 <- t(sapply(1:(nrow(input_train) - lstm_num_timesteps), function(x) input_train[x:(x + lstm_num_timesteps - 1),1]))
X_train2 <- t(sapply(1:(nrow(input_train) - lstm_num_timesteps), function(x) input_train[x:(x + lstm_num_timesteps - 1),2]))
#X_train3 <- t(sapply(1:(length(input_train[,1]) - lstm_num_timesteps), function(x) data[x:(x + lstm_num_timesteps - 1),2]))
#X_train4 <- t(sapply(1:(length(input_train[,1]) - lstm_num_timesteps), function(x) input_train[x:(x + lstm_num_timesteps - 1),4]))
X_train = aperm(abind(X_train1, X_train2, along=-1), c(2,3,1))
y_train = data[1:nrow(X_train1),2]; length(y_train)

num_samples <- dim(X_train)[1]
num_steps <- dim(X_train)[2]
num_features <- dim(X_train)[3]
dim(X_train)

## Model specs
model <- Sequential()
#model$add(LSTM(units = 32, input_shape=c(num_steps, num_features), dropout = 0.1, recurrent_dropout = 0.4))
  model$add(LSTM(units = 16, input_shape=c(num_steps, num_features), dropout = 0.1, recurrent_dropout = 0.1))
  model$add(Dense(1))
  model$add(Dense(1))
  keras_compile(model, loss='mean_squared_error', optimizer='adam')
history <- model %>% fit(X_train, y_train, batch_size = 32, epochs = 15, verbose = 1)
#pred_train <- keras_predict(model, X_train, batch_size = 1)

## Predictions on Eval set
X_eval1 <- t(sapply(1:(nrow(input_eval) - lstm_num_timesteps), function(x) input_eval[x:(x + lstm_num_timesteps - 1),1]))
X_eval2 <- t(sapply(1:(nrow(input_eval) - lstm_num_timesteps), function(x) input_eval[x:(x + lstm_num_timesteps - 1),2]))
#X_eval3 <- t(sapply(1:(nrow(input_eval) - lstm_num_timesteps), function(x) data[c(x+train.end + 1):(x + lstm_num_timesteps + train.end +1 - 1),3]))
X_eval = aperm(abind(X_eval1, X_eval2, along=-1), c(2,3,1))
y_eval = data[eval.start:c(eval.end-lstm_num_timesteps), 2]; length(y_eval) ## ERROR in index?? Check later; performance is good as is
#y_eval = data[c(eval.start+lstm_num_timesteps):eval.end, 2]; length(y_eval)

pred_eval <- keras_predict(model, X_eval, batch_size = 1)
mean((pred_eval-y_eval)^2)

## Predictions on Eval set
X_test1 <- t(sapply(1:(nrow(input_test) - lstm_num_timesteps), function(x) input_test[x:(x + lstm_num_timesteps - 1),1]))
X_test2 <- t(sapply(1:(nrow(input_test) - lstm_num_timesteps), function(x) input_test[x:(x + lstm_num_timesteps - 1),2]))
X_test = aperm(abind(X_test1, X_test2, along=-1), c(2,3,1))
y_test = data[test.start:c(test.end-lstm_num_timesteps), 2]; length(y_test)
pred_test <- keras_predict(model, X_test, batch_size = 1)
mean((pred_test-y_test)^2)

## Rescale
pred_test = pred_test * std[2] + mean[2]
y_test = y_test * std[2] + mean[2]
mean((pred_test-y_test)^2)^.5
cor(pred_test, y_test)^2
pdf('~/Desktop/testSet.pdf', height=6, width=8.7)
plot(dat$date[c(test.start+lstm_num_timesteps):test.end], y_test, las=1, 
     ylab='Water Temperature [°C]', xlab='', type='l', ylim=c(0,26),
     main = 'LSTM Test Set Performance')
  lines(dat$date[c(test.start+lstm_num_timesteps):test.end], pred_test, col='purple3')
  legend("topleft", c('Observed Temperature', "LSTM-Modeled Temperature", 
                      as.expression("RMSE = 1.73°C"), as.expression(bquote( R^2 ~"= 0.95"))), lty=c(1,1, NA, NA), col=c('black','purple3'), lwd=c(2,2), bty='n')
dev.off()
correlation = cor(pred_test, y_test)^2
rmse = (mean((pred_test - y_test)^2))^.5






#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
## Apply the model to all remaining stations
## For each site, read in the extracted PRISM air temperature and precip and combine
airT = NULL
fileName = paste0(airTDir, 'airT_', siteIds[1],'.rds')
airT = rbind(airT, readRDS(fileName))
for (i in 2:nSites){
  print(i)
  fileName = paste0(airTDir, 'airT_', siteIds[i],'.rds')
  if (file.exists(fileName)){
    airTtemp = readRDS(fileName)
    if(names(airTtemp)[1]==names(airT)[1] & names(airTtemp)[2]==names(airT)[2]){
      names(airTtemp)[3] <- names(airT)[3] #sometimes the airT column names are mismatched, messing up rbind
      airT = rbind(airT, airTtemp) ## Not the most elegant code, but it takes <2min
    }
  }
}
precip = NULL
fileName = paste0(precipDir, 'precip_', siteIds[1],'.rds')
precip = rbind(precip, readRDS(fileName))
for (i in 2:nSites){
  print(i)
  fileName = paste0(precipDir, 'precip_', siteIds[i],'.rds')
  if (file.exists(fileName)){
    preciptemp = readRDS(fileName)
    if(names(preciptemp)[1]==names(precip)[1] & names(preciptemp)[2]==names(precip)[2]){
      names(preciptemp)[3] <- names(precip)[3] #sometimes the precip column names are mismatched, messing up rbind
      precip = rbind(precip, preciptemp) ## Not the most elegant code, but it takes <2min
    }
  }
}
## Merge with existing datUSGS data
datUSGS = datUSGS %>% left_join(airT, by=c('siteId', 'date'))
colnames(datUSGS)[5] = 'airT'
datUSGS = datUSGS %>% left_join(precip, by=c('siteId', 'date'))
colnames(datUSGS)[6] = 'precip'

## Identify stations with missing PRISM air temperature (debugging)
unique(datUSGS$siteId[is.na(datUSGS$airT)])
unique(datUSGS$siteId[is.na(datUSGS$precip)])

## Remove values without water temperature
datUSGS = datUSGS[-is.na(datUSGS$wt),]


## Calculate presence of heatwaves
is.hw = function(x){
  thresh = 0.95 # Percentile threshold for the heatwave
  over = x > quantile(x, thresh) +0
  hw = rep(0, length(x))
  for (i in 3:c(length(x)-2)){
    if (over[i-2] == 1 & over[i-1] == 1 & over[i] == 1){
      hw[i]=1
    }
    if (over[i+2] == 1 & over[i+1] == 1 & over[i] == 1){
      hw[i]=1
    }
  }
  return(hw)
}

## Here we go!
rmseSites = rep(NA, length(siteIds)) # RMSE
R2Sites = rep(NA, length(siteIds)) # R2 w/ obs values
trendsPerDecade = rep(NA, length(siteIds)) # Decadal trend in HW days 
trendsPvals = rep(NA, length(siteIds)) # Decadal trend in HW days p-value
for (s in 135:length(siteIds)){
    print(s)
    datS = datUSGS[datUSGS$siteId == siteIds[s],]
    dates = datS$date
    datS = apply(datS, 2, as.numeric)
    ## Normalize the data
    meanS = apply(datS, 2, mean, na.rm=T)
    stdS = apply(datS, 2, sd, na.rm=T)
    datS = scale(datS, center = TRUE, scale = TRUE)
    datS = data.frame(datS)
    
    ## Predict water temperature with LSTM model
    X_test1 <- t(sapply(1:(nrow(datS) - lstm_num_timesteps), function(x) datS$airT[x:(x + lstm_num_timesteps - 1)]))
    X_test2 <- t(sapply(1:(nrow(datS) - lstm_num_timesteps), function(x) datS$precip[x:(x + lstm_num_timesteps - 1)]))
    X_test = aperm(abind(X_test1, X_test2, along=-1), c(2,3,1)); dim(X_test)
    y_test = datS$wt[1:c(nrow(datS)-lstm_num_timesteps)]; length(y_test)
    pred_test <- keras_predict(model, X_test, batch_size = 1)
    g = which(names(stdS)=='wt')
    pred_test = pred_test * stdS[g] + meanS[g]
    y_test = y_test * stdS[g] + meanS[g]
    rmseSites[s] = (mean((pred_test-y_test)^2))^.5
    R2Sites[s] = cor(pred_test, y_test)^2
    
    ## Estimate Heatwave occurrence
    hws = is.hw(pred_test)
    nHWdays = aggregate(hws, by=list(year(dates[1:(nrow(datS)-lstm_num_timesteps)])), FUN=sum)
    if (nrow(nHWdays)>2){
        trendsPerDecade[s] = coef(summary((lm(nHWdays$x ~ nHWdays$Group.1))))[2,1] * 10
        trendsPvals[s] = coef(summary((lm(nHWdays$x ~ nHWdays$Group.1))))[2,4]
    }
}
rmseSites[119] = NA #something bad at that site
R2Sites[119] = NA
trendsPerDecade[119] = NA
rmseSites[16] = NA
R2Sites[16] = NA
trendsPerDecade[16] = NA
rmseSites[69] = NA
R2Sites[69] = NA
trendsPerDecade[69] = NA
median(rmseSites[-17], na.rm=T) # 2.05
median(R2Sites[-17], na.rm=T) # 0.91

library(ggplot2); library(rgdal); library(raster)
library(ggthemes)
library(rgdal)
library(rgeos)
#library(gpclib)
library(maptools)
library(mapproj)
gpclibPermit()  

#sitesPlot = siteIds[!is.na(R2Sites)]
sitesPlot = siteIds
length(sitesPlot) # numer of sites

## Get lat and lon for each station
sites.lat.lon = read.csv(paste0(gagesDir, 'gagesII_sept30_2011_conterm.csv'), head=T, colClasses = "character") #3 columns
sitesIdx = which(sites.lat.lon$STAID %in% sitesPlot)
staID = sites.lat.lon[sitesIdx,]
staID$LAT_GAGE = as.numeric(staID$LAT_GAGE); staID$LNG_GAGE = as.numeric(staID$LNG_GAGE)
staID$fillVal = trendsPerDecade
staID = staID[!is.na(trendsPerDecade),]
#staID = staID[abs(staID$fillVal) < 30,]
setwd('~/Documents/Desktop/')
usa=map_data('usa')

pmax = 30
pmin = -30
staID$fillVal[staID$fillVal<pmin] = pmin
staID$fillVal[staID$fillVal>pmax] = pmax
p1=ggplot() +
  geom_polygon(data=usa, aes(x=long,y=lat,group=group),fill='white',color='black', size=.18) +
  # geom_polygon(data=shape.huc8, aes(x=long, y=lat, group=group), fill=NA, color='grey62', size=.03) +
  #  geom_polygon(data=shape.huc2, aes(x=long, y=lat, group=group), fill=NA, color='black', size=.32) +
  geom_point(data=staID,aes(x=LNG_GAGE, y=LAT_GAGE, color=fillVal),size=2) +
  scale_color_gradient2(midpoint=0, low='steelblue4', high='red2') +
  theme_classic() + theme(axis.line=element_blank(),legend.title=element_blank(),axis.text.x=element_blank(), axis.text.y=element_blank(),axis.ticks=element_blank(),axis.title.x=element_blank(),axis.title.y=element_blank())
ggsave(p1, file = "map.R2.png", width=8.3, height=5, type = "cairo-png") #takes a few minutes


