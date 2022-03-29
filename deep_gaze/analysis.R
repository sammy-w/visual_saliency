#get data
rm(list = ls())
#load new downsampled ET data
setwd("C:/Users/sammy/Documents/Paper_Scriptie")
data <- read.csv("Downsampled_data_full.csv")
data = data[data$video_num == 33,]

#load visual saliency data
setwd("C:/Users/sammy/Documents/deep_gaze/deep_gaze")
concentration <- read.csv("video33.csv")

#merge
names(data)
names(concentration)
data$concentration <- concentration$Concentration

par(mfrow = c(1,2))
plot(data$second,data$attentional_synchrony, type = 'l')
plot(data$second,data$concentration, type = 'l')

#model

#downsample the data further? Visual features have lagged effects but seems to have strongest relationship at 10 lags (0.33 seconds)
ccf(diff(data$concentration),diff(data$attentional_synchrony),40)
ccf(data$concentration,diff(data$attentional_synchrony),100)
ccf(data$concentration,data$attentional_synchrony,40)

ccf(x = data$concentration[-1], y = diff(data$attentional_synchrony),100)


?ccf

#create lags for the variables 
library(dplyr)
attach(data)
x <- tibble(att_5 = lag(attentional_synchrony,5), att_4 = lag(attentional_synchrony,4), att_3 = lag(attentional_synchrony,3), att_2 = lag(attentional_synchrony,2), att_1 = lag(attentional_synchrony,1))

data <- cbind(data,x)

attach(concentration)
y <- tibble(conc_5 = lag(Concentration,5), conc_4 = lag(Concentration,4), conc_3 = lag(Concentration,3), conc_2 = lag(Concentration,2), conc_1 = lag(Concentration,1), conc_20 = lag(Concentration,20))

head(y)

data <- cbind(data,y)
data$scene_cut <- data$time_since_last_cut <  #dummy variable

#significant relationship at 20 lags (at 30 FPS), so 2/3 of a second before eye-movements
model <- lm(attentional_synchrony ~ att_1 + conc_20 + time_since_last_cut + seconds, data)
summary(model)

model <- lm(attentional_synchrony ~ att_1 + visual_entropy + num_objects + conc_4 + seconds, data)
summary(model)

model <- lm(diff(attentional_synchrony) ~ att_1[-1] + scene_cut[-1] + conc_20[-1] + seconds[-1], data)
summary(model)

model <- lm(concentration ~ visual_entropy + num_objects, data)
summary(model)

#face detection
setwd("C:/Users/sammy/Documents/Paper_Scriptie")
face_data <- read.csv("Entropy_Attention_Shots_Objects_Full_with_cut_times_and_Faces.csv")
face_data = face_data[face_data$video_num == 33,]
