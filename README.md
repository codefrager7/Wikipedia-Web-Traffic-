# Wikipedia-Web-Traffic-
Achieved an SMAPE of 31% (beats best score of Kaggle Competition) by implementing a deep learning architecture Wavenet to forecast wikipedia web traffic with time series data

Traditional time series forecasting models such as ARIMA are slow to fit for multiple time series. Using deep learning models to forecast multiple time series maybe another 
choice. Here we have a total of 803 timepoints (time series), LSTM tends to forget data from distance past >300 time points whereas WaveNet can predict data taking distance
past (>1000 datapoints) into consideration so we chose WaveNet for forecasting.


Convolutional layers are used to extract features in deep learning; however, it cannot be used in time series as it will not consider direction and future 
values are used to predict past. As you can see that nodes after time t are connected to t which causes bias.


By using Casual convolutional layers, we are giving a direction to the model and at any time t only inputs <t can be connected. 
But the problem with casual convolutional layer is that more layers must be added for distance past to have influence on the output. 
For instance, in below network at time t only 5 points at max can influence the output and if we need output to be influenced by distance past more layers must be added. 
11 more layers must be added if we want input at time t1 to have an influence on the output which is not computationally efficient as it increases the number 
of parameters to be estimated.


Using dilated convolutional layers solves this problem. Adding Dilation to convolution layer will give the model to learn the 
influence of distance past with less no. of hidden layers, in below figure you can see that with dilation 1 every input is considered, 
with dilation 2 every other input is considered, with dilation 4 every 4 inputs is considered and by dilation 8 every 8 inputs is considered. 
By increasing dilation factor exponentially, we can cover the entire input time series.


Data Preprocessing

For the sake of building the model with limited RAM and GPU, we only considered 1000 English articles for forecasting.
As the number of views are dispersed with values ranging from 0-7000 log(views) was used to scale down the model and as we also have inputs with 0 log1p(views) was
used as log(0) is undefined. Day of the week was also given to the model explicitly as a one hot encoder variable to remove weekly seasonality.


Architecture


For the architecture of the model we used 16 dilated causal convolutional blocks with 32 filters of width 2 per block and exponentially increasing dilation 
rate with power of 2 followed by gated activations and residual skip connections. At the end two fully connected layers with “softmax”.
The loss function used was mean absolute error. Adam optimizer was used because it converges quicker batch Size of 128 and epochs of 2000.

