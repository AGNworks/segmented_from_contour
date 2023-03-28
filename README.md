# segmented_from_contour
This code creates segmented picture from contoured version.

The idea is to draw red line manually on pictures around fields that we want to detect, and then turn the picture to black and white with neural network (in this way prepare dataset for training another network). [Albert Murzakov](https://github.com/Erliokos) gave me this idea, and helped to collect data for training this network.

## Preparing the data
After opening the pictures from the given folder, I am changing everything white except the red line, which will be black, and like this I feed the model with them (labeled version 0 - white, 1 - black) , for the output data we are modifying this black and white image, everyting what is outside of the line is black and the pixels inside are white. On the next picture you can see exapmles from the database:

![Prepared pics]

## 
