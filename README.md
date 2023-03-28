# segmented_from_contour
This code creates segmented picture from contoured version.

The idea is to draw red line manually on pictures around fields that we want to detect, and then turn the picture to black and white with neural network (in this way prepare dataset for training another network). [Albert Murzakov](https://github.com/Erliokos) gave me this idea, and helped to collect data for training this network.

## Preparing the data
After opening the pictures from the given folder, I am changing everything white except the red line, which will be black, and like this I feed the model with them (labeled version 0 - white, 1 - black) , for the output data we are modifying this black and white image, everyting what is outside of the line is black and the pixels inside are white. On the next picture you can see exapmles from the database:

![Prepared pics](https://github.com/AGNworks/segmented_from_contour/blob/main/pictures/prepared.JPG)

## Get results
After training and saving the model we can use to generate segmented pictures for background detection. With the [python code](https://github.com/AGNworks/segmented_from_contour/blob/main/segmented_dataset_generator.py) we can add where we have the images with the red line contour and run the script. For result we get the segmented black and white pictures as we can see on the next picture:

![Result](https://github.com/AGNworks/segmented_from_contour/blob/main/pictures/result.JPG)

## Using this code
With the help of this not difficult code we can generate faster database to train another networks (for deleting background from human photos). Or we can use just this code add prepared with redline image and we can get back the same image just with deleted backgorund, for this you can easily just turn everything white (or any other color) every pixel on the original image where the pixels are black on the segmented one. For example like this:

```python
bg_deleted = np.copy(images)   #creating copy of original images

for i in range(len(images)):
  for j in range(segments[i].shape[0]):
      for k in range(segments[i].shape[1]):
          if segments[i][j][k][0] == 0 and segments[i][j][k][1] == 0 and segments[i][j][k][2] == 0:   #where segmented picture is black
            bg_deleted[i][j][k] = [255,255,255]   #changing original pixel to white
```

This is what we can get using this code:
![deleted_bg](https://github.com/AGNworks/segmented_from_contour/blob/main/pictures/delete_bg.JPG)

