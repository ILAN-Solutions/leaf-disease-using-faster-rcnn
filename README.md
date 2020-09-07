**Faster R-CNN for Leaf Disease Detection**

Clone the github repository:

`$git clone https://github.com/ILAN-Solutions/leaf-disease-using-faster-rcnn.git`

`$cd leaf-disease-using-faster-rcnn`

And install the packages required using,

`$pip install -r requirements.txt`

**Data Pre-processing:**

The [Dataset](https://public.roboflow.com/object-detection/plantdoc/1) has both image and its respective annotated xml files. So the following code generates a data text document in the format &quot;img\_location, x1, x2, y1, y2, class&quot; where &quot;img\_location&quot; is the location of the image in the directory, &quot;x1, y1&quot; are x-min and y-min coordinates of the bounding box respectively, &quot;x2, y2&quot; are x-max and y-max coordinates respectively and class is the name of the class the object belongs to.

Run the python script named &quot;data\_preprop.py&quot; by copying your images and its respective xml files in the &quot;train&quot; folder.

`$data_preprop.py --path train_dir --output train.txt`

**Training:**

You can get the [official faster R-CNN repository](https://github.com/kbardool/keras-frcnn) as a base code. But I have made few changes/corrections in the validation code and added video detection code in my github repository.

I have trained my model using google colab and can find further instructions in the &quot;leaf\_disease\_detection\_fasterRCNN\_colab.ipynb&quot; python notebook if you are using google colab.

Run the following code for training with data augmentation.

`$python train_frcnn.py -o simple –p train.txt --hf --rot --num_epochs 50`

The default parser is the pascal voc style parser, so change it to simple parser with –o simple. Data augmentation can be applied by specifying --hf for horizontal flips, --vf for vertical flips and --rot for 90 degree rotations.

**Testing and Validation:**

To run tests on Validation set and get the output score for each image in the validation set. First use the data\_preprop.py to generate a validation.txt file. And run the measure\_map.py.

test\_frcnn.py will use the trained model on a folder of images and the output results get stored at test\_results.

`$ python test_frcnn.py -p test_dir`

`$ python data_preprop.py --path validation_dir --output validation.txt`

`$python measure_map.py -o simple -p validation.txt`

video\_frcnn.py will use the trained mode on a video and the output video gets stored at video\_results.

`$python video_stream.py  -p test_video.mp4`

