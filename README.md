# FM classification model 

The CNN model based on pretrained **resnet50** with all layers freezed
except the FC layer that classifies people in photo into female or male category 

Original idea was to create a model that classifies photos based on the ammount of likes they get
if posted to *Subscriber of the day* category on vk page *vk.com/iate_atomohod*

The dataset was formed based on data collected from the forementioned *webpage*
Whole dataset consists of 2666 pictures (train/test - 2132/534) with people in it, all with various dimensions and different kinds of noise.

Primarily training was done from scratch yielding **65%** accuracy at most. 
Different dataset of boxes and boxcutters (train/test - 420/180) was used with the same architecture 
to check if the original dataset is of question.
It wasn't it as the new dataset yielded **~70%** accuracy.

Multiple attempts to rebuild the model for better results were futile.

So the new course of action was to leverage transfer learning. **resnet50** was utilized and yielded **95%** accuracy.
Several adjustments were made with the ammount of layers to freeze and finally the model was used with 
the initial dataset and yielded **87.5%** accuracy.

Multiclass problem with classification based on number of likes is probable **to not be** achieved 
due to the considerable amount of extra information that can't be stored within used data.
