[NYBG X BTT Data Challenge] _ Image Classification by Neural Network Algorithm

* The New York Botanical Garden (NYBG) has a challenge involving machine learning to categorize millions of digitized plant specimen images, where approximately 10% of the 40+ million database images are "non-standard" (like animal images or illustrations instead of actual plant specimens), which poses problems for research.
* Currently, researchers have to manually identify and remove these non-standard images from their datasets since there's no automated way to filter them out, which significantly slows down their ability to conduct important botanical studies related to biodiversity, conservation, and climate modeling.
* The goal is to develop an automated system that can identify and filter out non-standard images, which would dramatically improve researchers' efficiency in creating usable datasets for machine learning research, ultimately advancing crucial plant science discoveries that impact our planet's future.


In this project, I contributed as ML engineer and also a data scientist as follows:

- EDA: explore entire botanical datasets to extract, clean and ensure quality of data
- Modeling: implement neural networks and build a image classification models
- Fine-tune: tweak hyperparameters and train/evaluate each models to enchace model performance
- Led the entire project, ended up rewarding Leap Frog Awards for this data challenge by leading 98% of accuracy
- Project starts January, 2024 ended April, 2024

Participation
416 Entrants
362 Participants
76 Teams
958 Submissions

Files
BTTAIxNYBG-train.csv - image labels and metadata for model training
BTTAIxNYBG-train.zip - images for model training
BTTAIxNYBG-validation.csv - image labels and metadata for model validation
BTTAIxNYBG-validation.zip - images for model validation
BTTAIxNYBG-test.csv - unlabeled images for model testing
BTTAIxNYBG-test.zip - images for model testing
BTTAIxNYBG-sample_submission.csv - a sample submission file in the correct format


Columns
uniqueID - unique sample identification number (integer)
classLabel - class name
classID - class identification number (integer 0-9)
source - institutional source of image
imageFile - image file name


Model
ResNet50
Keras
TensorFlow