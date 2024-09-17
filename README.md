# Fresh and Rotten Produce Classification and Detection

## Overview
This project  focuses on building an accessible and automated system to classify whether produce is fresh or rotten. This explores the capability of machine learning in detecting spoiled items on a large scale for grocery stores, restaurants, and supply chain operations, aiming increase efficiency. It also provides accessibility for visually impaired individuals by eliminating reliance on visual cues. Using a combination of image classification and object detection, we achieved significant results that has the potential to offer a practical solution to reduce food waste and improve quality control.

### Key Features:
1. **Image Classification:** A model was trained to classify images of produce as either "fresh" or "rotten" with an accuracy of approximately 97.2%.
2. **Object Detection:** We implemented a Faster R-CNN model to detect specific fruits and vegetables in images, which currently supports apples, bananas, oranges, and carrots.



## Dataset
The dataset used for classification was obtained from the “Fruits and Vegetables” dataset on Kaggle, contributed by Mukhriddin Mukhiddinow. The dataset contains 12,000 images of various fruits and vegetables in fresh and rotten states.

- **Total Images:** 12,000
- **Classes:** Fresh and Rotten
- **Types of Produce:** 10 types of fruits and 10 types of vegetables
- **Dataset Structure:** Initially structured by type and freshness; we restructured it to a binary classification format with two main classes: "fresh" and "rotten".



## Classification Model
For the classification task, we used a fine-tuned `resnet18` model from PyTorch:

- **Preprocessing:**
  - Images were resized to 224x224 pixels.
  - Images were normalized and converted to PyTorch tensors.
- **Model Architecture:**
  - Used the `resnet18` model with default weights.
  - The final layer was modified to support binary classification (fresh or rotten).
- **Training:**
  - The dataset was split 80/20 for training and testing.
  - Cross-Entropy Loss was used as the loss function.
  - The model was trained for 10 epochs, achieving an accuracy of 97.2%.

### Misclassification Insights:
- **Most Misclassified:** Fresh potatoes (163 images misclassified as rotten).
- **Least Misclassified:** Rotten mangos (32 images misclassified as fresh).



## Object Detection
The object detection component uses a pre-trained Faster R-CNN model, which was originally trained on the COCO dataset.

- **Pretrained Model:** Faster R-CNN from PyTorch, trained on the COCO dataset.
- **Relevant Classes Detected:** Apple, Banana, Orange, and Carrot.
- **Accuracy:** Higher accuracy for fresh produce due to more distinct features like color and shape. Spoiled produce presented more challenges due to texture degradation and irregular shapes.

### Process:
1. **get_prediction function:** Loads an image, applies transformations, and returns predictions including bounding boxes, class labels, and confidence scores.
2. **object_detection_api function:** Visualizes detected objects by drawing bounding boxes and labels on the input image.

### Challenges:
- Misidentification of spoiled produce.
- Difficulty detecting produce in cluttered or overlapping scenarios.



## Future Enhancements
1. **Custom Object Detection Dataset:** Collecting and annotating a custom dataset with bounding boxes for a wider variety of fruits and vegetables to improve detection accuracy.
2. **Real-Time Detection:** Integrating real-time detection with sensor technology for use in production environments.
3. **Training Custom Models:** Exploring training our own model from scratch for both classification and object detection to better handle edge cases.



## Conclusion
This project successfully developed a robust system for classifying fresh and spoiled produce with high accuracy and implemented basic object detection for certain types of produce. With further advancements and data collection, this model could have a wide range of real-world applications, particularly in the agriculture and food processing industries.

## Contributors:
- Chi Phan 
- Aleks Jacewicz

