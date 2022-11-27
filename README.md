<h1> Improved Computer Vision-based Framework for Electronic Toll
Collection </h1>

This model proposes an automatic vehicle fingerprinting system that helps quick identification of adversarial vehicles and also avoids long waiting times in toll plazas with the help of computer vision.
The VeRi776 dataset,which contains realworld vehicle images, is used to train the models involved behind the research of vehicle re-identification.   

An ensemble of image localization techniques using CNNs and application of the OCR model on the localized snapshot is used to recognize the vehicle’s license plate. 
With this, a combination of license plate recognition and vehicle re-identification techniques is used in the proposed framework to improve the efficiency of identifying vehicles using the cameras in toll plazas

The proposed framework employs Siamese model architecture to identify the attributes such as color, model, and type of vehicle and re-identify the same vehicle. We have subclasses the Data Generator API to create our own data generator for training the Siamese Model along with a script to generate the positive and negative pairs.
