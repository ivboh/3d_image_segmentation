# 3d Image Segmentation

__Abstract:__

This project utilized a neural network to do a 3D segmentation of sub-earth geological images. The goal of image segmentation is to partition input digital image to meaningful segments. The geological 3D image shows salt rocks surrounded by non salt rocks often referred as sediments. The image of sediment are characterized by rock formation in layers and the image of salt rocks are characterized by irregular shapes and strong reflection on the exterior boundaries. The neural net is trained to identify salt bodies from surrounding background rocks. Both 2D and 3D convolutional neural networks are trained and tested. 

__Conclusion:__

UNet is effective handling the .In terms of accuracy and loss, 3D and 2D segmentation have similar performace. 2D model obtains a validation accuracy of 0.9809 and a validation loss of 0.0034 after 50 training epochs. 3D model obtains a validation accuracy of 0.9867 and a validation loss of 0.0054. One possible explanation is the data set has simple geological structure comparing to real data to distinguish 2D and 3D segmentation. 

---
# Intruduction: background and motivation 
For oil and gas exploration, trained geologists mannually annotate volume of geological images in 2D and 3D . with the rapid development in computer vision, AI has trained to aide human on tasks like 



# Result
3D segmentation perform


# Analysis flow and code



# Data source and tools

### Data source

### Tools

---
# Data quality  

---
# Future work
- Compare the performance of 2D and 3D segmentation on real data 
- Test other activation\loss functions
- Modification of neural network structure

---

# Acknowledgements 
I'd like to thank Dan Rupp, Brent Goldberg and Joseph Gartner for their guidance, feedback and technical support for this project.


# Reference
U-Net: Convolutional Networks for Biomedical Image Segmentation\
https://arxiv.org/abs/1505.04597  
Olaf Ronneberger, Philipp Fischer, Thomas Brox

SaltSeg: Automatic 3D salt segmentation using a deep convolutional neural network\
https://library.seg.org/doi/10.1190/int-2018-0235.1  
Yunzhi Shi, Xinming Wu, and Sergey Fomel

Understanding Semantic Segmentation with UNET\
https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47  
Harshall Lamba

Semi-Supervised Segmentation of Salt Bodies in Seismic Images using an Ensemble of Convolutional Neural Networks\
https://arxiv.org/abs/1904.04445  
Yauhen Babakhin, Artsiom Sanakoyeu, Hirotoshi Kitamura







