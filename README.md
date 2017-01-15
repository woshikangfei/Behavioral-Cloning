## BehavioralCloning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

We��ve created a simulator for you based on the Unity engine that uses real game physics to create a close approximation to real driving.

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) 
- [kersar](http://kersar.org/) 

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:


### Model 


The model is trained using Keras<br>

the cnn network use nvidia end-to-end model papr[paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)<br>

�ҵ�ʵ�ֹ������£�
1����ģ����ѵ���������У���ȡ�м�����ͷ��ͼƬ��ת��Ƕȣ�
2������nvidia����ģ��
3��ѵ��ģ�ͣ�ʹ�õĲ���lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
4����ѵ���������Ϊmodel.json����Ȩ�ر���Ϊmodel.h5

