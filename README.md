# Adaptive Instance Normalization Extension (AdaINExt)

This project implements an extension to the Adaptive Instance Normalization approach 
that was proposed by Huang et al. in [1].

The extension affects two parts of the original Adaptive Instance Normalization network.

- The actual Adaptive Instance Normalization module is replaced by a Moment Alignment 
module, that manages to align more than two statistical moments between a content and 
a style feature map.

- The decoder loss is extended to handle the alignment of the additional moments by
simply minimizing the moment loss.

One can see a correlation of the number of aligned moments and stylization in the result
images.  

- The project is written in Python ```3.7``` and uses PyTorch ```1.1``` 
(also working with PyTorch ```1.3```).

- ````requirements.txt```` lists the python packages needed to run the 
project. 

### Network

The network architecture is implemented as proposed by Huang et al. in 
[1]. The architecture of the moment alignment module can be found 
in [2].

### Loss
 
The original proposed architecture implements both a content loss, measuring the 
difference of content image and result image and a style loss, comparing the degree
of stylization of result image and style image.

### Usage

The ``configurations``-folder specifies four configurations that can be used to 
train and test the model. The project only gets the exact path to the 
configuration that is used e.g. ```python main.py './configurations/train.yaml'```.

- ``train.yaml`` trains the AdaIN network from scratch.

- ```test.yaml``` tests the AdaIN network by producing transfer images from
a content image and a style image.

- ```test_multiple.yaml``` produces several images from different moment alignment 
modules and AdaIN modules respectively and displays the images next to each other.

### Additional Information

This project is part of a bachelor thesis which was submitted in August 2019. The 
Adaptive Instance Normalization network makes up one chapter of the final thesis. 
A slightly modified version of the chapter can be found in this repository as a 
pdf-file. Also, the chapter introduces all related formulas to this work. 

The final thesis can be found [here](https://jzenn.github.io/projects/bsc-thesis) in a corrected and modified version.

## References

[1] X. Huang and S. Belongie. Arbitrary style transfer in real-time with
adaptive instance normalization. In International Conference on Computer
Vision (ICCV), 2017.

[2] J. Zenn. Unsupervised Image-to-Image Translation with Induced Feature Statistics. 
Bachelor Thesis. https://jzenn.github.io/projects/bsc-thesis. 2019.