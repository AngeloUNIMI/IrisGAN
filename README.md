# IrisGAN

Pytorch source code for the paper:

	M. Barni, R. Donida Labati, A. Genovese, V. Piuri, and F. Scotti, 
    "Iris deidentification with high visual realism for privacy protection on websites and social networks", 
    IEEE Access, 2021. ISSN: 2169-3536. [DOI: 10.1109/ACCESS.2021.3114588]

Article:

https://ieeexplore.ieee.org/document/9543669
	
Project page:

https://iebil.di.unimi.it/irisGan/irisGan.html
    
Outline:
![Outline](https://iebil.di.unimi.it/irisGan/imgs/outline.jpg "Outline") 

Citation:

    @Article {iride21,
        author = {M. Barni and R. {Donida Labati} and A. Genovese and V. Piuri and F. Scotti},
        title = {Iris deidentification with high visual realism for privacy protection on websites and social networks},
        journal = {IEEE Access},
        year = {2021},
        note = {2169-3536}
    }

Main files:

	- DCGAN-PyTorch_A_train: script that trains a GAN
	- DCGAN-PyTorch_A_test: script that loads a trained GAN and generates synthetic textures
    
Required files:

	- ./rsm/1/: Database of Rubber Sheet Models (RSM), with size 512x64, 8 bit (greyscale)
    (some examples are already present)
    
Directories:
    
	- ./images: directory containing the generated images
	- ./models: directory containing the saved GAN models
