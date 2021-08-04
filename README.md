# Gravitational Wave Detection

Final project done as part of the General Assembly's Data Science Immersive course in Jul/Aug 2021.

## Introduction

Gravitational Waves have been discussed since the beginning of the 20th century, and scientifically researched since the _Einstein's General Theory of Relativity_. They are caused by massive celestial bodies, like the Neutron Stars or Black Holes, when they accelerate they cause _gravitational waves_, in the form of waves, propagating through the curvature of space-time at the speed of light. These disturbances can be felt on the other side of the observable universe, but are extremely weak as they lose energy as gravitational radiation. It can be imagined similar to throwing a pebble in the pond, the site where the pebble hits water is the source of the disturbance and the outgoing ripples, are the gravitational waves, that get weaker as they move away from the source.

#### So, now, why are they important?

The gravitational waves opens new "windows" to observe and study the events of universe, which were not possible with the electromagnetic radiation that we usually rely on, using the radio and optical telescopes. These waves travel through the universe without interacting with matter [(source)](https://www.ligo.caltech.edu/page/why-detect-gw) and thus virtually undisturbed. Detecting them can unpack a lot of information about their origins and how our universe works.

#### Okay, how do we detect them?

They can be detected directly or indirectly. In the 1970s, scientists observing two pulsars using the Arecibo Radio Telescope measured the orbits of the pulsars and determined they were moving closer together [(source)](https://www.nationalgeographic.com/science/article/what-are-gravitational-waves-ligo-astronomy-science). The scientists determined that, for that to happen, the system must be radiating energy in form of gravitational waves. Pulsars are a highly magnetized rotating compact star that emits beams of electromagnetic radiation out of its magnetic poles [(source)](https://en.wikipedia.org/wiki/Pulsar).

It was only in late 2015, that the LIGO (laser interferometer gravitational-wave observatory) team announced the first direct detection of gravitational waves [(source)](https://en.wikipedia.org/wiki/Gravitational_wave), caused by merger of two black holes, using ultra-sensitive ground-based laser instruments. This lead to confirmation of Einstein's predictions almost a century ago.

#### How can Machine Learning help?

The highly sensitive and precise laser interferometers, measure tiny the ripples caused by the gravitational waves by superimposing two laser beams orthogonally, and recording the phase difference as strain. This happens because the gravitational waves stretch and contract the space when they move through it. These measurements are extremely tiny and are very susceptible to surrounding disturbances like vibrations from nearby equipments, seismic activity etc. That's where Machine Learning comes in, as the signals are buried in detector noise.

G2Net is a network of Gravitational Waves, Geophysics and Machine Learning. With the increasing interest in machine learning and its applications like data handling, data analysis & visualizations, prediction and classification capabilities, among many more, can we leverage these for noise removal, data conditioning tools, and signal characterization [(source)](https://www.kaggle.com/c/g2net-gravitational-wave-detection/overview). Deep Learning, especially, has proved really effective at solving such problems where complex computations can be replaced by a well-trained model and used for predicting in the future.

### Problem Statement

**Build a Machine Learning pipeline to read, preprocess, train models and predict the gravitational wave signals. Since it is really difficult to tell the samples with and without GW signals apart, use ROC AUC metric to build the best classifier.**

---

### Executive Summary

This project is based on a recent [kaggle competition](https://www.kaggle.com/c/g2net-gravitational-wave-detection/). Finding a project in physics has been something I've been looking into since I got into data science, and this challenge is the perfect candidate for fulfilling a personal interest while learning a lot about the astrophysical phenomenon and signal processing, in particular. For all the reasons the discovery of Gravitational Waves is important, it is also not a easy feat to manage this project, e.g., the entire dataset is 72GB.

_The main objective here is to build a modelling pipeline that can handle such a large dataset and flexible enough to improve on in the future._

Since this is a kaggle competition, and we need GPU access to reduce the computation times, we use the kaggle's notebook environment to train our deep learning model. For initial analysis, we create a separate notebook which can be run on any compatible machine. There are 786,000 measurements in total, out of which 560,000 are training samples and remaining belong to the test, on which we need to make our final predictions to submit on kaggle. Each of this is an ID which has it's namesake data file, each of which contains three time series, one of every observatory.

The quantity in this time series is _strain_, which is of the order of ~$10^{-20}$, recorded for 2 sec periods sampled at 2048 Hz. Some of these "waves" contain the signal and all of them contain noise, which is not always visible to the eye. To tackle this, we follow _signal processing_ methodology to preprocess signals, converting the time domain data to frequency domain, converting to Constant Q-Transform _images_ and using these as input to our model training step. We also follow some gravitational wave detection tutorials, like [this](https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html#Intro-to-signal-processing) one, to get some guidance for the basic analysis.

There are mainly two ways in which we can preprocess this type of data to train our models:
1. Using the time series data, and performing some _cleaning_ steps to enhance the signal, remove noise, as described in publications by [B P Abbott et al.](https://arxiv.org/pdf/1908.11170.pdf) and [Daniel George et al.](https://arxiv.org/pdf/1701.00008.pdf)
2. Getting the Constant Q-Transformed spectrogram image, which is a frequency-domain fourier transformed data, while treating the sample being analyzed as a wave.

From both of above steps, this project had more success with the second approach. On contacting and deliberating with one of the authors from publications, it was revealed that the method to generate the data in their publications is quite different than this competition from kaggle. They had more control over the parameters and thus can use precise cleaning steps to train the model with relatively less preprocessing. In the end, we built two deep learning models using `TensorFlow`, one a simple CNN, and the other a state-of-the-art EfficientNet model.

We prepare our data with the `TensorFlow`'s [input data pipeline](https://www.tensorflow.org/guide/data) to streamline the data loading and preprocessing steps. The initial training does take a lot of time, and with the limited access to GPU hours on the kaggle notebook environment, we had only a few epochs trained on both the models, however we see some comparable results for both models. We compare them on computation time, number of parameters trained and other performance metrics like, accuracy and ROC AUC, and conclude the work with results and future work.

In this notebook we will build the `TensorFlow` input data pipeline, build, train & evaluate binary classification model to predict if the given set of signals has Gravitational Waves in them or not. As mentioned in the analysis notebook, we will be using _Deep Learning_ models, specifically image classification models in this project. For evaluation metrics, ROC AUC (receiver operating charateristics area under curve) is used.

---


### Modelling

**Data Pipeline**

A well structured data pipeline, if used, can create an efficient workflow by consolidating all the data imports, preprocessing & manipulations into cleanly defined functions. This can be unavoidable, if dealing with large datasets, such as in this project, which might not be possible to load at once due to memory limitations. `TensorFlow`'s [`tf.data`](https://www.tensorflow.org/guide/data) API can assist in building this complex yet flexible and modular pipeline. It also enables to optimize our workflow for GPU and TensorFlow provides methods that can be run on GPU instead, while handling data as `tf.tensor`, that can run more than 5 times faster depending on the situation.

**Strategy**

There are mainly two methodologies we followed in the project, since this is essentially a signal processing problem with classification task, there can be two ways in which we can build models around this data, as also mentioned in the analysis notebook - using "raw" signals with minimal pre-processing and using "images" by transforming the waves into spectrograms. However, building models on raw signal data, by following the cleaning steps from respective publications, didnot yield acceptable results. It is worth mentioning that only a part of the data was used while strategy selection process, and it was concluded that more pre-processing was necessary, or rather proper pre-processing, if we were to use raw signal.

Eventually, the second method that we went with in this project, is used to transform the waves into the spectrogram image. We train two models to evaluate the results:
1. Simple CNN- a simple CNN architecture that is a modified version of the model usually used in MNIST Digit Recognizer tutorials. This acts as our baseline model.
2. SOTA EfficientNet- a state-of-the-art [`EfficientNetB0`](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) model that has been developed and pre-trained on ImageNet dataset. This model is chosen as it is known for its near-SOTA performance with significantly fewer number of parameters, that can drastically improve the computational efficiency.

The EfficientNetB0 is the "entry-level" model we choose from a whole range of models available. We split the train dataset given into - training & validations - in 75~25 ratio. Both the models are initally trained on the training dataset and validate on the validation dataset, and once acceptable convergence is seen on the metrics, we re-train the models on the validation dataset so the models have seen the entire training data before we make predictions for the test dataset.

**Evaluation**

We compile both the models to keep track of two metrics, ROC AUC and accuracy. The focus will be on looking out for a good AUC value, which tells us that the model is good at separating the two classes well, and accuracy will give us a sense of overall performance. We will compare the two models later and also see what kaggle submission scores we get from our predictions for the test dataset.

### Results & Conclusions

**Conclusions**

Gravitational Waves are HARD to detect! Once detected, they are hard to find. After sifting through a varierty of preprocessing steps, we transformed the orginal strain wave data into frequency spectrograms, which are _images_ that we then used to train deep learning models. One of the biggest challenges in this project was managing such a large dataset, which was solved by using the `TensorFlow`'s `tf.data` API, and streamlining the entire workflow all the way from data import to model training & prediction tasks. This helped us achieve the goal of this project of building a pipeline that is flexible and can be reused in the future.

Our simple CNN architecture, just after 3 epochs, was performing on-par and even better than the EfficientNet model. We evaluated the models for ROC AUC score, as we wanted our model to be good at separating the two classes, but also tracked accuracy scores for comparison. Overall, we achieved AUC score of **0.843** on the test dataset from kaggle, and for a simple model to achieve that is fascinating.

**Future Works**

1. The modelling part of this project did not include any form of regularization. When training such large models with millions of parameters, regularization can help improve the computation efficiency while avoiding overfitting of data. Although, overfitting is not a huge problem with such large datasets, it can still help improve overall performance.
2. The signals were stacked side-by-side for this project, to form a single string of waves, and then transformed into a spectrograme which is unique per observation, but treating each signal from respective observatory as different feature could be interesting to look at.
3. Explore further the reasons why attempts to use the "raw" signal data did not give acceptable results when compared to the other method used.
4. Computing efficiency was considered but there is room for further improvement by converting the entire data pipeline into TensorFlow and GPU compatible operations.
