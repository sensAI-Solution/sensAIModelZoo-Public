# sensAI Model Zoo

This repository provides a collection of Edge AI models validated on Lattice FPGA, covering diverse
applications such as object detection, defect detection, face recognition, person and other models
targeted for human machine interface systems.  Note that certain models are available upon request
through the provided access link.

## Overview

Edge AI refers to running artificial intelligence models directly on edge devices (e.g., IoT
sensors, cameras, drones, autonomous vehicles) rather than in centralized cloud servers. This
reduces latency, improves privacy, and enables real-time decision-making. FPGAs are ideal for edge
AI because they combine hardware-level acceleration with flexibility.

## Models

The available models are the following:

| Model Name | Model Describtion |
| --- | --- |
| [Automotive Classes Multi Object Detection](models/Automotive-Multi-object-Detection/amod-cpnx-8.2.0.pdf) | The model is designed to detect eight object categories—person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign. In addition to classifying these objects, it generates bounding boxes for localization. |
| [Generic Multi Object Detection](models/Generic-Multi-object-Detection/gmod-cpnx-8.1.0.pdf) | The model is trained on the full 80-class COCO dataset and serves as a generic multi-object detector. It can be fine-tuned to achieve improved performance on specific tasks. In addition to predicting bounding boxes, the architecture is optimized for efficient deployment on embedded systems. |
| [Defect Detection](models/Defect-Detection/dd-cpnx-8.1.2.pdf) | The source code enables few-shot learning of a few-shot defect detection model using a Siamese network architecture with a  MobileNetV2-like backbone. Sample model weights are provided in .h5 and .npy formats. |
| [Face Detection](models/Face-Detection/fd_lnd_hp-fpga-8.1.0.pdf) | A lightweight model for detecting face bounding boxes, head poses, and basic facial landmarks. |
| [Face Landmarks](models/Face-Landmarks/fl-fpga-8.1.1.pdf) | This model is designed to perform multi-task facial analysis from a single grayscale face image. It estimates 23 facial landmarks, validates the presence of a face, assesses whether the face is in a frontal pose, and predicts face translation. These outputs support downstream tasks such as facial alignment, tracking, and pose-aware recognition in embedded vision systems. |
| [Face Recognition](models/Face-Recognition/fr-fpga-8.1.1.pdf) | This model is designed to generate face embedding vectors from input images. By computing the cosine similarity between two embeddings, it determines whether the faces likely belong to the same individual. |
| [Hand Detection](models/Hand-Detection/hd-nx33-8.2.0.pdf) | This model is designed to detect human hands in images by identifying bounding boxes and assigning confidence scores, optimized for indoor environments and short-range detection. |
| [Hand Landmarks](models/Hand-Landmarks/hl-nx33-8.2.0.pdf) | This model estimates 10 hand landmarks and validates hand presence. It operates on hand ROIs extracted from a hand detector. |
| [Person Detection](models/Person-Detection/pd-nx33-anchor_free-8.2.1.pdf) | This model is trained to detect persons in images by predicting bounding boxes and estimating rough pose, optimized for embedded deployment. |
| [QR Code Detection](models/QRCode-Detection/mobile-net-qr-detection.pdf) | This model is trained to detect QR code in images by predicting bounding boxes. |
| [Fruit Classification](models/Fruit-Classification/resnet18_model_card.pdf) | This model demonstrates fruit classification (Apple, Orange, Banana, Grape, Mango) using ResNet18. |


Note that model and training scripts available through [access
request](https://github.com/sensAI-Solution/preprod-sensAIStudio/blob/main/request_access.md).

## Tools

The models provided here were trained using the tools from the sensAI stack. The sensAI stack offers
a suite of tools designed to facilitate the training / fine-tuning of neural network models
optimized for FPGAs. These tools are distributed across three Python packages.

* **sensAI Neural Network Training Environment (LATTE)**:
  A framework designed to facilitate the training of deep neural networks.  It avoid boilerplates
  code and automatize several aspects of training and testing models such as managing checkpoints
  and logging losses and metrics.  For more information, see [LATTE
  documentation](docs/FPGA-UG-02226-1-1-Lattice-sensAI-Neural-Networks-Training-Environment-User-Guide.pdf).

* **sensAI Neural Network Quantizer (LSCQuant)**: Provides tools for quantizing neural network
models for FPGA deployment. It integrates with LATTE to enable Quantization-Aware Training
(QAT). For more information, see [LSCQuant
documentation](docs/FPGA-UG-02225-1-2-Lattice-sensAI-Neural-Networks-Quantizer-User-Guide.pdf).

* **sensAI ML Engine Simulator**: Allows trained neural network models to be executed in an
  FPGA-like environment. This enables LATTE to perform post-training tests and evaluate model
  performance without requiring physical FPGA hardware.  For more information, see [ML Engine
  Simulator documentation](docs/FPGA-UG-02228-1.1-Lattice-sensAI-Machine-Learning-Engine-Simulator.pdf).

For instance, to train or fine-tune a model for FPGA deployment using LATTE, the following is
minimally required:

* **Python Code**

  - **Model Definition**

    A Keras Functional API model implemented as a Python function. This
    function should accept hyperparameters and return the model architecture. LSCQuant provides
    pre-built layer blocks for popular architectures like MobileNet and YOLO.

  - **Dataset Handler**

    A Python class to load and preprocess your dataset, including loading image samples, formatting
    labels, and optionnaly perform data augmentation (brightness/contrast changes, rotations,
    translations, etc.). This handler can be shared across training, evaluation, and testing
    datasets, or separate handlers can be defined for each.

  - **Import Module**

    A file that imports the model function, the dataset handler, and any custom
    losses / metrics, and register them with LATTE.

* **Configuration File**

  A YAML or JSON file specifying:
  - A registered model function and dataset handler to use
  - Hyperparameters (dataset path, augmentation settings, etc.)
  - Training parameters (epochs, learning rate, optimizer)
  - Quantization settings for QAT
  - Optional path to pre-trained weights for fine-tuning

Training is then perform by calling the following command, which is provided by the LATTE python package:

```
latte train experiment-1.yaml imports.py
```

where `experiment-1.yaml` is the configuration file, and `imports.py` is the Python module that
imports and registers with LATTE the model function, dataset handlers, and any custom losses or
metrics. LATTE automatically manages checkpoints and logs, and if training is stopped, it can be
resumed by reruning the same command.

Once the model is trained / fine-tuned, the tests on test dataset(s) can be performed by running the
following command:

```
latte test experiment-1.yaml imports.py
```

By default, LATTE selects the best checkpoint, i.e. the one corresponding to the epoch that achieved
the lowest loss or highest evaluation metrics during training. The configuration file can also
specify that LATTE should run tests using the ML Engine Simulator, which executes the model as if it
were running on an FPGA.

The trained model can then be converted into a file format compatible with the sensAI SDK compiler
by running the following command.

```
latte convert experiment-1.yaml imports.py -c sensai-h5
```

Similar to testing, the conversion process uses the model weights from the best checkpoint by
default. The model file (here an H5) can then be used as input to the [sensAI
Compiler](https://github.com/sensAI-Solution/sensAICompiler) to obtain a compiled model for use on
FPGA.
