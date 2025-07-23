# UNet++ Brain Tumor Segmentation

## Overview

This project implements a full pipeline for **brain tumor segmentation** using a **UNet++** convolutional neural network. The pipeline handles:

* Training a nested encoder-decoder segmentation model on paired brain MRI slices and tumor masks.
* Visualizing the model’s nested architecture.
* Predicting segmentation masks for unseen MRI scans.
* Deriving a binary tumor/no-tumor classification signal from the pixel-wise segmentation mask.
* Saving and reloading the model for further inference or evaluation.

The trained model weights are available [here](https://drive.google.com/file/d/1qhJ7T--3H981gHRZAGBsRNw9CPXnbkei/view?usp=sharing).
**Note:** This model was trained on limited compute resources (Google Colab free tier) with a small number of training epochs and reduced steps per epoch, so it is intended as a demonstrative prototype. Performance will not match production-grade models trained on high-volume compute infrastructure.

---

## What is UNet++?

**UNet++**, introduced by Zhou et al. in *UNet++: A Nested U-Net Architecture for Medical Image Segmentation* (MICCAI 2018), is an advanced extension of the classical **UNet** architecture. While UNet is a well-established encoder-decoder structure for biomedical image segmentation, UNet++ enhances this architecture with **nested dense skip pathways** between encoder and decoder sub-networks.

Key properties:

* **Nested and dense skip connections:** Unlike UNet, which connects encoder and decoder layers only once at each level, UNet++ inserts intermediate convolution blocks between encoder and decoder paths, forming a grid-like topology. This progressive refinement improves feature fusion between low-level and high-level representations.
* **Improved gradient flow:** The dense skip pathways help gradients propagate more effectively, enabling deeper segmentation models.
* **Better segmentation quality on heterogeneous regions:** The nested connections improve boundary delineation, which is crucial for accurately segmenting small or fuzzy tumors in brain MRIs.

---

## Why UNet++ instead of UNet?

* **Improved localization:** For medical image segmentation, especially when tumors occupy small regions of an image, standard UNet skip connections can cause coarse or imprecise boundaries. UNet++ addresses this with nested pathways that enable multi-scale feature re-exploitation.
* **Parameter efficiency:** With proper pruning and configuration, UNet++ achieves higher performance without significantly increasing parameter count relative to classical UNet.
* **State-of-the-art performance:** UNet++ and its variants remain competitive benchmarks for brain tumor segmentation tasks in medical image computing challenges.

---

## Technical Workflow

1. **Architecture**:
   The network is built using Keras/TensorFlow 2.x. It consists of a five-level encoder-decoder structure:

   * Encoder: Successive convolutional blocks with batch normalization, followed by max pooling.
   * Decoder: Progressive upsampling, concatenation with encoder features via nested skip pathways, and refinement through repeated convolutional blocks.
   * Final output: A 1-channel sigmoid-activated convolution layer predicting pixel-wise tumor presence.

2. **Data Pipeline**:

   * Training images and masks are read from disk as PNGs.
   * Images are resized to 256×256 pixels and normalized.
   * Masks are binarized above a threshold and resized.
   * TensorFlow `tf.data.Dataset` handles batching, shuffling, and prefetching.

3. **Training**:

   * Loss function: Binary cross-entropy.
   * Optimizer: Adam with a small learning rate (1e-4).
   * Training runs for a limited number of steps per epoch due to constrained compute.

4. **Model Saving**:
   The trained model is saved in HDF5 (`.h5`) format. This snapshot includes weights, architecture, and optimizer state.

5. **Inference**:
   For inference, the model:

   * Loads an unseen test MRI slice.
   * Predicts the segmentation mask.
   * Binarizes the output mask.
   * Computes the tumor area ratio.
   * Declares a binary tumor/no-tumor decision if the mask area exceeds a predefined threshold.
   * Overlays the mask onto the original image for visual inspection.

---

## How to Use

1. Mount Google Drive or manually download the `.h5` model file.
2. Load the model with:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model("path/to/unet_model.h5")
   ```
3. Prepare an input MRI slice (PNG, 256×256) and preprocess as done during training.
4. Run `model.predict` to obtain the segmentation mask.
5. Threshold the mask and inspect the tumor area ratio to infer tumor presence.
6. Visualize the mask overlay for interpretability.

---

## Limitations

* The model was trained on a small subset of slices using the free Colab tier. For clinical-grade performance, larger datasets and robust compute are essential.
* The binary tumor/no-tumor classification is a post-processing heuristic based on segmentation output, not a dedicated classification branch.
* This prototype does not include test-time augmentation or ensemble averaging.

---

## Reference

Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., & Liang, J. (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*. MICCAI 2018.

---

## Caution

This repository is intended for academic and educational use only. It does not provide medical advice or diagnostic functionality for clinical use.

---

**Contact:**
For questions, please contact the repository author or open an issue.

