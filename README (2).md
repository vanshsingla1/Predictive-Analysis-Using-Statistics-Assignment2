# Learning Probability Density Function using GAN

## Project Overview

This project learns the probability density function (PDF) of a
transformed NO₂ concentration variable using a Generative Adversarial
Network (GAN). Since the analytical form of the distribution is unknown,
the GAN learns the distribution directly from data samples.

------------------------------------------------------------------------

## Dataset

-   Source: India Air Quality Dataset (Kaggle)
-   Feature Used: NO₂ concentration
-   Missing values were removed before processing.

------------------------------------------------------------------------

## Step 1: Data Transformation

Each NO₂ value x is transformed into z:

z = x + a_r sin(b_r x)

Where: a_r = 0.5 × (r mod 7)
b_r = 0.3 × ((r mod 5) + 1)

r = University roll number

------------------------------------------------------------------------

## Step 2: GAN Methodology

A GAN is used because the PDF of z is unknown.

### GAN Architecture

  Component       Structure
  --------------- --------------------------------------
  Generator       3-layer MLP with ReLU
  Discriminator   3-layer MLP with LeakyReLU + Sigmoid
  Noise Input     1D Gaussian N(0,1)
  Loss Function   Binary Cross Entropy
  Optimizer       Adam (lr = 0.0002)

### Training Process

1.  Generator creates fake samples.
2.  Discriminator distinguishes real vs fake samples.
3.  Generator improves by trying to fool the discriminator.
4.  Training continues until distributions match.

------------------------------------------------------------------------

## Step 3: PDF Approximation

After training: - 10,000 samples are generated. - PDF estimated using
Histogram and Kernel Density Estimation (KDE).

------------------------------------------------------------------------

## Results

### Histogram Comparison

GAN-generated samples overlap closely with real z distribution.

<img width="680" height="451" alt="image" src="https://github.com/user-attachments/assets/1e1ed117-0c22-4421-8354-ab9ce4c5c9a6" />

### KDE Curve

Smooth curve confirms GAN captured main distribution modes.

<img width="707" height="470" alt="image" src="https://github.com/user-attachments/assets/db141b54-2c34-441a-8c21-7cd79afdd1b6" />

------------------------------------------------------------------------

## Result Table

  Aspect               Observation
  -------------------- ----------------------------------
  Mode Coverage        Major peaks captured
  Training Stability   Stable oscillations
  Data Quality         Generated data matches real data
  PDF Approximation    KDE smooth and accurate

------------------------------------------------------------------------

## Observations

-   GAN successfully learns unknown distribution.
-   Training remains stable without divergence.
-   Generated samples closely follow real data distribution.

------------------------------------------------------------------------

## Tools Used

Python, PyTorch, NumPy, Matplotlib, Scikit-learn, Google Colab

------------------------------------------------------------------------

## Conclusion

This project demonstrates that GANs can learn complex probability
density functions purely from sample data without requiring the
analytical PDF.
