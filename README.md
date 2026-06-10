# DiffTW: Time Series Classification through Diffeomorphic Time Warping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for the paper: **"Time Series Classification through Diffeomorphic Time Warping (DiffTW)"** by Vicky Haney, Kamel Lahouel, Victor Rielly, and Bruno Jedynak (2026).

## 📖 Abstract
Time series classification involves learning a mapping from a continuous, temporally ordered sequence of real-valued observations to a discrete response variable. While Dynamic Time Warping (DTW) is a standard technique for measuring similarity, it is restricted to discrete point matching. 

To move beyond pairwise alignment, we propose **DiffTW**, a theoretical framework that learns mappings between real-valued functions. These mappings approximate the flow associated with the characteristic curves of a linear transport equation with a space-dependent velocity field, providing a diffeomorphic transformation between two time series. DiffTW optimizes an underlying vector field using the fundamental theorem of calculus and Random Fourier Features to ensure smooth, continuous alignment.

## ⚙️ Installation

Clone the repository and install the required dependencies. We recommend using a virtual environment.

```bash
git clone [https://github.com/vgeneva/DiffTW.git](https://github.com/vgeneva/DiffTW.git)
cd DiffTW
pip install -r requirements.txt
