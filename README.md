# HWAGNN
![](.\Fig\model.png)

> Recent studies have shown that integrating wavelets into Graph Neural Networks (GNNs) can effectively enhance graph representation learning. However, challenges such as feature extraction and heterophilic issues remain underexplored, with two key limitations in existing works: (1) Graph node features contain hierarchical spectral information, where low-frequency components capture global patterns and high-frequency components encode localized details. Existing methods often overlook this hierarchy and rely on complex graph filters, limiting their ability to fully exploit multi-frequency information. (2) Most approaches employ multi-head attention to fuse different components, resulting in high computational cost and poor alignment with the properties of wavelet transforms. To this end, we propose a multi-level wavelet transform framework that iteratively decomposes node embeddings into different frequency components, enabling the extraction of both short- and long-range features. We further introduce a lightweight Wavelet Attention (WaveAttn) module to adaptively fuse low- and high-frequency representations at each level, effectively capturing both structural distance and heterophily characteristics. Based on these designs, we develop the Hierarchical Wavelet Attention Graph Neural Network (HWAGNN), a simple yet robust framework for heterophilic graph representation learning. Extensive experiments conducted on 17 benchmarks confirmed that our proposed HWAGNN is efficient and effective.

## Dependencies
- python 3.8.19
- pytorch 1.12.0
- torchvision 0.13.0
- torchaudio 0.12.0
- scikit-learn 1.3.2
- pywavelets==1.4.1
- torch-geometric==2.1.0

## Easy Start

- Unzip the datasets

  ```
  unzip data.zip -d data
  ```

- Install the dependences: 

  ```
  pip install -r requirement.yaml
  ```

- Executing the following command to run the code:

  ```
  bash run.sh
  ```

- To run the model on other settings, use `--dataset` to specify datasets, `--model` to indicate the running model, `--wave` to set the wavelet function for DWT, `--levels` to set the wavelet decomposition levels, and `--device` to set the device number.

