# LHR-of-AIM-in-ISCA25

## Overview

## Table of Contents

- [LHR-of-AIM-in-ISCA25](#lhr-of-aim-in-isca25)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Citation](#citation)

## Features

- **1**: LHR (lower hamming rate) is a component of AIM (AIM: Software and Hardware Co-design for Architecture-level IR-drop Mitigation in High-performance PIM in ISCA'25).
- **2**: LHR is a regularization term to reduce HR by penalizing high-HR weights with negligible accuracy loss during quantization. 
- **3**: LHR can be easily combined with QAT or PTQ for lower hamming rate in quantized weights.

## Installation

```shell
mkdir data
pip install -r requirements
```
## Usage

1. prepare imagenet in data like ./data/imagent
2. Use the following command to optimize resnet18
   ```python
   python resnet/train_resnet.py --arch resnet18  --w-bit 8 --x-bit 8
   ```

## Contributing


## Citation

If you find it useful in your research, please consider citing our paper: Yuanpeng Zhang, Xing Hu, Xi Chen, Zhihang Yuan, Cong Li, Jingchen Zhu, Zhao Wang, Chenguang Zhang, Xin Si, Wei Gao, Qiang Wu, Runsheng Wang, and Guangyu Sun. 2025. AIM: Software and Hardware Co-design for Architecture-level IR-drop Mitigation in High-performance PIM. In Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA ’25), June 21–25, 2025, Tokyo, Japan. ACM, New York, NY, USA, 18 pages. https://doi.org/10.1145/3695053.3730987

```bibtex



