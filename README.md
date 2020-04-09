# layout-data

[![Build Status](https://travis-ci.com/zweien/layout-data.svg?branch=master)](https://travis-ci.com/zweien/layout-data)
![Python package](https://github.com/zweien/layout-data/workflows/Python%20package/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/zweien/layout-data/badge)](https://www.codefactor.io/repository/github/zweien/layout-data)
[![codecov](https://codecov.io/gh/zweien/layout-data/branch/master/graph/badge.svg)](https://codecov.io/gh/zweien/layout-data)

[layout-generator](https://git.idrl.site/idrl/layout-generator) 生成数据集配套 Dataset 封装，包含 [pytorch-lightning](https://github.com/PytorchLightning/pytorch-lightning) 封装的 FPN 模型，可以方便的配置参数。

- [x] LoadResponse
- [x] LayoutDataset

## Install

```bash
git clone https://git.idrl.site/zweien/layout-data.git
cd layout-data
pip install -U -e .
```

## Example

- `example` 目录下包含基本示例
  - `train.py`: 训练模型主程序
  - `config.yml`: 参数配置文件
- 使用 `python train.py -h` 查看支持的参数列表
  - `data_dir` 为数据集目录，该目录下应包含 `train` 与 `test` 子目录， 使用 [layout-generator](https://git.idrl.site/idrl/layout-generator) 生成相应数据
- `python train.py --config config.yml` 开始训练
- 默认使用 `tensorboard` 记录中间训练数据
