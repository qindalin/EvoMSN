# EvoMSN
This repo is the Pytorch implementation of our submitted paper to ICLR 2025: Evolving Multi-Scale Normalization for Time Series Forecasting Under Distribution Shifts

### Usage

#### Environment and dataset setup

```bash
pip install -r requirements.txt
mkdir datasets
```
All the 9 datasets are available at the [Google Driver](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by Autoformer. Many thanks to their efforts and devotion!

#### Running

We provide ready-to-use scripts for **EvoMSN** enhanced **online** forecasting and **MSN** enhanced **offline** forecasting.

```bash
sh run_<model_name>.sh 
```
### Acknowledgement

This repo is built on the pioneer works. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

[SAN](https://github.com/icantnamemyself/SAN)

[FSNet](https://github.com/salesforce/fsnet)
