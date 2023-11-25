# phd_utils

A repository of helper functions for the various projects done during my PhD.

## Usage

Clone and install this repository using:

```bash
git clone https://github.com/souljaboy764/phd_utils --recursive
cd phd_utils
pip install -r requirements.txt
pip install -e .
```

### Dataset preprocessing

Currently the Buetepage dataset and the NuiSI dataset are preprocessed and stored in the [`data_preproc`](data_preproc) folder. To manually preprocess the data, please run:
```bash
python data_raw/nuisi_preproc.py --src data_raw/nuisi_dataset/ --dst data_preproc/nuisi # for the NuiSI dataset
python data_raw/buetepage_preproc.py --src data_raw/human_robot_interaction_data/ --dst data_preproc/buetepage # for the Buetepage HHI dataset
python data_raw/buetepage_preproc.py --src data_raw/human_robot_interaction_data/ --dst data_preproc/buetepage_hr --robot # for the Buetepage HRI dataset
```
