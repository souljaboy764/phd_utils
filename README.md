# phd_utils

A repository of helper functions for the various projects done during my PhD.

## Usage

Clone and install this repository using:

```bash
git clone https://github.com/souljaboy764/phd_utils
cd phd_utils
pip install -r requirements.txt
pip install -e .
```

### Dataset preprocessing

Currently the Buetepage dataset and the NuiSI dataset are preprocessed and stored in the [`data_preproc`](data_preproc) folder. To manually preprocess the data, please run:

```bash
git submodule update --init
python data_raw/nuisi_preproc.py --src data_raw/nuisi_dataset/ --dst data_preproc/nuisi # for the NuiSI dataset
python data_raw/buetepage_preproc.py --src data_raw/human_robot_interaction_data/ --dst data_preproc/buetepage # for the Buetepage HHI dataset
python data_raw/buetepage_preproc.py --src data_raw/human_robot_interaction_data/ --dst data_preproc/buetepage_hr --robot # for the Buetepage HRI dataset
```

For the [Handovers Dataset](https://zenodo.org/records/7767535#.ZB2-43bMLIU), it is recommended to use the preprocessed data, as the original dataset is very large (8.9 GB compressed, 18.6GB uncompressed)
To preprocess the data, first download the dataset and unzip the file `Bimanual Handovers Dataset.zip` to a suitable location . Assuming that the unzipped folder is at `~/Documents/Bimanual Handovers Dataset/`, to preprocess the data, run:

```bash
python data_raw/alap_preproc.py --src ~/Documents --dst data_preproc/alap/ # dataset for the results reported in the paper
python data_raw/alap_preproc.py --src ~/Documents --dst data_preproc/alap_kobo/ --robot # dataset for training the model to be executed on the Kobo robot
```

Currently only the case of Robot-to-Human handover is explored
