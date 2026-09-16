## Data

The data folder is not included in this repository. It is excluded
through .gitignore because the dataset is several gigabytes, too large
for Git and GitHub.

Dataset: Breast Histopathology Images (IDC)
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

To reproduce this project, download the dataset from the link above, then
place the patient folders directly inside data, so the structure looks
like this.

```text
data/
  10253/
    0/
    1/
  8864/
    0/
    1/
```

Each numbered folder is a patient ID. The 0 subfolder holds IDC negative
patches and the 1 subfolder holds IDC positive patches. There are 279
patient folders and 277,524 total patches.

If the Kaggle zip extracts a nested IDC_regular_ps50_idx5 wrapper folder
containing a duplicate copy of the same patient folders, only one copy is
needed. Move the patient folders directly into data and remove the
wrapper.