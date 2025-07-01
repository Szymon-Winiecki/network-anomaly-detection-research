# Datasets

There are several extrnal datasets used in this research. Those are not included in the repository, however one can easily get them from the original source. List of the datasets with links and instructions how to reproduce folder structure is given below. 

### [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

1. Download `CSV Files` folder (~700MB) from OneDrive linked on the dataset's page ([OneDrive](https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset&ga=1)).

2. The folder should be downloaded in .zip format, if not (or if you want to reduce file size with higher compression level) repack the archive to .zip format

3. Place `CSV Files.zip` in the `data/UNSW-NB15/raw/` directory

### [KDD Cup 1999](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

1. Download `kddcup.names`, `kddcup.data.gz`, `corrected.gz` from [dataset's page](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) and place them in the `data/KDDCUP99/raw` directory

### [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

1. Download `CIC-IDS-2017/CSVs/MachineLearningCSV.zip` file from [dataset's download page](http://cicresearch.ca/CICDataset/CIC-IDS-2017/) and place `MachineLearningCSV.zip` in the `data/CIC-IDS2017/raw` directory

### [CTU-13](https://www.stratosphereips.org/datasets-ctu13)

1. Run `download_CTU13.bat` script to download scenarios 8, 9, 10 and 13 of CTU-13 dataset.

2. One can optionally compress download csv files  to .tar.gz to reduce disk usage (preprocessing script can read data both from csv and .tar.gz file)
