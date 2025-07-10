# Project-SPE

Repository for the performance evaluation project of Simulation and Performance Evaluation course @ UNITN.

Powered by @me and @Sano.

# Setup
Need to install directory for datasets in CSV format from <a href="https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS">here</a>.

Then it must inserted in the /datasets directory inside the repository.

**Note**: rename the downloaded directory "CSV Files" in "CSV_Files".

# Project structure

PROJECT-SPE/
├── analysis/                                # Contains Python modules for statistical analysis
│   ├── a)all_to_all_correlation/            # Analysis of all-to-all feature correlation
│   ├── b)correlation_with_label/            # Correlation analysis between features and labels
│   └── c)correlation_with_attack_cat/       # Correlation analysis with attack categories
│
├── datasets/                               # Datasets used in the project
│   └── CSV_Files/                          # Raw and processed CSV dataset files
│       ├── training_testing_sets/          # Dataset splits for training and testing
│       ├── UNSW-NB15_features.csv          # Feature metadata
│       ├── UNSW-NB15_GT.csv                # Ground truth labels
│       ├── UNSW-NB15_1.csv to 4.csv        # Raw dataset in parts
│       ├── UNSW-NB15_LIST_EVENTS.csv       # List of events in dataset
│       └── The UNSW-NB15 description.pdf   # Dataset documentation
│
├── theoretical_notes/               # Notes on theoretical background
├── related-works/                   # Related works
├── idea_of_meaningful_features.txt  # Notes on selected features for analysis
├── Project-Idea.pdf                 # PDF document outlining the project idea
├── .gitignore                       # Git ignore file
└── README.md                        # Project overview and documentation
