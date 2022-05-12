# MTP
## Abstract
Accurately predicting bugs and defects in software units helps the developers and testers to
find the defective unit and save their efforts in other software developing aspects. Previous studies
used the concept of machine-learning to build models to detect defective units in software. We re-
visited the previous studies and pointed out the areas of potential improvements. The conventional
classifiers give poor results as they perform well in only some part of data, so there is a need of
an advance classifier which is not very complex in implementation and give a decent result. This
study will provide a way to use a popular concept of Mixture of Experts in this domain and will
provide comparison of various variations.
Keywords: Software defect prediction, Machine Learning, Mixture of experts.

## Dataset
Dataset used will be a publicly available dataset provided by AEEEM-JIRA-PROMISE,
which includes 22 software defect datasets in ARFF format. It is one of the biggest
exploration information stores in computer programming. It is an assortment of various datasets,
including the NASA dataset which was utilized by various investigations before. Other than numerous different classes like code investigation, testing, software maintenance, it also contains a category for defects.

## Tools used
Anaconda environment along with python was used, some important python libraries and
packages used is listed below-
* Pandas: In software programming, pandas is a product library composed for the Python
programming language for information control and investigation. Specifically, it offers 11
information designs and activities for controlling mathematical tables and time series. It
is free programming delivered under the three-condition BSD permit.
* NumPy: NumPy is a library for the Python programming language, adding support
for enormous, multi-dimensional exhibits and frameworks, alongside a huge assortment of
significant level numerical capacities to work on these arrays.
* MatplotLib: Matplotlib is a complete library for making static, vivified, and intelligent
representations in Python.It has an assortment of capacities that make matplotlib work
like MATLAB

## Folder structure
All the datasets are present in dataset folder and all the models resides inside Models folder.

## Steps to run - 
* Install the python setup
* Run any notebook of your choice by clicking the run button of notebook.
* To use the ME models, import them from models.py and call the functions by passing suitable arguments.
