import kagglehub
import shutil
import os

source = kagglehub.dataset_download("jakewright/house-price-data")

destination = os.path.join(os.getcwd(), "data", "housing-in-london")

shutil.move(source, destination)

