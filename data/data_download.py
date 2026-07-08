import kagglehub
import shutil
import os

source = kagglehub.dataset_download("justinas/housing-in-london")

destination = os.path.join(os.getcwd(), "data", "housing-in-london")

shutil.move(source, destination)
