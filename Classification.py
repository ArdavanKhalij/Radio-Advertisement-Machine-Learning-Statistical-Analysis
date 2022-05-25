# Libraries
import sklearn
import pandas as pd
import matplotlib

# Load dataset
column_names = ['B', 'G', 'R', 'Class']
read_file = pd.read_csv(r'Skin_NonSkin.txt')
read_file.to_csv(r'Skin_NonSkin.csv', index=None)
