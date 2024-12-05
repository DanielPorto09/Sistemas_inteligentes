import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Separando a base de treino em treino e teste  [
data_train, labels_train, data_test, labels_test = train_test_split


