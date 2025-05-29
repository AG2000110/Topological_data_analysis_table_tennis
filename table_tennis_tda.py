#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy as sp
import pandas as pd
import gudhi as gd
import csv
import os
import matplotlib.pyplot as plt

from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering
from gudhi.wasserstein import wasserstein_distance
from gtda.diagrams import PairwiseDistance
from persim import bottleneck
from persim import sliced_wasserstein

get_ipython().run_line_magic('matplotlib', 'inline')


# wczytanie pliku
df = pd.read_excel("table_tennis_stats.xlsx", engine="openpyxl")

#print(df.columns)
#print(df)

ATS_Akanza_AZS_UMCS_Lublin = []
KS_Bronowianka_Krakow_II = []
KTS_Gliwice = []
KU_AZS_Politechnika_Lublin = []
Luvena_LKTS_Lubon = []
MKS_Czechowice_Dziedzice = []
Palmiarnia_ZKS_Zielona_Gora = []
Plotbud_Skarbek_Tarnowskie_Gory = []
Uniwersytet_Ekonomiczny_AZS_Wroclaw_II = []
Wamet_KS_Dabcze = []



for row in df.values:
    klub = row[1] 
    zawodniczka = list(map(int, row[2:]))  

    if klub == "KS Bronowianka Kraków II":
        KS_Bronowianka_Krakow_II.append(zawodniczka)
    elif klub == "ATS Akanza AZS UMCS Lublin":
        ATS_Akanza_AZS_UMCS_Lublin.append(zawodniczka)
    elif klub == "KTS Gliwice":
        KTS_Gliwice.append(zawodniczka)
    elif klub == "KU AZS Politechnika Lublin":
        KU_AZS_Politechnika_Lublin.append(zawodniczka)
    elif klub == "Luvena LKTS Luboń":
        Luvena_LKTS_Lubon.append(zawodniczka)
    elif klub == "MKS Czechowice-Dziedzice":
        MKS_Czechowice_Dziedzice.append(zawodniczka)
    elif klub == "Palmiarnia ZKS Zielona Góra":
        Palmiarnia_ZKS_Zielona_Gora.append(zawodniczka)
    elif klub == "Płotbud Skarbek Tarnowskie Góry":
        Plotbud_Skarbek_Tarnowskie_Gory.append(zawodniczka)
    elif klub == "Uniwersytet Ekonomiczny AZS Wrocław II":
        Uniwersytet_Ekonomiczny_AZS_Wroclaw_II.append(zawodniczka)
    elif klub == "Wamet KS Dąbcze":
        Wamet_KS_Dabcze.append(zawodniczka)
    else:
        print(0)


# print("KS Bronowianka Kraków:", KS_Bronowianka_Krakow_II)
# print()
# print("ATS Akanza AZS UMCS Lublin:", ATS_Akanza_AZS_UMCS_Lublin)
# print()
# print("KTS Gliwice:", KTS_Gliwice)
# print()
# print("KU AZS Politechnika Lublin:", KU_AZS_Politechnika_Lublin)
# print()
# print("Luvena LKTS Luboń:", Luvena_LKTS_Lubon)
# print()
# print("MKS Czechowice-Dziedzice:", MKS_Czechowice_Dziedzice)
# print()
# print("Palmiarnia ZKS Zielona Góra:", Palmiarnia_ZKS_Zielona_Gora)
# print()
# print("Płotbud Skarbek Tarnowskie Góry:", Plotbud_Skarbek_Tarnowskie_Gory)
# print()
# print("Uniwersytet Ekonomiczny AZS Wrocław II:", Uniwersytet_Ekonomiczny_AZS_Wroclaw_II)
# print()
# print("Wamet KS Dąbcze:", Wamet_KS_Dabcze)


def simplex_Tree_create(data):
    data_array = np.array(data)
    skeleton = gd.RipsComplex(
        points=data_array
    ) 
    Simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    return Simplex_tree


def filtration(Simplex_Tree, folder, club_names, method="bottleneck"):
    os.makedirs(f'diag/{folder}', exist_ok=True)
    for dim in range(3):
        os.makedirs(f'diag{dim}/{folder}', exist_ok=True)

    def compute_distance(diag1, diag2):
        if method == "bottleneck":
            return bottleneck(diag1, diag2)
        elif method == "wasserstein":
            return wasserstein_distance(diag1, diag2, matching=True, order=1, internal_p=2)[0]
        elif method == "sliced":
            return sliced_wasserstein(diag1, diag2)
        else:
            raise ValueError(f"Nieznana metoda: {method}")

    def remove_infinite(diagram):
        return diagram[np.isfinite(diagram).all(axis=1)]

    for dimension in range(3):
        dim_intervals = []

        for tree in Simplex_Tree:
            tree.compute_persistence()
            interv = tree.persistence_intervals_in_dimension(dimension)
            interv = remove_infinite(interv) if method == "sliced" else interv
            dim_intervals.append(interv)

        for i, interv in enumerate(dim_intervals):
            gd.plot_persistence_barcode(interv)
            plt.title(club_names[i])
            plt.xlim(0, 130)
            plt.savefig(f'diag{dimension}/{folder}/diagram_{club_names[i]}.png')
            plt.close()

        n = len(dim_intervals)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                d = compute_distance(dim_intervals[i], dim_intervals[j])
                dist[i][j] = dist[j][i] = d

        clustering = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
        labels = clustering.fit_predict(dist)

        print(f"\nEtykiety klastrów dla wymiaru {dimension} (metoda: {method}):")
        for name, label in zip(club_names, labels):
            print(f"{name}: {label}")



Simplex_Tree = []
Simplex_Tree.append(simplex_Tree_create(ATS_Akanza_AZS_UMCS_Lublin))
Simplex_Tree.append(simplex_Tree_create(KS_Bronowianka_Krakow_II))
Simplex_Tree.append(simplex_Tree_create(KTS_Gliwice))
Simplex_Tree.append(simplex_Tree_create(KU_AZS_Politechnika_Lublin))
Simplex_Tree.append(simplex_Tree_create(Luvena_LKTS_Lubon))
Simplex_Tree.append(simplex_Tree_create(MKS_Czechowice_Dziedzice))
Simplex_Tree.append(simplex_Tree_create(Palmiarnia_ZKS_Zielona_Gora))
Simplex_Tree.append(simplex_Tree_create(Plotbud_Skarbek_Tarnowskie_Gory))
Simplex_Tree.append(simplex_Tree_create(Uniwersytet_Ekonomiczny_AZS_Wroclaw_II))
Simplex_Tree.append(simplex_Tree_create(Wamet_KS_Dabcze))

club_names = [
    "ATS Akanza AZS UMCS Lublin",
    "KS Bronowianka Kraków II",
    "KTS Gliwice",
    "KU AZS Politechnika Lublin",
    "Luvena LKTS Luboń",
    "MKS Czechowice-Dziedzice",
    "Palmiarnia ZKS Zielona Góra",
    "Płotbud Skarbek Tarnowskie Góry",
    "Uniwersytet Ekonomiczny AZS Wrocław II",
    "Wamet KS Dąbcze"
]

filtration(Simplex_Tree,'oryginal', club_names, method="bottleneck")


# In[ ]:




