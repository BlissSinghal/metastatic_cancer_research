from pathlib import Path
from typing import List
import os
from os import listdir

def load_data():
    path = Path("~/Downloads/cancer_data").expanduser()
    lines = (path/"train_labels.csv").read_text().splitlines()
    training_labels = dict(parse_labels(lines[1:len(lines)]))
    #test_set = dict(parse_labels(lines[10001: 20001]))
    return training_labels

def parse_labels(lines:List[str]):
    for line in lines:
        id, label = line.split(",")
        yield id, int(label)

