import os
import pickle
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm

from insightface.app import FaceAnalysis
from sklearn.neighbors import NearestNeighbors