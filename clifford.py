import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import numba

HOMEDIR = os.environ['HOME']
os.chdir(HOMEDIR + "/projects/clifford_alg/data")
cliffords2 = np.load('cliffords_2Qnp.npy')
rtol=1e-03
atol=1e-05
L = 500

def flatten_and_uniquenize(clifford):
    flattened = clifford.ravel('C')
    for j in range(len(flattened)):
        if not np.isclose(0, flattened[j]):
            normalizer = np.conj(flattened[j]) / np.abs(flattened[j])
            break
        else:
            continue
    flattened = flattened * normalizer
    return flattened

def uniquenize(cliffords):
    p = Pool()
    ret = p.map(flatten_and_uniquenize, cliffords)
    p.close
    return np.asarray(ret)

def uniquenize_elements(flatarray):
    for i in range(len(flatarray)):
        for j in range(i, len(flatarray)):
            if np.isclose(flatarray[i], flatarray[j], rtol, atol):
                flatarray[j] = flatarray[i]
    return np.unique(flatarray)

def create_element_hashdict(cliffords):
    dict = {}
    flatten = cliffords.ravel('C')
    list = uniquenize_elements(np.unique(flatten))
    for i in range(len(list)):
        dict[i] = list[i]
    return dict

def hashnize(clifford, dict):
    hexstr = ""
    for i in range(len(clifford)):
        for j in range(len(dict)):
            if np.isclose(dict[j], clifford[i]):
                hexstr += format(j, 'x')
                break
            else:
                continue
    return hexstr

def wrapper_hashnize(arg):
    return hashnize(*arg)

def create_cliffords_hashdict(cliffords, elementDict):
    p = Pool()
    ret = p.map(wrapper_hashnize, [(clifford, elementDict) for clifford in cliffords])
    p.close
    dict = {}
    for i in range(len(ret)):
        dict[i] = ret[i]
    return dict

def prod_and_search(clifford1, clifford2, elementDict, hashDict):
    product = np.matmul(clifford1, clifford2)
    product = flatten_and_uniquenize(product)
    prodHash = hashnize(product, elementDict)
    for key, value in hashDict.items():
        if value == prodHash:
            return key
        else:
            continue
    print("{} is missing".format(prodHash))

# don't use
def wrapper_prod_and_search(args):
    return prod_and_search(*args)

def calc_sub(clifford1, cliffords, elementDict, hashDict):
    ret = []
    for clifford in cliffords:
        ret.append(prod_and_search(clifford1, clifford, elementDict, hashDict))
    return np.asarray(ret)

def wrapper_calc_sub(args):
    return calc_sub(*args)

def calc(cliffords, elementDict, hashDict):
    p = Pool()
    ret = p.map(wrapper_calc_sub, [(clifford, cliffords, elementDict, hashDict) for clifford in cliffords])
    p.close
    return np.asarray(ret)

def make_group_map(cliffords):
    uniqueCliffords = uniquenize(cliffords)
    elementDict = create_element_hashdict(uniqueCliffords)
    cliffordsHashDict = create_cliffords_hashdict(uniqueCliffords, elementDict)
    group_map = calc(cliffords[:L], elementDict, cliffordsHashDict)
    return group_map

np.savetxt("cliffords.csv", make_group_map(cliffords2), delimiter=',' , fmt = '%i')