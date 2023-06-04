from multiprocessing import Pool
import time

def f(x):
    return x[0]*x[1]

def mutil_process():
    with Pool(5) as p:
        print(p.map(f,[(1,2),(2,3),(2,4)]))

if __name__ == '__main__':
    #print(f([1,2,3])
    mutil_process()