import multiprocessing as mp
import time

def f(i,j):
    localinit = time.time()
    time.sleep(1)
    elaps = time.time()-localinit
    print(f"{i+j}, time = {elaps}")

if __name__ == "__main__":

    localinit = time.time()

    processes = []

    for j in range(10):
        p = mp.Process(target=f,args=[j,j])
        p.start()
        processes.append(p)

#    for process in processes:
#        process.join()

    elaps = time.time()-localinit

    print(f"Ended, elapsed: {elaps}")
