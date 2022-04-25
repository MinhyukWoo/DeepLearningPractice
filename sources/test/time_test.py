from time import time_ns

if __name__ == '__main__':
    last = 100000

    t0 = time_ns()
    for i in range(last):
        a = i * i
        if i + 1 == last:
            b = i * i * i
    t1 = time_ns()
    print('time 1:', t1 - t0)

    t2 = time_ns()
    for j in range(last - 1):
        c = j * j
    d = last * last * last
    t3 = time_ns()
    print('time 2:', t3 - t2)