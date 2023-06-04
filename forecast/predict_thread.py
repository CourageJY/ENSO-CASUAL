from threading import Thread

class PredictThread(Thread):
    def __init__(self, func, args):
        super(PredictThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def cal_sum(begin, end):
    # global _sum
    _sum = 0
    for i in range(begin, end + 1):
        _sum += i
    return  _sum

if __name__ == '__main__':
    t1 = PredictThread(cal_sum, args=(1, 5))
    t2 = PredictThread(cal_sum, args=(6, 10))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = t1.get_result()
    res2 = t2.get_result()

    print(res1 + res2)
