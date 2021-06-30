import math

class TinyStatistician:
    def __init__(self):
        pass

    def mean(self, datas):
        return sum(datas) / len(datas)
        # for data in datas:

    def median(self, datas):
        datas = sorted(datas)
        mid = int(len(datas)/2)
        if len(datas) % 2 == 0:
            return (datas[mid - 1] + datas[mid]) / 2
        else:
            mid = int(mid)
            return datas[mid]

    def quartile(self, datas, quartile):
        datas = sorted(datas)
        quartile = int(len(datas)*quartile/100)
        if len(datas) % 2 == 0:
            return (datas[quartile - 1] + datas[quartile]) / 2
        else:
            quartile = int(quartile)
            return datas[quartile]

    def var(self, datas):
        datas = sorted(datas)
        mean = self.mean(datas)
        diff = [(x - mean) ** 2 for x in datas]
        return self.mean(diff)

    def std(self, datas):
        return math.sqrt(self.var(datas))



if __name__ == "__main__":
    a = [1, 42, 300, 10, 59]
    print("mean: ", TinyStatistician().mean(a))
    # Output:
    # 82.4
    print("median :", TinyStatistician().median(a))
    # Output:
    # 42.0

    print("quartile 25% :", TinyStatistician().quartile(a, 25))
    # Output:
    # 10.0

    print("quartile 75 :", TinyStatistician().quartile(a, 75))
    # Output:
    # 59.0

    print("variance :", TinyStatistician().var(a))
    # Output:
    # 12279.439999999999

    print("ecart-type :", TinyStatistician().std(a))
    # Output:
    # 110.81263465868862
    exit()