import numpy as np

class WtKendallTau:
    def __init__(self, loc_x: np.ndarray, loc_y: np.ndarray, loc_w: np.ndarray):
        self._loc_x = loc_x
        self._loc_y = loc_y
        self._loc_w = loc_w

    def compute(self) -> float:
        l = len(self._loc_x)
        ranked_data = self._rank(self._loc_x, self._loc_y, self._loc_w)
        ranked_idx = np.zeros(l, dtype=np.int32)
        ranked_w = np.zeros(l, dtype=np.float64)

        for i in range(l):
            ranked_idx[i] = int(ranked_data[i, 0])
            ranked_w[i] = ranked_data[i, 2]

        merge_sort = MergeSort(ranked_idx, ranked_w)
        swap = merge_sort.sort()
        tw = self._totw(self._loc_w)/2
        
        return (tw - 2 * swap) / tw
    
    def _rank(self, loc_x: np.ndarray, loc_y: np.ndarray, loc_w: np.ndarray) -> np.ndarray:
        combo_data = np.column_stack((loc_x, loc_y, loc_w))

        # sort by x
        combo_data = combo_data[np.argsort(combo_data[:, 0])]
        start = 0
        end = 0
        rank = 0
        while end < combo_data.shape[0] -  1:
            while np.isclose(combo_data[start, 0], combo_data[end, 0]):
                end += 1
                if end >= combo_data.shape[0]:
                    break
            for i in range(start, end):
                combo_data[i, 0] = rank + np.random.random()
            rank += 1
            start = end

        combo_data = combo_data[np.argsort(combo_data[:, 0])]
        for i in range(len(loc_x)):
            combo_data[i, 0] = i + 1

        # sorty by y
        combo_data = combo_data[np.argsort(combo_data[:, 1])]
        start = 0
        end = 0
        rank = 0
        while end < combo_data.shape[0] - 1:
            while np.isclose(combo_data[start, 1], combo_data[end, 1]):
                end += 1
                if end >= combo_data.shape[0]:
                    break
            for i in range(start, end):
                combo_data[i, 1] = rank + np.random.random()
            rank += 1
            start = end

        combo_data = combo_data[np.argsort(combo_data[:, 1])]
        for i in range(len(loc_x)):
            combo_data[i, 1] = i + 1

        return combo_data
    
    def _totw(self, w):
        sum_w = 0
        sum_square_w = 0
        for i in range(len(w)):
            sum_w += w[i]
            sum_square_w += w[i] * w[i]

        return sum_w * sum_w - sum_square_w


class MergeSort:
    def __init__(self, ranked_idx: np.ndarray, ranked_w: np.ndarray):
        self._ranked_idx = ranked_idx
        self._ranked_w = ranked_w

    def compare_int(self, a: int, b: int):
        return (a > b) - (a < b)
    
    def sort(self) -> float:
        swap: float = 0.0
        step: int = 1
        begin1: int = 0
        begin2: int = 0
        end: int = 0
        k: int = 0
        n: int = len(self._ranked_idx)
        idx1: np.ndarray = self._ranked_idx.copy()
        idx2: np.ndarray = np.zeros(n, dtype=np.int32)
        w1: np.ndarray = self._ranked_w.copy()
        w2: np.ndarray = np.zeros(n, dtype=np.float64)
        cum_w: np.ndarray = np.zeros(n, dtype=np.float64)

        while step < n:
            begin1 = 0
            k = 0
            cum_w[0] = w1[0]
            for i in range(1, n):
                cum_w[i] = cum_w[i - 1] + w1[i]
            
            while True:
                begin2 = begin1 + step
                end = begin2 + step
                if end > n:
                    if begin2 > n:
                        break
                    end = n

                i = begin1
                j = begin2
                while i < begin2 and j < end:
                    if self.compare_int(int(idx1[i]), int(idx1[j])) > 0:
                        if i == 0:
                            temp_swap = w1[j] * cum_w[begin2 - 1]
                        else:
                            temp_swap = w1[j] * (cum_w[begin2 - 1] - cum_w[i - 1])
                        swap += temp_swap
                        idx2[k] = idx1[j]
                        w2[k] = w1[j]
                        k += 1
                        j += 1
                    else:
                        idx2[k] = idx1[i]
                        w2[k] = w1[i]
                        k += 1
                        i += 1

                if i < begin2:
                    while i < begin2:
                        idx2[k] = idx1[i]
                        w2[k] = w1[i]
                        k += 1
                        i += 1
                else:
                    while j < end:
                        idx2[k] = idx1[j]
                        w2[k] = w1[j]
                        k += 1
                        j += 1

                begin1 = end

            if k < n:
                while k < n:
                    idx2[k] =  idx1[k]
                    w2[k] = w1[k]
                    k += 1

            for i in range(n):
                idx1[i] = idx2[i]
                w1[i] = w2[i]

            step *= 2

        return swap