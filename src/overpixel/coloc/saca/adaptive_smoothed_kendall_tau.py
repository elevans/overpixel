import numpy as np
import math
from typing import Sequence
from overpixel.coloc.saca.wt_kendall_tau import WtKendallTau

class AdaptiveSmoothedKendallTau:
    def __init__(self, narr_a:np.ndarray, narr_b: np.ndarray, thre_a: float, thre_b: float):
        self.narr_a = narr_a
        self.narr_b = narr_b
        self.thre_a = thre_a
        self.thre_b = thre_b
        self.shape = narr_a.shape
        self._Dn = None
        self._TL = None
        self._TU = None
        self._lambda = None
        self._stop = None

    def compute(self) -> np.ndarray:
        result = np.zeros(self.shape)
        oldtau = np.zeros(self.shape)
        newtau = np.zeros(self.shape)
        oldsqrt_n = np.zeros(self.shape)
        newsqrt_n = np.zeros(self.shape)
        self._stop = np.zeros((self.shape[0], self.shape[1], 3))
        self._Dn = math.sqrt(math.log(self.shape[0] * self.shape[1])) * 2
        self._TU = 15
        self._TL = 8
        self._lambda = self._Dn
        size = 1
        step_size = 1.15
        int_size = 0
        is_check = False

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                oldtau[i, j] = 0
                oldsqrt_n[i, j] = 1
                for k in range(3):
                    self._stop[i, j, k] = 0

        for s in range(self._TU):
            int_size = math.floor(size)
            self._singleiteration(
                oldtau, oldsqrt_n, newtau, newsqrt_n, result, int_size, is_check
            )
            size *= step_size
            if s == self._TL:
                is_check = True
                self._stop[:, :, 1] = newtau[:, :]
                self._stop[:, :, 2] = newsqrt_n[:, :]

        return result

    def _get_data(
            self,
            narr_a: np.ndarray,
            narr_b: np.ndarray,
            kernel: np.ndarray,
            oldtau: np.ndarray,
            oldsqrt_n: np.ndarray,
            loc_x: np.ndarray,
            loc_y: np.ndarray,
            loc_w: np.ndarray,
            row_range: np.ndarray,
            col_range: np.ndarray,
            tot_num: int
            ):
        kernel_k = row_range[0] - row_range[2] + row_range[3]
        idx = 0

        for k in range(row_range[0], row_range[1] + 1):
            kernel_l = col_range[0] - col_range[2] + col_range[3]
            for l in range(col_range[0], col_range[1] + 1):
                loc_x[idx] = narr_a[k, l]
                loc_y[idx] = narr_b[k, l]
                loc_w[idx] = kernel[kernel_k, kernel_l]
                tau_diff = oldtau[k, l] - oldtau[row_range[2], col_range[2]]
                tau_diff_abs = abs(tau_diff) * oldsqrt_n[row_range[2], col_range[2]]
                tau_diff_abs = tau_diff_abs / self._Dn
                if tau_diff_abs < 1:
                    loc_w[idx] = loc_w[idx] * (1 - tau_diff_abs) * (1 - tau_diff_abs)
                else:
                    loc_w[idx] = loc_w[idx] * 0
                kernel_l += 1
                idx += 1
            kernel_k += 1

        while idx < tot_num:
            loc_x[idx] = 0
            loc_y[idx] = 0
            loc_w[idx] = 0
            idx += 1

    def _generate_kernel(self, int_size: int):
        l = int_size * 2 + 1
        kernel = np.zeros((l, l))
        center = int_size
        r_size = int_size * np.sqrt(2.5)

        for i in range(int_size + 1):
            for j in range(int_size + 1):
                temp = np.sqrt(i * i + j * j) / r_size
                if temp >= 1:
                    temp = 0
                else:
                    temp = 1 - temp
                kernel[center + i, center + j] = temp
                kernel[center - i, center + j] = temp
                kernel[center + i, center - j] = temp
                kernel[center - i, center - j] = temp

        return kernel

    def _get_range(self, location: int, radius: int, boundary: int):
        r = np.zeros(4, dtype=np.int32)
        r[0] = location - radius
        if r[0] < 0:
            r[0] = 0
        r[1] = location + radius
        if r[1] >= boundary:
            r[1] = boundary - 1
        r[2] = location
        r[3] = radius
        
        return r

    def _singleiteration(
        self,
        oldtau: np.ndarray,
        oldsqrt_n: np.ndarray,
        newtau: np.ndarray,
        newsqrt_n: np.ndarray,
        result: np.ndarray,
        int_size: int,
        is_check: np.ndarray
    ):
        kernel = self._generate_kernel(int_size)
        row_range = np.zeros(4, dtype=np.int32)
        col_range = np.zeros(4, dtype=np.int32)
        tot_num = (2 * int_size + 1) * (2 * int_size + 1)
        loc_x = np.zeros(tot_num)
        loc_y = np.zeros(tot_num)
        loc_w = np.zeros(tot_num)
        tau = 0
        tau_diff = 0

        for i in range(self.shape[0]):
            row_range = self._get_range(i, int_size, self.shape[0])
            for j in range(self.shape[1]):
                if is_check:
                    if self._stop[i, j, 0] != 0:
                        continue
                
                col_range = self._get_range(j, int_size, self.shape[1])
                self._get_data(self.narr_a, self.narr_b, kernel, oldtau, oldsqrt_n, loc_x, loc_y, loc_w, row_range, col_range, tot_num)
                newsqrt_n[i, j] = math.sqrt(self._n_tau(loc_x, loc_y, loc_w))

                if newsqrt_n[i, j] <= 0:
                    newtau[i, j] = 0
                    result[i, j] = 0
                else:
                    kendaltau = WtKendallTau(loc_x, loc_y, loc_w)
                    tau = kendaltau.compute()
                    newtau[i, j] = tau
                    result[i, j] = tau * newsqrt_n[i, j] * 1.5

                if is_check:
                    tau_diff = abs(self._stop[i, j, 1] - newtau[i, j]) * self._stop[i, j, 2]
                    if tau_diff > self._lambda:
                        self._stop[i, j, 0] = 1
                        newtau[i, j] = oldtau[i ,j]
                        newsqrt_n[i, j] = oldsqrt_n[i, j]

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                oldtau[i, j] = newtau[i, j]
                oldsqrt_n[i, j] =  newsqrt_n[i, j]

    def _n_tau(self, loc_x: np.ndarray, loc_y: np.ndarray, loc_w: np.ndarray):
        sum_w = 0
        sum_sqrt_w = 0
        temp_w = 0
        nw = 0
        
        for i in range(len(loc_w)):
            if loc_x[i] < self.thre_a or loc_y[i] < self.thre_b:
                loc_w[i] = 0
            temp_w = loc_w[i]
            sum_w += temp_w
            temp_w = temp_w * loc_w[i]
            sum_sqrt_w += temp_w

        denomi = sum_w * sum_w

        if denomi <= 0:
            nw = 0
        else:
            nw = denomi / sum_sqrt_w

        return nw