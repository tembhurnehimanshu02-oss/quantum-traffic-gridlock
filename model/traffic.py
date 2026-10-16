from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from itertools import product
x = [f'{zq}{bt + 1}' for zq in ('A', 'B') for bt in range(4)]
z = (0, 0, 0, 0, 0, -1, -1)

@dataclass
class TrafficModel:
    scenario: dict

    def __post_init__(self):
        bg = self.scenario
        self.capacity = bg['service_capacity']
        self.storage = bg['storage_capacity']
        self.aa = bg['overflow_weight']
        self.br = bg['switch_weight']
        self.zd = 0 if bg['link']['phase'] == 'NS' else 1
        self.arrivals = tuple((tuple(bg['arrivals'][zq][ae]) for zq in ('A', 'B') for ae in ('NS', 'EW')))
        yk = Decimal(str(bg['link']['fraction']))
        self.transfers = tuple((int((yk * xs).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) for xs in range(self.capacity + 1)))

    def step(self, bt, bo, m, xz):
        ar = list(bo[:4])
        ad, za, zb = bo[4:]
        ay = 0
        for zq, yq in enumerate((m, xz)):
            for ae in range(2):
                zy = 2 * zq + ae
                xy = ar[zy] + self.arrivals[zy][bt]
                if zq == 1 and ae == self.zd:
                    xy += ad
                yy = min(xy, self.capacity) if yq == 1 - ae else 0
                ar[zy] = xy - yy
                if zq == 0 and ae == self.zd and (bt < 3):
                    ay = self.transfers[yy]
        xw = sum(ar)
        az = sum((max(0, al - self.storage) for al in ar))
        bs = int(m != za) + int(xz != zb) if bt else 0
        return ((*ar, ay, m, xz), (xw, az, bs))

    def stage_cost(self, xn):
        xs, zt, bq = xn
        return xs + self.aa * zt + self.br * bq

    def evaluate(self, xe):
        xe = tuple(map(int, xe))
        if len(xe) != 8 or any((xz not in (0, 1) for xz in xe)):
            raise ValueError('Expected eight bits')
        bo = z
        cy = [0, 0, 0]
        for bt in range(4):
            bo, xn = self.step(bt, bo, xe[bt], xe[bt + 4])
            cy = [cg + ch for cg, ch in zip(cy, xn)]
        return {'cost': round(self.stage_cost(cy), 6), 'delay': cy[0], 'overflow': cy[1], 'switches': cy[2], 'final_queues': {'A': {'NS': bo[0], 'EW': bo[1]}, 'B': {'NS': bo[2], 'EW': bo[3]}}}

    def expected_cost(self, yh=None):
        yh = {} if yh is None else dict(yh)

        @lru_cache(maxsize=None)
        def value(bt, bo):
            if bt == 4:
                return 0.0
            xk = (yh[bt],) if bt in yh else (0, 1)
            xl = (yh[bt + 4],) if bt + 4 in yh else (0, 1)
            cx = 0.0
            for m, xz in product(xk, xl):
                yj, xn = self.step(bt, bo, m, xz)
                cx += self.stage_cost(xn) + value(bt + 1, yj)
            return cx / (len(xk) * len(xl))
        return value(0, z)

    def greedy(self):
        bo = z
        af = [[], []]
        for bt in range(4):
            n = []
            for zq in range(2):
                xy = [bo[2 * zq + ab] + self.arrivals[2 * zq + ab][bt] for ab in range(2)]
                if zq == 1:
                    xy[self.zd] += bo[4]
                n.append(int(xy[0] >= xy[1]))
                af[zq].append(n[-1])
            bo, l = self.step(bt, bo, *n)
        return ''.join(map(str, af[0] + af[1]))
