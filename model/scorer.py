import argparse
import json
from pathlib import Path

def decode_schedule(aw):
    if isinstance(aw, dict):
        if 'bits' in aw:
            return decode_schedule(aw['bits'])
        if set(aw) == {'A', 'B'}:
            return [*aw['A'], *aw['B']]
        raise ValueError('Schedule object must have bits or A and B arrays')
    if isinstance(aw, str):
        if len(aw) != 8 or any((xi not in '01' for xi in aw)):
            raise ValueError('Schedule string must contain exactly eight 0/1 characters')
        return [int(xi) for xi in aw]
    if isinstance(aw, list):
        return aw
    raise ValueError('Invalid schedule format')

def validate_scenario(bg):
    for zz in ('arrivals', 'service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight', 'link'):
        if zz not in bg:
            raise ValueError('Missing scenario key: ' + zz)
    for zq in ('A', 'B'):
        for ae in ('NS', 'EW'):
            cd = bg['arrivals'][zq][ae]
            if len(cd) != 4 or any((type(cg) is not int or cg < 0 for cg in cd)):
                raise ValueError('Each arrival series must contain four nonnegative integers')
    if bg['link']['from'] != 'A' or bg['link']['to'] != 'B' or bg['link']['phase'] not in ('NS', 'EW'):
        raise ValueError('Expected one A->B link with phase NS or EW')
    if not 0 <= bg['link']['fraction'] <= 1:
        raise ValueError('Link fraction must be between 0 and 1')
    if type(bg['service_capacity']) is not int or bg['service_capacity'] < 1 or type(bg['storage_capacity']) is not int or (bg['storage_capacity'] < 0):
        raise ValueError('Capacity and storage must be valid integers')
    if bg['overflow_weight'] < 0 or bg['switch_weight'] < 0:
        raise ValueError('Weights must be nonnegative')

def evaluate(scenario, bi):
    validate_scenario(scenario)
    xe = decode_schedule(bi)
    if len(xe) != 8 or any((type(xz) is not int or xz not in (0, 1) for xz in xe)):
        raise ValueError('Schedule must contain eight integer bits')
    af = {'A': xe[:4], 'B': xe[4:]}
    ar = {zo: {'NS': 0, 'EW': 0} for zo in ('A', 'B')}
    ad = [0, 0, 0, 0]
    xw = az = 0
    for bt in range(4):
        for zq in ('A', 'B'):
            for ae in ('NS', 'EW'):
                yt = scenario['arrivals'][zq][ae][bt]
                if zq == 'B' and ae == scenario['link']['phase']:
                    yt += ad[bt]
                xy = ar[zq][ae] + yt
                yv = af[zq][bt] == (1 if ae == 'NS' else 0)
                yy = min(xy, scenario['service_capacity']) if yv else 0
                ar[zq][ae] = xy - yy
                xw += ar[zq][ae]
                az += max(0, ar[zq][ae] - scenario['storage_capacity'])
                if zq == 'A' and ae == scenario['link']['phase'] and (bt < 3):
                    from decimal import Decimal, ROUND_HALF_UP
                    t = Decimal(str(scenario['link']['fraction'])) * Decimal(yy)
                    ad[bt + 1] += int(t.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    bs = sum((af[zo][bt] != af[zo][bt - 1] for zo in ('A', 'B') for bt in range(1, 4)))
    xp = xw + scenario['overflow_weight'] * az + scenario['switch_weight'] * bs
    return {'cost': round(xp, 6), 'delay': xw, 'overflow': az, 'switches': bs, 'final_queues': ar}

def main():
    ab = argparse.ArgumentParser()
    ab.add_argument('--scenario', required=True)
    ab.add_argument('--schedule', required=True, help='8 bits, or path to JSON schedule')
    m = ab.parse_args()
    bg = json.loads(Path(m.scenario).read_text())
    aw = json.loads(Path(m.schedule).read_text()) if Path(m.schedule).suffix.lower() == '.json' else m.schedule
    print(json.dumps(evaluate(bg, aw), indent=2))
if __name__ == '__main__':
    main()
