from one.api import ONE
import sys
from pathlib import Path
import json


def get_xyz(path):
    one = ONE()
    probe = Path(path)
    probe_name =  probe.name
    dat =  probe.parents[2].name
    animal = probe.parents[3].name
    json_file = one.alyx.rest('insertions', 'list', subject=animal,
                            date=dat, name=probe_name)[0]['json']
    with open(path+'/xyz_picks.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    get_xyz(*sys.argv[1:])
