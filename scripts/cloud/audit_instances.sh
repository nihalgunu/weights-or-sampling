#!/bin/bash
# Lists every live Lambda instance with its age, which job file claims it, and
# whether a local driver is attached. Run at session start and before/after any
# cloud action. Flags anything older than MAX_H hours or with no driver.
# Usage: bash scripts/cloud/audit_instances.sh [MAX_H=24]
set -u
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
MAX_H="${1:-24}"
NOW=$(date +%s)
curl -s --max-time 30 -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances \
| python3 -c "
import json, sys, subprocess, glob, os, datetime
d = json.load(sys.stdin).get('data', [])
now = $NOW; max_h = $MAX_H
claims = {}
for f in glob.glob('.lambda_instance_g_*'):
    claims[open(f).read().strip()] = f.replace('.lambda_instance_g_', '')
drivers = subprocess.run(['pgrep', '-fl', 'babysit_generic.sh'], capture_output=True, text=True).stdout
print(f'{len(d)} live instance(s)')
bad = 0
for i in d:
    fh = i.get('first_healthy') or ''
    age_h = (now - datetime.datetime.fromisoformat(fh.replace('Z', '+00:00')).timestamp()) / 3600 if fh else float('nan')
    name = claims.get(i['id'], 'UNCLAIMED')
    has_driver = name != 'UNCLAIMED' and (f'babysit_generic.sh {name} ' in drivers or drivers.rstrip().endswith(f'babysit_generic.sh {name}'))
    flag = ''
    if name == 'UNCLAIMED': flag += ' !! no job file'
    if not has_driver: flag += ' !! no driver'
    if age_h == age_h and age_h > max_h: flag += f' !! older than {max_h}h'
    if flag: bad += 1
    rate = i['instance_type']['price_cents_per_hour'] / 100
    print(f\"  {i['id'][:8]} {i['instance_type']['name']:20s} {i['status']:9s} age {age_h:5.1f}h  \${rate:.2f}/h  job={name}{flag}\")
print('OK: every box is claimed and driven' if not bad else f'ATTENTION: {bad} box(es) need a look')
"
