"""Generate sample_claims.csv mock dataset for testing the fraud detection system."""
import random
import csv
from datetime import date, timedelta

random.seed(42)

hospitals = ['H001', 'H002', 'H003', 'H004', 'H005', 'H006', 'H007']
doctors   = ['D101', 'D102', 'D103', 'D104', 'D105', 'D106', 'D107', 'D108']
diagnosis = ['ICD-J00', 'ICD-K35', 'ICD-I21', 'ICD-C18', 'ICD-S72', 'ICD-N18', 'ICD-E11']
treatment = ['T-SURG01', 'T-PHYS02', 'T-DIAG03', 'T-CHEM04', 'T-ORTHO05', 'T-CARD06']
locations = ['Maharashtra', 'Delhi', 'Tamil Nadu', 'Karnataka', 'Rajasthan', 'UP', 'Gujarat']

def rand_date(start=date(2024,1,1), end=date(2025,1,31)):
    return start + timedelta(days=random.randint(0, (end-start).days))

rows = []
for i in range(1, 301):
    hosp = random.choice(hospitals)
    admit = rand_date()
    disc  = admit + timedelta(days=random.randint(1,15))
    base  = random.uniform(5000, 80000)
    # Inject fraud patterns for ~15% of claims
    fraud = random.random() < 0.15
    if fraud:
        base = base * random.uniform(2.5, 5)  # Inflated amount
    claim_amt    = round(base, 2)
    approved_amt = round(claim_amt * random.uniform(0.5, 1.0), 2)
    # Duplicate patient IDs for some fraud claims
    pid = f'P{random.randint(1000, 1010) if fraud and random.random()<0.5 else random.randint(1000, 9999):04d}'
    rows.append({
        'Claim_ID':       f'CLM{i:05d}',
        'Patient_ID':     pid,
        'Hospital_ID':    hosp,
        'Doctor_ID':      random.choice(doctors),
        'Diagnosis_Code': random.choice(diagnosis),
        'Treatment_Code': random.choice(treatment),
        'Admission_Date': admit.strftime('%Y-%m-%d'),
        'Discharge_Date': disc.strftime('%Y-%m-%d'),
        'Claim_Amount':   claim_amt,
        'Approved_Amount':approved_amt,
        'Location':       random.choice(locations),
    })

with open('sample_claims.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f'Generated {len(rows)} sample claims -> sample_claims.csv')
