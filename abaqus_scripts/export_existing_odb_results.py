from abaqus import session
import os


SCRIPT_DIR = r'F:/Projects/PINNShell/fem/abaqus_scripts'
RESULT_ROOT = r'F:/Projects/PINNShell/fem/results'
ODB_ROOT = r'E:/ABAQUS/temp/shell'

CASES = (
    'case01_flat_bending',
    'case02_c_flattening',
    'case03_omega_flattening',
)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


for case_id in CASES:
    odb_path = os.path.join(ODB_ROOT, case_id + '.odb')
    result_dir = os.path.join(RESULT_ROOT, case_id)
    if not os.path.exists(odb_path):
        print('Skip missing ODB: {}'.format(odb_path))
        continue
    ensure_dir(result_dir)
    os.chdir(result_dir)
    ODB_PATH_TO_EXPORT = odb_path
    print('Re-exporting {}'.format(odb_path))
    execfile(os.path.join(SCRIPT_DIR, 'datasavebystep.py'), globals())

print('Done.')
