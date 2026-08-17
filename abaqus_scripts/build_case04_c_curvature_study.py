# Abaqus/CAE noGUI script.
# Case 4: curvature study for the composite C-shell flattening benchmark.
#
# The material, laminate, arc length, width, thickness and final flattened
# configuration are identical to Case 2.  Only the initial included angle is
# varied, so the resulting data isolate the influence of initial curvature.
# Default new angles are 120, 170 and 260 degrees; Case 2 already supplies the
# 220-degree baseline.
from abaqus import *
from abaqusConstants import *
import math
import mesh
import os
import sys
import glob
from section import SectionLayer


SCRIPT_DIR = r'F:/Projects/PINNShell/fem/abaqus_scripts'
LENGTH = 20.0
WIDTH = 7.0
THICKNESS = 0.04
ANGLES_DEG = (120.0, 170.0, 260.0)
NX = 160
NZ = 24
MID_INDEX = NX // 2
SUBMIT_JOBS = True
JOB_EXTENSIONS = (
    '.cae', '.inp', '.odb', '.lck', '.dat', '.msg', '.sta', '.com',
    '.prt', '.sim', '.stt', '.mdl', '.res', '.pac', '.abq', '.sel',
)

# Typical unidirectional carbon/epoxy engineering constants in MPa.
CFRP_E1 = 135000.0
CFRP_E2 = 10000.0
CFRP_E3 = 10000.0
CFRP_NU12 = 0.30
CFRP_NU13 = 0.30
CFRP_NU23 = 0.45
CFRP_G12 = 5000.0
CFRP_G13 = 5000.0
CFRP_G23 = 3800.0
PLY_ANGLES = (0.0, 90.0, 90.0, 0.0)
BASE_FIELD_OUTPUT = ('S', 'LE', 'U', 'UR', 'RF', 'RM', 'COORD')
OPTIONAL_FIELD_OUTPUT = ('E', 'SF', 'SM', 'SE', 'SK', 'SENER')
ENERGY_HISTORY_OUTPUT = ('ALLIE', 'ALLSE', 'ALLWK', 'ALLAE', 'ALLSD', 'ETOTAL')


def _script_arguments():
    """Read arguments placed after ``--`` in an Abaqus command line."""
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    options = {}
    i = 0
    while i < len(args):
        key = args[i]
        if key == '--no-submit':
            options['submit'] = False
            i += 1
        elif key in ('--angles', '--nx', '--nz') and i + 1 < len(args):
            options[key[2:]] = args[i + 1]
            i += 2
        else:
            raise ValueError('Unknown or incomplete argument: {}'.format(key))
    return options


def configure_field_output(model):
    accepted = list(BASE_FIELD_OUTPUT)
    model.fieldOutputRequests['F-Output-1'].setValues(variables=tuple(accepted))
    for variable in OPTIONAL_FIELD_OUTPUT:
        try:
            model.fieldOutputRequests['F-Output-1'].setValues(variables=tuple(accepted + [variable]))
            accepted.append(variable)
        except Exception as exc:
            print('Skipped optional field output {}: {}'.format(variable, exc))
    print('Final field output variables: {}'.format(tuple(accepted)))


def configure_history(model, name, step_name, variables, region=None):
    kwargs = dict(name=name, createStepName=step_name, variables=variables)
    if region is not None:
        kwargs['region'] = region
    try:
        model.HistoryOutputRequest(**kwargs)
    except Exception as exc:
        print('Skipped history output {}: {}'.format(name, exc))


def c_centerline(angle_deg):
    alpha = math.radians(angle_deg)
    radius = LENGTH / alpha
    phi0 = -0.5 * alpha
    points = []
    for i in range(NX + 1):
        s = LENGTH * float(i) / float(NX)
        phi = phi0 + alpha * s / LENGTH
        x = radius * (math.sin(phi) - math.sin(phi0))
        y = -radius * (math.cos(phi) - math.cos(phi0))
        points.append((x, y))
    return points


def make_strip_part(model, centerline):
    part = model.Part(name='C_SHELL', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    nodes = {}
    for i, point in enumerate(centerline):
        for j in range(NZ + 1):
            eta = (float(j) / float(NZ) - 0.5) * WIDTH
            label = i * (NZ + 1) + j + 1
            nodes[label] = part.Node(coordinates=(point[0], point[1], eta), label=label)
    element_labels = []
    label = 1
    for i in range(NX):
        for j in range(NZ):
            n1 = i * (NZ + 1) + j + 1
            n2 = (i + 1) * (NZ + 1) + j + 1
            n3 = (i + 1) * (NZ + 1) + j + 2
            n4 = i * (NZ + 1) + j + 2
            part.Element(nodes=(nodes[n1], nodes[n2], nodes[n3], nodes[n4]), elemShape=QUAD4, label=label)
            element_labels.append(label)
            label += 1
    part.SetFromElementLabels(name='ALL_ELEMENTS', elementLabels=tuple(element_labels))
    return part


def edge_labels(index):
    return tuple(index * (NZ + 1) + j + 1 for j in range(NZ + 1))


def add_material_and_section(model):
    material = model.Material(name='CFRP_UD')
    material.Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((CFRP_E1, CFRP_E2, CFRP_E3, CFRP_NU12, CFRP_NU13,
                CFRP_NU23, CFRP_G12, CFRP_G13, CFRP_G23),),
    )
    ply_thickness = THICKNESS / float(len(PLY_ANGLES))
    layup = tuple(
        SectionLayer(
            material='CFRP_UD', thickness=ply_thickness, orientAngle=angle,
            numIntPts=3, plyName='Ply-{}_{:g}'.format(i + 1, angle),
        )
        for i, angle in enumerate(PLY_ANGLES)
    )
    model.CompositeShellSection(
        name='ShellSection', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, symmetric=False,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, useDensity=OFF,
        integrationRule=SIMPSON, layup=layup,
    )


def export_results(job, case_id):
    if job.status != COMPLETED:
        print('Job {} ended with status {}; export skipped.'.format(job.name, job.status))
        return
    odb_path = os.path.abspath(job.name + '.odb')
    session.openOdb(name=odb_path)
    result_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results', case_id))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    old_cwd = os.getcwd()
    os.chdir(result_dir)
    try:
        # Do not execute the exporter in this builder's global namespace.
        # ``datasavebystep.py`` imports every Abaqus symbolic constant; one of
        # them is named THICKNESS and would replace the numeric shell thickness
        # after the first case in a multi-case sweep.  A private namespace also
        # prevents any future exporter variable from changing the next model.
        export_namespace = {
            '__name__': '__abaqus_result_export__',
            '__file__': os.path.join(SCRIPT_DIR, 'datasavebystep.py'),
            'ODB_PATH_TO_EXPORT': odb_path,
            'CASE_ID': case_id,
        }
        execfile(export_namespace['__file__'], export_namespace)
    finally:
        os.chdir(old_cwd)


def existing_case_artifacts(case_id):
    """Return every path that would be overwritten by this case.

    Abaqus 2020 and the shared exporter use ordinary files rather than an
    atomic output transaction.  Refuse to run if either a job artifact or the
    dedicated export directory already exists.
    """
    existing = []
    for extension in JOB_EXTENSIONS:
        path = os.path.abspath(case_id + extension)
        if os.path.exists(path):
            existing.append(path)
    result_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results', case_id))
    if os.path.exists(result_dir):
        existing.append(result_dir)
    return existing


def completed_export(case_id):
    """Return True only when the shared exporter finished this case.

    A directory alone is not sufficient: an interrupted exporter may leave a
    partial set of frame files.  The history file is written last, and at least
    one shell-resultant file must also exist for a completed shell analysis.
    """
    result_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results', case_id))
    history_path = os.path.join(result_dir, case_id + '_history.csv')
    shell_pattern = os.path.join(result_dir, case_id + '_*_shell_resultants.csv')
    return os.path.isfile(history_path) and len(glob.glob(shell_pattern)) > 0


def assert_new_case(case_id):
    existing = existing_case_artifacts(case_id)
    if existing:
        message = [
            'Refusing to overwrite existing Abaqus data for {}.'.format(case_id),
            'Move or rename these paths explicitly before rerunning:',
        ]
        message.extend(['  ' + path for path in existing])
        raise RuntimeError('\n'.join(message))


def build_one(angle_deg, submit_job=True):
    angle_tag = int(round(angle_deg))
    case_id = 'case04_c_angle_{:03d}'.format(angle_tag)
    if case_id in mdb.models.keys():
        del mdb.models[case_id]
    model = mdb.Model(name=case_id)
    add_material_and_section(model)

    centerline = c_centerline(angle_deg)
    part = make_strip_part(model, centerline)
    part.SectionAssignment(region=part.sets['ALL_ELEMENTS'], sectionName='ShellSection')
    part.SetFromNodeLabels(name='LEFT_EDGE', nodeLabels=edge_labels(0))
    part.SetFromNodeLabels(name='MID_EDGE', nodeLabels=edge_labels(MID_INDEX))
    part.SetFromNodeLabels(name='RIGHT_EDGE', nodeLabels=edge_labels(NX))
    part.setElementType(
        regions=(part.sets['ALL_ELEMENTS'].elements,),
        elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),),
    )

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    instance = assembly.Instance(name='C_SHELL-1', part=part, dependent=ON)
    assembly.Set(name='LEFT_EDGE', nodes=instance.nodes.sequenceFromLabels(edge_labels(0)))
    assembly.Set(name='MID_EDGE', nodes=instance.nodes.sequenceFromLabels(edge_labels(MID_INDEX)))
    assembly.Set(name='RIGHT_EDGE', nodes=instance.nodes.sequenceFromLabels(edge_labels(NX)))

    model.StaticStep(
        name='FlattenStep', previous='Initial', nlgeom=ON,
        initialInc=1.0e-4, maxInc=0.005, minInc=1.0e-12,
        maxNumInc=10000,
        stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        stabilizationMagnitude=2.0e-4,
    )
    configure_field_output(model)
    configure_history(model, 'H_MODEL_ENERGY', 'FlattenStep', ENERGY_HISTORY_OUTPUT)
    edge_history = ('U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3',
                    'RF1', 'RF2', 'RF3', 'RM1', 'RM2', 'RM3')
    for edge_name in ('LEFT_EDGE', 'MID_EDGE', 'RIGHT_EDGE'):
        configure_history(model, 'H_' + edge_name, 'FlattenStep', edge_history, assembly.sets[edge_name])

    model.EncastreBC(name='BC_MID_FIXED', createStepName='Initial', region=assembly.sets['MID_EDGE'])
    mid_x, mid_y = centerline[MID_INDEX]
    target_left_x = mid_x - 0.5 * LENGTH
    target_right_x = mid_x + 0.5 * LENGTH
    left_dx = target_left_x - centerline[0][0]
    left_dy = mid_y - centerline[0][1]
    right_dx = target_right_x - centerline[-1][0]
    right_dy = mid_y - centerline[-1][1]
    model.DisplacementBC(
        name='BC_LEFT_PULL', createStepName='FlattenStep', region=assembly.sets['LEFT_EDGE'],
        u1=left_dx, u2=left_dy, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET,
    )
    model.DisplacementBC(
        name='BC_RIGHT_PULL', createStepName='FlattenStep', region=assembly.sets['RIGHT_EDGE'],
        u1=right_dx, u2=right_dy, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET,
    )

    # Record the exact boundary data required by the corresponding PINN case.
    print('{}: angle={:.3f}, radius={:.6f}'.format(case_id, angle_deg, LENGTH / math.radians(angle_deg)))
    print('  left displacement = ({:.10f}, {:.10f}, 0)'.format(left_dx, left_dy))
    print('  right displacement = ({:.10f}, {:.10f}, 0)'.format(right_dx, right_dy))

    job = mdb.Job(name=case_id, model=case_id, numCpus=1, numDomains=1)
    job.writeInput()
    mdb.saveAs(pathName=case_id + '.cae')
    if submit_job:
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()
        export_results(job, case_id)
    return case_id


def main():
    global ANGLES_DEG, NX, NZ, MID_INDEX
    options = _script_arguments()
    if 'angles' in options:
        ANGLES_DEG = tuple(float(value) for value in options['angles'].split(','))
    if 'nx' in options:
        NX = int(options['nx'])
    if 'nz' in options:
        NZ = int(options['nz'])
    MID_INDEX = NX // 2
    submit = options.get('submit', SUBMIT_JOBS)
    print('Case 4 curvature study: angles={}, mesh={}x{}, submit={}'.format(ANGLES_DEG, NX, NZ, submit))
    # A direct CAE ``execfile`` call cannot conveniently pass command-line
    # arguments.  Safely skip cases whose exporter demonstrably completed, but
    # still reject partial directories or same-named job artifacts.
    case_ids = ['case04_c_angle_{:03d}'.format(int(round(value))) for value in ANGLES_DEG]
    pending = []
    for angle_deg, case_id in zip(ANGLES_DEG, case_ids):
        if completed_export(case_id):
            print('Skipping completed case without overwrite: {}'.format(case_id))
        else:
            pending.append((angle_deg, case_id))
    # Preflight every pending case before constructing the first one.
    for angle_deg, case_id in pending:
        assert_new_case(case_id)
    for angle_deg, case_id in pending:
        build_one(angle_deg, submit_job=submit)
    if not pending:
        print('All requested Case 4 exports already exist; nothing was changed.')


main()
