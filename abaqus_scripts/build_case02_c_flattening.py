# Abaqus/CAE noGUI script.
# Case 2: C-shaped shell strip. The middle section is fixed; both ends are
# pulled apart simultaneously to flatten the strip.
from abaqus import *
from abaqusConstants import *
import math
import mesh
import os
import sys
from section import SectionLayer


SCRIPT_DIR = r'F:/Projects/PINNShell/fem/abaqus_scripts'
CASE_ID = 'case02_c_flattening'
LENGTH = 20.0
ANGLE_DEG = 220.0
WIDTH = 7.0
THICKNESS = 0.04
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
PLY_THICKNESS = THICKNESS / 4.0
PLY_ANGLES = (0.0, 90.0, 90.0, 0.0)
NX = 160
NZ = 24
MID_INDEX = NX // 2
BASE_FIELD_OUTPUT = ('S', 'LE', 'U', 'UR', 'RF', 'RM', 'COORD')
OPTIONAL_FIELD_OUTPUT = ('E', 'SF', 'SM', 'SE', 'SK', 'SENER')
ENERGY_HISTORY_OUTPUT = ('ALLIE', 'ALLSE', 'ALLWK', 'ALLAE', 'ALLSD', 'ETOTAL')


def configure_field_output(model):
    accepted = list(BASE_FIELD_OUTPUT)
    model.fieldOutputRequests['F-Output-1'].setValues(variables=tuple(accepted))
    for variable in OPTIONAL_FIELD_OUTPUT:
        trial = accepted + [variable]
        try:
            model.fieldOutputRequests['F-Output-1'].setValues(variables=tuple(trial))
            accepted.append(variable)
            print('Accepted optional field output: {}'.format(variable))
        except Exception as exc:
            print('Skipped optional field output {}: {}'.format(variable, exc))
    print('Final field output variables: {}'.format(tuple(accepted)))
    return tuple(accepted)


def configure_energy_history(model, step_name):
    try:
        model.HistoryOutputRequest(
            name='H_MODEL_ENERGY',
            createStepName=step_name,
            variables=ENERGY_HISTORY_OUTPUT,
        )
        print('Energy history output variables: {}'.format(ENERGY_HISTORY_OUTPUT))
    except Exception as exc:
        print('Skipped model energy history output: {}'.format(exc))


def configure_region_history(model, name, step_name, variables, region):
    try:
        model.HistoryOutputRequest(
            name=name,
            createStepName=step_name,
            variables=variables,
            region=region,
        )
        print('History output {} variables: {}'.format(name, variables))
    except Exception as exc:
        print('Skipped history output {}: {}'.format(name, exc))


def c_centerline():
    alpha = math.radians(ANGLE_DEG)
    radius = LENGTH / alpha
    phi0 = -0.5 * alpha
    pts = []
    for i in range(NX + 1):
        s = LENGTH * float(i) / float(NX)
        phi = phi0 + alpha * s / LENGTH
        x = radius * (math.sin(phi) - math.sin(phi0))
        y = -radius * (math.cos(phi) - math.cos(phi0))
        pts.append((x, y))
    return pts


def make_strip_part(model, name, centerline):
    part = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    node_by_label = {}
    for i, p in enumerate(centerline):
        for j in range(NZ + 1):
            a = (float(j) / float(NZ) - 0.5) * WIDTH
            label = i * (NZ + 1) + j + 1
            node_by_label[label] = part.Node(coordinates=(p[0], p[1], a), label=label)
    elem_labels = []
    label = 1
    for i in range(len(centerline) - 1):
        for j in range(NZ):
            n1 = i * (NZ + 1) + j + 1
            n2 = (i + 1) * (NZ + 1) + j + 1
            n3 = (i + 1) * (NZ + 1) + j + 2
            n4 = i * (NZ + 1) + j + 2
            # Connectivity n1-n2-n3-n4 makes the shell local 1 direction
            # follow the centreline and local 2 follow the strip width.
            elem = part.Element(
                nodes=(node_by_label[n1], node_by_label[n2], node_by_label[n3], node_by_label[n4]),
                elemShape=QUAD4,
                label=label,
            )
            elem_labels.append(label)
            label += 1
    part.SetFromElementLabels(name='ALL_ELEMENTS', elementLabels=tuple(elem_labels))
    return part


def edge_labels(i):
    return tuple(i * (NZ + 1) + j + 1 for j in range(NZ + 1))


def shell_elements(part):
    return part.sets['ALL_ELEMENTS'].elements


def save_results_if_complete(job):
    if job.status != COMPLETED:
        print('Job {} ended with status {}; skip ODB export.'.format(job.name, job.status))
        return
    global ODB_PATH_TO_EXPORT
    odb_path = os.path.abspath(job.name + '.odb')
    ODB_PATH_TO_EXPORT = odb_path
    session.openOdb(name=odb_path)
    result_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results', CASE_ID))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    old_cwd = os.getcwd()
    os.chdir(result_dir)
    try:
        execfile(os.path.join(SCRIPT_DIR, 'datasavebystep.py'), globals())
    finally:
        os.chdir(old_cwd)


def build():
    if CASE_ID in mdb.models.keys():
        del mdb.models[CASE_ID]
    model = mdb.Model(name=CASE_ID)
    mat = model.Material(name='CFRP_UD')
    mat.Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((
            CFRP_E1,
            CFRP_E2,
            CFRP_E3,
            CFRP_NU12,
            CFRP_NU13,
            CFRP_NU23,
            CFRP_G12,
            CFRP_G13,
            CFRP_G23,
        ),),
    )
    layup = tuple(
        SectionLayer(
            material='CFRP_UD',
            thickness=PLY_THICKNESS,
            orientAngle=angle,
            numIntPts=3,
            plyName='Ply-{}_{:g}'.format(i + 1, angle),
        )
        for i, angle in enumerate(PLY_ANGLES)
    )
    model.CompositeShellSection(
        name='ShellSection',
        preIntegrate=OFF,
        idealization=NO_IDEALIZATION,
        symmetric=False,
        thicknessType=UNIFORM,
        poissonDefinition=DEFAULT,
        temperature=GRADIENT,
        useDensity=OFF,
        integrationRule=SIMPSON,
        layup=layup,
    )
    centerline = c_centerline()
    part = make_strip_part(model, 'C_SHELL', centerline)
    part.SectionAssignment(region=part.sets['ALL_ELEMENTS'], sectionName='ShellSection')
    part.SetFromNodeLabels(name='LEFT_EDGE', nodeLabels=edge_labels(0))
    part.SetFromNodeLabels(name='MID_EDGE', nodeLabels=edge_labels(MID_INDEX))
    part.SetFromNodeLabels(name='RIGHT_EDGE', nodeLabels=edge_labels(NX))
    part.setElementType(regions=(shell_elements(part),), elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),))

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    inst = asm.Instance(name='C_SHELL-1', part=part, dependent=ON)
    asm.Set(name='LEFT_EDGE', nodes=inst.nodes.sequenceFromLabels(edge_labels(0)))
    asm.Set(name='MID_EDGE', nodes=inst.nodes.sequenceFromLabels(edge_labels(MID_INDEX)))
    asm.Set(name='RIGHT_EDGE', nodes=inst.nodes.sequenceFromLabels(edge_labels(NX)))

    model.StaticStep(
        name='FlattenStep',
        previous='Initial',
        nlgeom=ON,
        initialInc=1.0e-4,
        maxInc=0.005,
        minInc=1.0e-12,
        maxNumInc=10000,
        stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        stabilizationMagnitude=2.0e-4,
    )
    configure_field_output(model)
    configure_energy_history(model, 'FlattenStep')
    edge_history = ('U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3', 'RF1', 'RF2', 'RF3', 'RM1', 'RM2', 'RM3')
    configure_region_history(model, 'H_LEFT_EDGE', 'FlattenStep', edge_history, asm.sets['LEFT_EDGE'])
    configure_region_history(model, 'H_RIGHT_EDGE', 'FlattenStep', edge_history, asm.sets['RIGHT_EDGE'])
    configure_region_history(model, 'H_MID_EDGE', 'FlattenStep', edge_history, asm.sets['MID_EDGE'])
    model.EncastreBC(name='BC_MID_FIXED', createStepName='Initial', region=asm.sets['MID_EDGE'])
    mid_x, mid_y = centerline[MID_INDEX]
    target_left_x = mid_x - 0.5 * LENGTH
    target_right_x = mid_x + 0.5 * LENGTH
    target_y = mid_y
    left_dx = target_left_x - centerline[0][0]
    left_dy = target_y - centerline[0][1]
    right_dx = target_right_x - centerline[-1][0]
    right_dy = target_y - centerline[-1][1]
    model.DisplacementBC(
        name='BC_LEFT_PULL',
        createStepName='FlattenStep',
        region=asm.sets['LEFT_EDGE'],
        u1=left_dx, u2=left_dy, u3=0.0,
        ur1=UNSET, ur2=UNSET, ur3=UNSET,
    )
    model.DisplacementBC(
        name='BC_RIGHT_PULL',
        createStepName='FlattenStep',
        region=asm.sets['RIGHT_EDGE'],
        u1=right_dx, u2=right_dy, u3=0.0,
        ur1=UNSET, ur2=UNSET, ur3=UNSET,
    )
    job = mdb.Job(name=CASE_ID, model=CASE_ID, numCpus=1, numDomains=1)
    job.writeInput()
    mdb.saveAs(pathName=CASE_ID + '.cae')
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    save_results_if_complete(job)


build()
