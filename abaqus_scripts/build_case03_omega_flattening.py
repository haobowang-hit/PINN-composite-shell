# Abaqus/CAE noGUI script.
# Case 3: Omega-shaped shell strip flattened between two rigid plates.
from abaqus import *
from abaqusConstants import *
import math
import mesh
import os
import sys
from section import SectionLayer


SCRIPT_DIR = r'F:/Projects/PINNShell/fem/abaqus_scripts'
CASE_ID = 'case03_omega_flattening'
RADIUS = 5.0
LEG_LENGTH = 2.0
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
TOP_INITIAL_Y = RADIUS + 1.0
TOP_FINAL_Y = 1.0
NX = 180
NZ = 24
PLATE_HALF_X = 15.0
PLATE_HALF_Z = 5.0
BASE_FIELD_OUTPUT = ('S', 'LE', 'U', 'UR', 'RF', 'RM', 'COORD')
OPTIONAL_FIELD_OUTPUT = ('E', 'SF', 'SM', 'SE', 'SK', 'SENER', 'CSTRESS', 'CDISP')
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


def append_line(points, p0, p1, n):
    start = 0
    if points:
        start = 1
    for i in range(start, n + 1):
        t = float(i) / float(n)
        points.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))


def append_arc(points, center, radius, angle0, angle1, n):
    start = 0
    if points:
        start = 1
    for i in range(start, n + 1):
        t = float(i) / float(n)
        angle = angle0 + t * (angle1 - angle0)
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))


def omega_centerline():
    a = RADIUS * math.sin(60.0 * math.pi / 180.0)
    line_n = 14
    side_arc_n = 36
    center_arc_n = NX - 2 * line_n - 2 * side_arc_n
    pts = []
    append_line(pts, (-2.0 * a - LEG_LENGTH, 0.0), (-2.0 * a, 0.0), line_n)
    append_arc(pts, (-2.0 * a, RADIUS), RADIUS, -0.5 * math.pi, -math.pi / 6.0, side_arc_n)
    append_arc(pts, (0.0, 0.0), RADIUS, 5.0 * math.pi / 6.0, math.pi / 6.0, center_arc_n)
    append_arc(pts, (2.0 * a, RADIUS), RADIUS, -5.0 * math.pi / 6.0, -0.5 * math.pi, side_arc_n)
    append_line(pts, (2.0 * a, 0.0), (2.0 * a + LEG_LENGTH, 0.0), line_n)
    return pts


def make_shell_strip(model, name, centerline):
    part = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    node_by_label = {}
    for i, p in enumerate(centerline):
        for j in range(NZ + 1):
            z = (float(j) / float(NZ) - 0.5) * WIDTH
            label = i * (NZ + 1) + j + 1
            node_by_label[label] = part.Node(coordinates=(p[0], p[1], z), label=label)
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


def make_rigid_plate(model, name, y):
    part = model.Part(name=name, dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
    n1 = part.Node(coordinates=(-PLATE_HALF_X, y, -PLATE_HALF_Z), label=1)
    n2 = part.Node(coordinates=(PLATE_HALF_X, y, -PLATE_HALF_Z), label=2)
    n3 = part.Node(coordinates=(PLATE_HALF_X, y, PLATE_HALF_Z), label=3)
    n4 = part.Node(coordinates=(-PLATE_HALF_X, y, PLATE_HALF_Z), label=4)
    part.SetFromNodeLabels(name='PLATE_NODES', nodeLabels=(1, 2, 3, 4))
    part.Element(nodes=(n1, n2, n3, n4), elemShape=QUAD4, label=1)
    part.SetFromElementLabels(name='PLATE_SURF', elementLabels=(1,))
    part.ReferencePoint(point=(0.0, y, 0.0))
    rp_key = part.referencePoints.keys()[-1]
    part.Set(name='RP', referencePoints=(part.referencePoints[rp_key],))
    part.Surface(name='CONTACT_SURF', side1Elements=part.sets['PLATE_SURF'].elements)
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
    centerline = omega_centerline()
    strip = make_shell_strip(model, 'OMEGA_SHELL', centerline)
    strip.SectionAssignment(region=strip.sets['ALL_ELEMENTS'], sectionName='ShellSection')
    strip.SetFromNodeLabels(name='LEFT_EDGE', nodeLabels=edge_labels(0))
    strip.SetFromNodeLabels(name='RIGHT_EDGE', nodeLabels=edge_labels(NX))
    strip.Surface(name='SHELL_SURF', side1Elements=shell_elements(strip))
    strip.setElementType(regions=(shell_elements(strip),), elemTypes=(mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),))
    bottom = make_rigid_plate(model, 'BOTTOM_PLATE', -0.2)
    top = make_rigid_plate(model, 'TOP_PLATE', TOP_INITIAL_Y)

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    strip_i = asm.Instance(name='OMEGA_SHELL-1', part=strip, dependent=ON)
    bottom_i = asm.Instance(name='BOTTOM_PLATE-1', part=bottom, dependent=ON)
    top_i = asm.Instance(name='TOP_PLATE-1', part=top, dependent=ON)
    asm.Set(name='LEFT_EDGE', nodes=strip_i.nodes.sequenceFromLabels(edge_labels(0)))
    asm.Set(name='RIGHT_EDGE', nodes=strip_i.nodes.sequenceFromLabels(edge_labels(NX)))
    bottom_rp_key = bottom_i.referencePoints.keys()[-1]
    top_rp_key = top_i.referencePoints.keys()[-1]
    asm.Set(name='BOTTOM_RP', referencePoints=(bottom_i.referencePoints[bottom_rp_key],))
    asm.Set(name='TOP_RP', referencePoints=(top_i.referencePoints[top_rp_key],))

    model.RigidBody(name='RB_BOTTOM', refPointRegion=asm.sets['BOTTOM_RP'], bodyRegion=bottom_i.sets['PLATE_SURF'])
    model.RigidBody(name='RB_TOP', refPointRegion=asm.sets['TOP_RP'], bodyRegion=top_i.sets['PLATE_SURF'])
    model.ContactProperty('FrictionlessContact')
    model.interactionProperties['FrictionlessContact'].TangentialBehavior(formulation=FRICTIONLESS)
    model.interactionProperties['FrictionlessContact'].NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON)
    model.ContactStd(name='GeneralContact', createStepName='Initial')
    model.interactions['GeneralContact'].includedPairs.setValuesInStep(stepName='Initial', useAllstar=ON)
    model.interactions['GeneralContact'].contactPropertyAssignments.appendInStep(
        stepName='Initial',
        assignments=((GLOBAL, SELF, 'FrictionlessContact'),),
    )

    model.StaticStep(
        name='CompressStep',
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
    configure_energy_history(model, 'CompressStep')
    configure_region_history(model, 'H_TOP_RP', 'CompressStep', ('U2', 'RF2'), asm.sets['TOP_RP'])
    configure_region_history(model, 'H_BOTTOM_RP', 'CompressStep', ('U2', 'RF2'), asm.sets['BOTTOM_RP'])
    model.DisplacementBC(name='BC_BOTTOM_FIXED', createStepName='Initial', region=asm.sets['BOTTOM_RP'], u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
    model.DisplacementBC(name='BC_TOP_GUIDE', createStepName='Initial', region=asm.sets['TOP_RP'], u1=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
    top_start_y = TOP_INITIAL_Y
    model.DisplacementBC(name='BC_TOP_COMPRESS', createStepName='CompressStep', region=asm.sets['TOP_RP'], u2=TOP_FINAL_Y - top_start_y)
    model.DisplacementBC(name='BC_LEFT_OUT_OF_PLANE', createStepName='Initial', region=asm.sets['LEFT_EDGE'], u3=0.0)
    model.DisplacementBC(name='BC_RIGHT_OUT_OF_PLANE', createStepName='Initial', region=asm.sets['RIGHT_EDGE'], u3=0.0)

    job = mdb.Job(name=CASE_ID, model=CASE_ID, numCpus=1, numDomains=1)
    job.writeInput()
    mdb.saveAs(pathName=CASE_ID + '.cae')
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    save_results_if_complete(job)


build()
