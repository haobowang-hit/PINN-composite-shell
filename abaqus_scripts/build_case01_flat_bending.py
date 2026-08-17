# Abaqus/CAE script.
# Case 1: square flat shell in the XY plane. The x=0 edge, parallel to the
# Y axis, is fixed in translation and prescribed a rotation about Y. The
# opposite edge is constrained in Z, moved in -X, and prescribed the symmetric
# rotation about Y so the specimen bends toward one side.
from abaqus import *
from abaqusConstants import *
import mesh
import os
from section import SectionLayer


SCRIPT_DIR = r'F:/Projects/PINNShell/fem/abaqus_scripts'
CASE_ID = 'case01_flat_bending'
LENGTH = 20.0
WIDTH = 20.0
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
END_UR2 = 0.7853981633974483
LEFT_UR2 = -END_UR2
MESH_SIZE = LENGTH / 60.0
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

    sketch = model.ConstrainedSketch(name='square_shell_profile', sheetSize=2.0)
    sketch.rectangle(point1=(0.0, -0.5 * WIDTH), point2=(LENGTH, 0.5 * WIDTH))
    part = model.Part(name='SQUARE_SHELL', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches['square_shell_profile']

    part.Set(name='ALL_FACES', faces=part.faces[:])
    # The flat shell uses the default shell material direction. For this
    # rectangular part, ply 0 deg is aligned with the global X/edge direction.
    part.SectionAssignment(region=part.sets['ALL_FACES'], sectionName='ShellSection')
    part.Set(name='LEFT_EDGE', edges=part.edges.findAt(((0.0, 0.0, 0.0),)))
    part.Set(name='RIGHT_EDGE', edges=part.edges.findAt(((LENGTH, 0.0, 0.0),)))
    part.seedPart(size=MESH_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    elem_type = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
    part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type,))
    part.generateMesh()

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    inst = assembly.Instance(name='SQUARE_SHELL-1', part=part, dependent=ON)
    assembly.Set(name='LEFT_EDGE', edges=inst.edges.findAt(((0.0, 0.0, 0.0),)))
    assembly.Set(name='RIGHT_EDGE', edges=inst.edges.findAt(((LENGTH, 0.0, 0.0),)))

    model.StaticStep(
        name='BendStep',
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
    configure_energy_history(model, 'BendStep')
    edge_history = ('U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3', 'RF1', 'RF2', 'RF3', 'RM1', 'RM2', 'RM3')
    configure_region_history(model, 'H_LEFT_EDGE', 'BendStep', edge_history, assembly.sets['LEFT_EDGE'])
    configure_region_history(model, 'H_RIGHT_EDGE', 'BendStep', edge_history, assembly.sets['RIGHT_EDGE'])
    model.DisplacementBC(
        name='BC_LEFT_ZERO_CONSTRAINTS',
        createStepName='Initial',
        region=assembly.sets['LEFT_EDGE'],
        u1=UNSET,
        u2=0.0,
        u3=0.0,
        ur1=0.0,
        ur2=UNSET,
        ur3=0.0,
    )
    model.boundaryConditions['BC_LEFT_ZERO_CONSTRAINTS'].setValuesInStep(
        stepName='BendStep',
        ur2=LEFT_UR2,
    )
    model.DisplacementBC(
        name='BC_RIGHT_ZERO_CONSTRAINTS',
        createStepName='Initial',
        region=assembly.sets['RIGHT_EDGE'],
        u1=UNSET,
        u2=0.0,
        u3=0.0,
        ur1=0.0,
        ur2=UNSET,
        ur3=0.0,
    )
    model.boundaryConditions['BC_RIGHT_ZERO_CONSTRAINTS'].setValuesInStep(
        stepName='BendStep',
        ur2=END_UR2,
    )

    job = mdb.Job(name=CASE_ID, model=CASE_ID, numCpus=1, numDomains=1)
    job.writeInput()
    mdb.saveAs(pathName=CASE_ID + '.cae')
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    save_results_if_complete(job)


build()
