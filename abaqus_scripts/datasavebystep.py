from abaqus import session
from abaqusConstants import *
import csv
import os


if 'ODB_PATH_TO_EXPORT' in globals():
    odb_key = ODB_PATH_TO_EXPORT
    if odb_key not in session.odbs.keys():
        session.openOdb(name=odb_key)
    odb = session.odbs[odb_key]
elif 'ODB_NAME_TO_EXPORT' in globals():
    odb_key = ODB_NAME_TO_EXPORT
    odb = session.odbs[odb_key]
else:
    odb = session.odbs[session.odbs.keys()[-1]]
odb_basename = os.path.basename(odb.path).replace('.odb', '')
print('Exporting ODB: {}'.format(odb.path))


def padded(data, n):
    try:
        values = list(data)
    except TypeError:
        values = [data]
    while len(values) < n:
        values.append(0.0)
    return values[:n]


def field_values(frame, field_name, instance, position=None):
    if field_name not in frame.fieldOutputs.keys():
        return [], []
    field = frame.fieldOutputs[field_name]
    try:
        subset = field.getSubset(region=instance, position=position) if position is not None else field.getSubset(region=instance)
    except:
        subset = field.getSubset(region=instance)
    labels = list(getattr(field, 'componentLabels', []))
    return subset.values, labels


def section_point_info(value):
    section_point = getattr(value, 'sectionPoint', None)
    if section_point is None:
        return '', ''
    number = getattr(section_point, 'number', '')
    description = getattr(section_point, 'description', '')
    return number, description


def safe_value_attr(value, name, default=''):
    return getattr(value, name, default)


def write_nodal_file(filename, instance_name, instance, frame):
    displacement_values, _ = field_values(frame, 'U', instance, NODAL)
    rotation_values, _ = field_values(frame, 'UR', instance, NODAL)
    reaction_values, _ = field_values(frame, 'RF', instance, NODAL)
    moment_values, _ = field_values(frame, 'RM', instance, NODAL)
    node_disp_map = {}
    node_rot_map = {}
    node_rf_map = {}
    node_rm_map = {}
    for value in displacement_values:
        node_disp_map[value.nodeLabel] = padded(value.data, 3)
    for value in rotation_values:
        node_rot_map[value.nodeLabel] = padded(value.data, 3)
    for value in reaction_values:
        node_rf_map[value.nodeLabel] = padded(value.data, 3)
    for value in moment_values:
        node_rm_map[value.nodeLabel] = padded(value.data, 3)

    with open(filename, mode='wb') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Instance', 'NodeLabel',
            'X0', 'Y0', 'Z0', 'U1', 'U2', 'U3', 'X', 'Y', 'Z',
            'UR1', 'UR2', 'UR3', 'RF1', 'RF2', 'RF3', 'RM1', 'RM2', 'RM3'
        ])
        for node in instance.nodes:
            node_label = node.label
            x0, y0, z0 = node.coordinates
            u = node_disp_map.get(node_label, [0.0, 0.0, 0.0])
            ur = node_rot_map.get(node_label, [0.0, 0.0, 0.0])
            rf = node_rf_map.get(node_label, [0.0, 0.0, 0.0])
            rm = node_rm_map.get(node_label, [0.0, 0.0, 0.0])
            writer.writerow([
                instance_name, node_label,
                x0, y0, z0, u[0], u[1], u[2], x0 + u[0], y0 + u[1], z0 + u[2],
                ur[0], ur[1], ur[2], rf[0], rf[1], rf[2], rm[0], rm[1], rm[2],
            ])
    print('Saved nodal fields: {}'.format(filename))


def write_element_field_file(filename, instance_name, frame, instance, field_names):
    written = 0
    with open(filename, mode='wb') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Instance', 'Field', 'Position', 'ElementLabel', 'IntegrationPoint',
            'SectionPointNumber', 'SectionPointDescription', 'ComponentLabels',
            'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6'
        ])
        for field_name in field_names:
            values, component_labels = field_values(frame, field_name, instance, INTEGRATION_POINT)
            if not values:
                values, component_labels = field_values(frame, field_name, instance)
            label_text = ';'.join([str(x) for x in component_labels])
            for value in values:
                sp_number, sp_desc = section_point_info(value)
                data = padded(value.data, 6)
                writer.writerow([
                    instance_name,
                    field_name,
                    safe_value_attr(value, 'position', ''),
                    safe_value_attr(value, 'elementLabel', ''),
                    safe_value_attr(value, 'integrationPoint', ''),
                    sp_number,
                    sp_desc,
                    label_text,
                    data[0], data[1], data[2], data[3], data[4], data[5],
                ])
                written += 1
    if written:
        print('Saved element fields: {} rows={}'.format(filename, written))
    else:
        print('Warning: no requested element fields {} found for {}'.format(field_names, filename))
    return written


def write_available_fields_file(filename, frame):
    with open(filename, mode='wb') as file:
        writer = csv.writer(file)
        writer.writerow(['Field', 'ComponentLabels', 'Description'])
        for field_name in sorted(frame.fieldOutputs.keys()):
            field = frame.fieldOutputs[field_name]
            component_labels = ';'.join([str(x) for x in list(getattr(field, 'componentLabels', []))])
            description = getattr(field, 'description', '')
            writer.writerow([field_name, component_labels, description])


def vector_between(node_map, label_a, label_b):
    a = node_map[label_a].coordinates
    b = node_map[label_b].coordinates
    return b[0] - a[0], b[1] - a[1], b[2] - a[2]


def normalize(vector):
    norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]) ** 0.5
    if norm <= 1.0e-30:
        return 0.0, 0.0, 0.0
    return vector[0] / norm, vector[1] / norm, vector[2] / norm


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def write_element_orientation_file(filename, instance_name, instance):
    node_map = {}
    for node in instance.nodes:
        node_map[node.label] = node
    with open(filename, mode='wb') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Instance', 'ElementLabel',
            'Node1', 'Node2', 'Node3', 'Node4',
            'Tangent1_X', 'Tangent1_Y', 'Tangent1_Z',
            'Width2_X', 'Width2_Y', 'Width2_Z',
            'Normal_X', 'Normal_Y', 'Normal_Z',
        ])
        for element in instance.elements:
            conn = list(element.connectivity)
            if len(conn) < 4:
                continue
            t1 = normalize(vector_between(node_map, conn[0], conn[1]))
            t2 = normalize(vector_between(node_map, conn[0], conn[3]))
            normal = normalize(cross(t1, t2))
            writer.writerow([
                instance_name, element.label,
                conn[0], conn[1], conn[2], conn[3],
                t1[0], t1[1], t1[2],
                t2[0], t2[1], t2[2],
                normal[0], normal[1], normal[2],
            ])


for step_name, step in odb.steps.items():
    for frame_index, frame in enumerate(step.frames):
        for instance_name, instance in odb.rootAssembly.instances.items():
            clean_instance = instance_name.replace('/', '_')
            base = '{}_{}_{}_frame{}'.format(odb_basename, clean_instance, step_name, frame_index)

            fields_filename = '{}_available_fields.csv'.format(base)
            write_available_fields_file(fields_filename, frame)
            orientation_filename = '{}_element_orientation.csv'.format(base)
            write_element_orientation_file(orientation_filename, instance_name, instance)

            nodal_filename = '{}.csv'.format(base)
            write_nodal_file(nodal_filename, instance_name, instance, frame)

            ip_filename = '{}_ipfields.csv'.format(base)
            write_element_field_file(ip_filename, instance_name, frame, instance, ('LE', 'E', 'S', 'SE', 'SK', 'SENER'))

            shell_filename = '{}_shell_resultants.csv'.format(base)
            write_element_field_file(shell_filename, instance_name, frame, instance, ('SF', 'SM', 'SE', 'SK'))


history_filename = '{}_history.csv'.format(odb_basename)
with open(history_filename, mode='wb') as file:
    writer = csv.writer(file)
    writer.writerow(['Step', 'Region', 'Output', 'Time', 'Value'])
    for step_name, step in odb.steps.items():
        for region_name, region in step.historyRegions.items():
            for output_name, output in region.historyOutputs.items():
                for time_value in output.data:
                    writer.writerow([step_name, region_name, output_name, time_value[0], time_value[1]])
print('Saved: {}'.format(history_filename))
