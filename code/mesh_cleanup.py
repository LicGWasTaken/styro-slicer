import pymesh

def cleanup(mesh):
    print('\ncleaning up mesh...')

    # Remove isolated vertices
    result = pymesh.remove_isolated_vertices(mesh)
    output_mesh = result[0]
    num_vertex_removed = result[1].get('num_vertex_removed')
    if num_vertex_removed != 0:
        mesh = output_mesh
        print('--> removed ' + num_vertex_removed + ' isolated vertices')
    else:
        print('--> no isolated vertices')

    # Remove duplicate vertices
    result = pymesh.remove_duplicated_vertices(mesh)
    output_mesh = result[0]
    num_vertex_merged = result[1].get('num_vertex_merged')
    if num_vertex_merged != 0:
        mesh = output_mesh
        print('--> merged ' + num_vertex_merged + ' duplicate vertices')
    else:
        print('--> no duplicate vertices')

    # Remove duplicated vertices
    result = pymesh.remove_duplicated_faces(mesh)
    output_mesh = result[0]
    mesh = output_mesh
    print('--> removing duplicate faces...') #TODO add a check for weather there were actually duplicated faces

    # Remove degenerate triangles (= triangles with 0 area)
    result = pymesh.remove_degenerated_triangles(mesh)
    output_mesh == result[0]
    mesh = output_mesh
    print('--> removing degenerate triangles...')

    return mesh