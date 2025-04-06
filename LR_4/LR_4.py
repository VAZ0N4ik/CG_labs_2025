import numpy as np
from PIL import Image
from PIL import ImageOps
import math


def task3_4_loadvertices(filename):
    vertices = []
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                line = line.split()
                line = line[1:]
                vertice = [float(vert) for vert in line]
                vertice.append(i)
                i += 1
                vertices.append(vertice)
    return np.array(vertices)


def task3_4_loadverticest(filename):
    vertices = []
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('vt'):
                line = line.split()
                line = line[1:]
                vertice = [float(vert) for vert in line]
                vertice.append(i)
                i += 1
                vertices.append(vertice)
    return np.array(vertices)


def task5_loadfaces(filename):
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('f'):
                face = [int(vert.split('/')[0]) - 1 for vert in line.strip().split()[1:]]
                faces.append(face)
    return faces


def task5_loadfaces_text(filename):
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('f'):
                face = [int(vert.split('/')[1]) - 1 for vert in line.strip().split()[1:]]
                faces.append(face)
    return faces


def task15_rotate(vertices, alpha, beta, gamma):
    alpha = np.radians(alpha)  # замените на нужные значения
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_z = np.array([
        [np.cos(gamma), np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    R = R_x @ R_y @ R_z
    vertices[:, :3] = vertices[:, :3] @ R + [0, -0.04, 0.2]
    return vertices


def task10_drawmodel(target_path, faces, faces_text, text, vertices, verticest, alpha, beta, gamma):
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    height, width = img.shape[:2]
    z_buf = np.matrix(np.inf * np.ones((height, width)))
    scaled_vertices = vertices
    scaled_vertices = task15_rotate(scaled_vertices, alpha, beta, gamma)
    # verticest=task15_rotate(verticest,alpha,beta,gamma)
    norm = np.zeros((len(vertices), 3))
    i = 0
    for face in faces:
        xy1 = scaled_vertices[face[0]]
        xy2 = scaled_vertices[face[1]]
        xy3 = scaled_vertices[face[2]]
        nx, ny, nz = task_11_norm(xy1[0], xy1[1], xy1[2], xy2[0], xy2[1], xy2[2], xy3[0], xy3[1], xy3[2])
        norm[int(xy1[3])] += nx, ny, nz
        norm[int(xy2[3])] += nx, ny, nz
        norm[int(xy3[3])] += nx, ny, nz
    for face in faces:
        xy1 = scaled_vertices[face[0]]
        xy2 = scaled_vertices[face[1]]
        xy3 = scaled_vertices[face[2]]
        xy1_text = verticest[faces_text[i][0]]
        xy2_text = verticest[faces_text[i][1]]
        xy3_text = verticest[faces_text[i][2]]
        i += 1
        task_8_triangles((xy1[0]), (xy1[1]), (xy1[2]), (xy2[0]), (xy2[1]), (xy2[2]), (xy3[0]), (xy3[1]), (xy3[2]),
                         width, height, img, z_buf, int(xy1[3]), int(xy2[3]), int(xy3[3]), norm, text, xy1_text,
                         xy2_text, xy3_text)
    image = Image.fromarray(img)
    np.flip(img, 1)
    image = Image.fromarray(img)
    image = ImageOps.flip(image)
    image.save(target_path)


def task_11_norm(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x = (y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0)
    y = (x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)
    z = (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0)
    return (x, y, z)


def task_11_norm_dot(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x = (y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0)
    y = (x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)
    z = (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0)
    return (x, y, z)


def task_7_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if (abs(denominator) > 1e-6):
        lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
        lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
        lambda2 = 1.0 - lambda0 - lambda1
        return (lambda0, lambda1, lambda2)
    else:
        raise Exception("Not a triangle")


def task_8_triangles(x0, y0, z0, x1, y1, z1, x2, y2, z2, imagex, imagey, img, z_buf, ind1, ind2, ind3, norm, text,
                     xy1_text, xy2_text, xy3_text):
    px0 = 0.2 * 5000 * x0 / z0 + 500
    px1 = 0.2 * 5000 * x1 / z1 + 500
    px2 = 0.2 * 5000 * x2 / z2 + 500
    py0 = 0.2 * 5000 * y0 / z0 + 500
    py1 = 0.2 * 5000 * y1 / z1 + 500
    py2 = 0.2 * 5000 * y2 / z2 + 500
    x_min = min(px0, px1, px2) if min(px0, px1, px2) >= 0 else 0
    y_min = min(py0, py1, py2) if min(px0, px1, px2) >= 0 else 0
    x_max = max(px0, px1, px2) if max(px0, px1, px2) <= imagex else imagex
    y_max = max(py0, py1, py2) if max(px0, px1, px2) <= imagey else imagey
    # light [0,0,1]
    n0 = norm[ind1][2] / np.linalg.norm(norm[ind1])
    n1 = norm[ind2][2] / np.linalg.norm(norm[ind2])
    n2 = norm[ind3][2] / np.linalg.norm(norm[ind3])
    nx, ny, nz = task_11_norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cos = nz / math.sqrt((nx ** 2 + ny ** 2 + nz ** 2))
    if cos < 0:
        for x in range(int(x_min), int(x_max) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                lambda0, lambda1, lambda2 = task_7_coordinates(x, y, px0, py0, px1, py1, px2, py2)
                indx = round(1024 * (lambda0 * xy1_text[0] + lambda1 * xy2_text[0] + lambda2 * xy3_text[0]))
                indy = round(1024 * (lambda0 * xy1_text[1] + lambda1 * xy2_text[1] + lambda2 * xy3_text[1]))
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0 and z <= z_buf[y, x]:
                    z_buf[y, x] = z
                    img[y, x] = -(n0 * lambda0 + n1 * lambda1 + n2 * lambda2) * text[indy, indx]


def render_with_gourad_and_texture(model_path, texture_path, target_path):
    vertices = task3_4_loadvertices(model_path)
    faces = task5_loadfaces(model_path)
    verticest = task3_4_loadverticest(model_path)
    text = np.asarray(Image.open(texture_path))
    text = text[::-1]
    faces_text = task5_loadfaces_text(model_path)
    task10_drawmodel(target_path, faces, faces_text, text, vertices, verticest, 0, 120, 17)

if __name__ == "__main__":
    render_with_gourad_and_texture('../model.obj', 'bunny-atlas.jpg', 'textured.png')