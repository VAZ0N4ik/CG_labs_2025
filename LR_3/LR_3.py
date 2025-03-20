import numpy as np
from PIL import Image
import math
from PIL import ImageOps
import random
import math

def task3_4_loadvertices(filename):
    vertices=[]
    with open(filename,'r') as file:
          for line in file:
               line=line.split()
               if line[0]==('v'):
                     line=line[1:]
                     vertice=[float(vert) for vert in line]
                     vertices.append(vertice)
    return np.array(vertices)
def task5_loadfaces(filename):
    faces=[]
    with open(filename,'r') as file:
        for line in file:
                if line.startswith('f'):
                    face=[int(vert.split('/')[0])-1 for vert in line.strip().split()[1:]]
                    faces.append(face)
    return faces
def task15_rotate(vertices,alpha,beta,gamma):
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
     vertices= vertices@R +[0,-0.04,0.2]
     return vertices
def task10_drawmodel(faces,vertices,alpha,beta,gamma):
     img=np.zeros((1000,1000,3),dtype=np.uint8)
     height, width= img.shape[:2]
     z_buf=np.matrix(np.inf*np.ones((height,width)))
     scaled_vertices=vertices[:,:3]
     scaled_vertices=task15_rotate(scaled_vertices,alpha,beta,gamma)
     for face in faces:
               xy1=scaled_vertices[face[0]]
               xy2=scaled_vertices[face[1]]
               xy3=scaled_vertices[face[2]]
               task_8_triangles((xy1[0]),(xy1[1]),(xy1[2]),(xy2[0]),(xy2[1]),(xy2[2]),(xy3[0]),(xy3[1]),(xy3[2]),width,height,img,z_buf)
     image=Image.fromarray(img)
     np.flip(img,1)
     image=Image.fromarray(img)
     image=ImageOps.flip(image)
     image.save('model_img.png')
def task_11_norm(x0,y0,z0,x1,y1,z1,x2,y2,z2):
     x=(y1-y2)*(z1-z0)-(z1-z2)*(y1-y0)
     y=(x1-x2)*(z1-z0)-(z1-z2)*(x1-x0)
     z=(x1-x2)*(y1-y0)-(y1-y2)*(x1-x0)
     return (x,y,z)
def task_7_coordinates(x,y,x0,y0,x1,y1,x2,y2):
     denominator=(x0-x2)*(y1-y2)-(x1-x2)*(y0-y2)
     if(abs(denominator)>1e-6):
          lambda0=((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))
          lambda1=((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))
          lambda2=1.0-lambda0-lambda1
          return (lambda0,lambda1,lambda2)
     else:
          raise Exception("Not a triangle")
def task_8_triangles(x0,y0,z0,x1,y1,z1,x2,y2,z2,imagex,imagey,img,z_buf):
     px0=0.2*5000*x0/z0+500
     px1=0.2*5000*x1/z1+500
     px2=0.2*5000*x2/z2+500
     py0=0.2*5000*y0/z0+500
     py1=0.2*5000*y1/z1+500
     py2=0.2*5000*y2/z2+500
     x_min=min(px0,px1,px2) if min(px0,px1,px2)>=0 else 0
     y_min=min(py0,py1,py2) if min(px0,px1,px2)>=0 else 0
     x_max=max(px0,px1,px2) if max(px0,px1,px2)<=imagex else imagex
     y_max=max(py0,py1,py2) if max(px0,px1,px2)<= imagey else imagey
     nx,ny,nz=task_11_norm(x0,y0,z0,x1,y1,z1,x2,y2,z2)
     cos=nz/math.sqrt((nx**2+ny**2+nz**2))
     if cos<0:
          for x in range(int(x_min),int(x_max)+1):
               for y in range(int(y_min),int(y_max)+1):
                    lambda0,lambda1,lambda2=task_7_coordinates(x,y,px0,py0,px1,py1,px2,py2)
                    z=lambda0*z0+lambda1*z1+lambda2*z2
                    if lambda0>=0 and lambda1>=0 and lambda2>=0 and z<=z_buf[y,x] :
                         z_buf[y,x]=z
                         img[y,x]=(-255*cos,0,0)
if __name__=="__main__":
    vertices=task3_4_loadvertices('../model.obj')
    faces=task5_loadfaces('../model.obj')
    task10_drawmodel(faces,vertices,0,120,15)
