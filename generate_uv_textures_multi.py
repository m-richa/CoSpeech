import os, sys, glob
import torch
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
# from tqdm.notebook import tqdm

from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import pickle

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    PerspectiveCameras,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
)
from PIL import Image

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from pytorch3d.io import load_obj, load_ply
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, SfMPerspectiveCameras, PerspectiveCameras, BlendParams, FoVPerspectiveCameras, FoVOrthographicCameras
from pytorch3d.structures import Meshes
import json
import moviepy
import moviepy.editor

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def set_pytorch3D(T):
    n_frame = T.shape[0]
    R = np.array([[[-1,0,0],[0,1,0],[0,0,-1]]]).repeat(n_frame, 0)
    # T = np.array([[-0.19163294, -0.7803175, 5.7486043 ]])
    T = T
    
    # T[:,1] += 0.05 #oliver,conan test1
    # T[:,1] += 0.07 #oliver test1
    
    # fl = np.array([[19.53125]]).repeat(n_frame, 0) #depends on z_near and z_far
    # pp = np.array([[0.75, -0.37109375]]).repeat(n_frame, 0) #depends on bbox
    ###-----------------------------------------------------------------------------------------###
    # Shoulders with hands
    # fl = np.array([[16.129]]).repeat(n_frame, 0)
    # pp = np.array([[0.58065, -0.016129]]).repeat(n_frame, 0)
    
    # #john-oliver-SHOW
    # fl = np.array([[15.625]]).repeat(n_frame, 0)
    # pp = np.array([[0.5625, 0]]).repeat(n_frame, 0)
    
    # # #chemistry-SHOW
    fl = np.array([[16.66666667]]).repeat(n_frame, 0)
    pp = np.array([[-0.86666667, -0.06666667]]).repeat(n_frame, 0)
    
    # # #conan-SHOW
    # fl = np.array([[19.53125]]).repeat(n_frame, 0)
    # pp = np.array([[-0.1328125, -0.25]]).repeat(n_frame, 0)

    # # seth-SHOW
    # fl = np.array([[15.625]]).repeat(n_frame, 0)
    # pp = np.array([[0.09375, 0]]).repeat(n_frame, 0)
    
    return fl, pp, R, T


idx = 'chemistry'
audio_id = 2

device = 'cuda'
Tr = np.load(f'./data/{idx}-SHOW1/all_transl_SHOW1.npy',)
# Tr = np.load(f'./data/{idx}-SHOW1/all_transl_SHOW_{idx}.npy',)
# Tr = Tr[int(Tr.shape[0]*0.8):]
print(Tr.shape)
fl, pp, R, T = set_pytorch3D(Tr)

res = 512
sigma = 1e-4
raster_settings = RasterizationSettings(
        image_size=res,
        faces_per_pixel=1,
        # blur_radius=(np.log(1. / 1e-4 - 1.)*sigma),
        cull_backfaces=True,
        perspective_correct=True
)
lights = PointLights(
            device=device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

materials = Materials(
        specular_color=((0, 0, 0),),
        device=device,
)
camera = PerspectiveCameras(focal_length=fl, 
                             principal_point=pp, 
                             R=R, 
                             T=T, 
                             image_size=res, 
                             device=device)

blend = BlendParams(background_color=(1.0, 1.0, 1.0))
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights,
        materials=materials,
        blend_params=blend,
    )
)

files_meshes = glob.glob(f'/grogu/user/amahapat/LiveSpeechPortraits/data/{idx}-SHOW1/mesh_fullbody_rot180/*.obj')
# files_meshes = glob.glob(f'./data/{idx}-SHOW1/audio2mesh/audio{audio_id}/mesh/*.obj')
files_meshes = sorted(
            files_meshes, key=lambda x: int(x.split("/")[-1].split(".")[0])
        )
print(len(files_meshes))
# files_meshes = files_meshes[:100]
# files_meshes.sort()
# files_rgbs = glob.glob('./data/john-oliver-SHOW1/JO_SHOW1/matted/*.png')
# files_rgbs.sort()

path = f'./data/{idx}-SHOW1'
verts, faces, aux = load_obj('./data/smplx_uv.obj')
verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...].to(device)  # (1, F, 3)
texture_image = torch.load(os.path.join(path, 'texture_uv.pt')).to(device)
# texture_image = torch.full([1, res, res, 3], 0.5, device=device, requires_grad=True)

####### TEXTURED MESHES
save_folder = f'/grogu/user/amahapat/LiveSpeechPortraits/data/{idx}-SHOW1/texture_meshes'
# save_folder = f'./data/{idx}-SHOW1/audio2mesh/audio1/texture_meshes'
print(save_folder)
os.makedirs(save_folder, exist_ok=True)
cnt = 0
outVid = []
for i in tqdm(range(len(files_meshes))):
    # read meshes
    try:
        verts1, faces1, _ = load_obj(files_meshes[i], device=device)
        mesh = Meshes(verts=verts1[None], faces=faces1.verts_idx[None],).to(device)
        new_src_mesh = mesh.clone()
        new_src_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image).to(device)
        with torch.no_grad():
            predicted_image = renderer_textured(new_src_mesh, cameras=camera[0])[0][...,:3].detach().cpu()
            # img = Image.open(files_rgbs[i])
            # img = img.resize((res, res))
            # img = torch.from_numpy(np.array(img)/255.)[:,:,:3]
            img = Image.fromarray((predicted_image.numpy() * 255).astype(np.uint8)).resize((512,512))
            img.save(os.path.join(save_folder, '%05d.png'%i))
    except:
        print(i)
        # break
        continue
    
####### UNTEXTURED MESHES
texture_image = torch.full([1, res, res, 3], 0.5, device=device, requires_grad=True)
save_folder = f'/grogu/user/amahapat/LiveSpeechPortraits/data/{idx}-SHOW1/untexture_meshes'
# save_folder = f'./data/{idx}-SHOW1/audio2mesh/audio1/texture_meshes'
print(save_folder)
os.makedirs(save_folder, exist_ok=True)
cnt = 0
outVid = []
for i in tqdm(range(len(files_meshes))):
    # read meshes
    try:
        verts1, faces1, _ = load_obj(files_meshes[i], device=device)
        mesh = Meshes(verts=verts1[None], faces=faces1.verts_idx[None],).to(device)
        new_src_mesh = mesh.clone()
        new_src_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image).to(device)
        with torch.no_grad():
            predicted_image = renderer_textured(new_src_mesh, cameras=camera[0])[0][...,:3].detach().cpu()
            # img = Image.open(files_rgbs[i])
            # img = img.resize((res, res))
            # img = torch.from_numpy(np.array(img)/255.)[:,:,:3]
            img = Image.fromarray((predicted_image.numpy() * 255).astype(np.uint8)).resize((512,512))
            img.save(os.path.join(save_folder, '%05d.png'%i))
    except:
        print(i)
        # break
        continue

####### DEPTH AND NORMALS
def normalize_depth_01(x):
    x_flat = x.view(x.size(0), -1)
    x_valid = x_flat[x_flat >= 0.0]
    if x_valid.size(0) == 0:
        print('invalid depth')
        return x
    x_valid_min = x_valid.min()
    x_valid_max = x_valid.max()
    x_valid = (x_valid - x_valid_min) / (x_valid_max - x_valid_min)
    x_valid = 1.0 - x_valid
    x_valid = 0.1 + (x_valid * 0.9)
    x_norm = torch.zeros_like(x_flat)
    x_norm[x_flat >= 0.0] = x_valid
    return x_norm.view(x.size(0), x.size(1), x.size(2), x.size(3))

def normal_shader(fragments, meshes):
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    # pixel_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals)
    pixel_normals = interpolate_face_attributes(fragments.pix_to_face, ones, faces_normals)  # <~~~ Use fragments.bary_coords in place of ones for smoothed normals
    normals = pixel_normals[:, :, :, 0]
    normals_magnitude = normals.norm(dim=-1, keepdim=True).repeat(1, 1, 1, 3)
    normals = (normals / normals_magnitude)
    normals[normals_magnitude == 0] = 0
    return normals

save_folder_depth = f'/grogu/user/amahapat/LiveSpeechPortraits/data/{idx}-SHOW1/depths'
save_folder_normals = f'/grogu/user/amahapat/LiveSpeechPortraits/data/{idx}-SHOW1/normals'
# save_folder_depth = f'./data/{idx}-SHOW1/depths'
# save_folder_normals = f'./data/{idx}-SHOW1/normals'
# save_folder_depth = f'./data/{idx}-SHOW1/audio2mesh/audio{audio_id}/depths'
# save_folder_normals = f'./data/{idx}-SHOW1/audio2mesh/audio{audio_id}/normals'
os.makedirs(save_folder_depth, exist_ok=True)
os.makedirs(save_folder_normals, exist_ok=True)

for i in tqdm(range(len(files_meshes))):
    try:
        verts1, faces1, _ = load_obj(files_meshes[i], device=device)
        mesh = Meshes(verts=verts1[None], faces=faces1.verts_idx[None],).to(device)
        
        rasterizer = MeshRasterizer(cameras=camera[i], raster_settings=raster_settings)
        fragments = rasterizer(meshes_world=mesh)
        depth = fragments.zbuf
        # depth = normalize_depth_01(depth)
        depth = depth[0].detach().cpu().numpy()
        np.save(os.path.join(save_folder_depth, '%05d.npy'%i), depth)

        normals = normal_shader(fragments, mesh)
        normals = (normals[0].detach().cpu().numpy()+1.)/2
        normals = (normals * 255).astype(np.uint8)
        normals = Image.fromarray(normals)
        normals.save(os.path.join(save_folder_normals, '%05d.png'%i))
    except:
        print(i)
        # break
        continue