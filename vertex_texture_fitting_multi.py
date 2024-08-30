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

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
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
import pdb

def set_pytorch3D(T):
    n_frame = T.shape[0]
    R = np.array([[[-1,0,0],[0,1,0],[0,0,-1]]]).repeat(n_frame, 0)
    # T = np.array([[-0.19163294, -0.7803175, 5.7486043 ]])
    T = T
    
    # fl = np.array([[19.53125]]).repeat(n_frame, 0) #depends on z_near and z_far
    # pp = np.array([[0.75, -0.37109375]]).repeat(n_frame, 0) #depends on bbox
    ###-----------------------------------------------------------------------------------------###
    # Shoulders with hands
    fl = np.array([[16.129]]).repeat(n_frame, 0)
    pp = np.array([[0.58065, -0.016129]]).repeat(n_frame, 0)
    return fl, pp, R, T

def set_pytorch3D_upper(file, cnt=None):
    f = open(file)
    data = json.load(f)
    n_frame = len(data['frames']) if cnt is None else cnt
    R = np.array([[[-1,0,0],[0,1,0],[0,0,-1]]]).repeat(n_frame, 0)
    T = []
    for i in range(n_frame):
        T.append(np.array(data['frames'][i]['transform_matrix'])[:-1,-1])
    T = np.array(T)
    fl = np.array([[data['fl_x']]]).repeat(n_frame, 0)
    pp = np.array([[data['cx'], data['cy']]]).repeat(n_frame, 0)
    return fl, pp, R, T

def visualize_prediction(predicted_mesh, renderer=None, 
                         target_image=None, title='', 
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()

# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")

files = glob.glob('./data/johnoliver_upper_2100/faces_hands_mesh_rot180/*.obj')
# files = glob.glob('./data/johnoliver_hands_2100/faces_hands_mesh_rot180/*.obj')
files.sort()
n_train = int(0.8 * len(files))
# n_train = 200
device = 'cuda'
T = np.load('./data/johnoliver_hands_2100/all_transl.npy',)[:n_train]
print(n_train, T.shape)
fl, pp, R, T = set_pytorch3D(T)
print(fl.shape, pp.shape, R.shape, T.shape)

#save path for checkpoints
# save_path = './checkpoints/jhon-oliver-upper-hands-vertex-texture-fitting-2-200'
# save_path = './checkpoints/jhon-oliver-upper-vertex-texture-fitting'
save_path = './checkpoints/jhon-oliver-upper-hands-vertex-texture-fitting-2-nogloss'
os.makedirs(save_path, exist_ok=True)
save_path_images = f'{save_path}/images'
# save_path_images = './checkpoints/jhon-oliver-upper-vertex-texture-fitting/images'
os.makedirs(save_path_images, exist_ok=True)

# load_path = './checkpoints/jhon-oliver-upper-hands-vertex-texture-fitting'
load_path = './checkpoints/jhon-oliver-upper-hands-uv-texture-fitting'
# load_path = './checkpoints/jhon-oliver-upper-vertex-texture-fitting'
meshes = torch.load(os.path.join(load_path, 'meshes_train.pt'))[:n_train]
print(len(meshes))
# meshes = []
# for i in tqdm(range(n_train)):
#     verts1, faces1, _ = load_obj(files[i], device=device)
#     V = verts1.shape[0]
#     verts_rgb = torch.from_numpy(np.array([0.5, 0.5, 0.5])).float()[None, None, :].repeat(1,V,1)
#     textures = TexturesVertex(verts_features=verts_rgb,).to(device)
#     mesh = Meshes(verts=verts1[None], faces=faces1.verts_idx[None], textures=textures,).to(device)
#     meshes.append(mesh)

# torch.save(meshes, os.path.join(save_path, 'meshes_train.pt'))    
# pdb.set_trace()

raster_settings = RasterizationSettings(
        image_size=256,
        faces_per_pixel=1,
        cull_backfaces=True,
        perspective_correct=True
)
lights = PointLights(
            device=device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

camera = PerspectiveCameras(focal_length=fl, 
                             principal_point=pp, 
                             R=R, 
                             T=T, 
                             image_size=256, 
                             device=device)

blend = BlendParams(background_color=(1.0, 1.0, 1.0))
materials = Materials(
        specular_color=((0, 0, 0),),
        device=device,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights,
        blend_params=blend,
        materials=materials,
    )
)

# files = glob.glob('./data/johnoliver_upper_2100/images/*')
files = glob.glob('./data/johnoliver_hands_2100/matted/*')
files.sort()
target_rgb = []
for i in tqdm(range(n_train)):
    img = Image.open(files[i])
    img = img.resize((256, 256))
    img = torch.from_numpy(np.array(img)/255.)
    target_rgb.append(img[:,:,:3].to('cuda'))
print(len(target_rgb))

# Rasterization settings for differentiable rendering, where the blur_radius
# initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
# Renderer for Image-based 3D Reasoning', ICCV 2019
sigma = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    faces_per_pixel=50, 
)

# # Silhouette renderer 
# renderer_silhouette = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=camera, 
#         raster_settings=raster_settings_soft
#     ),
#     shader=SoftSilhouetteShader()
# )

# Rasterization settings for differentiable rendering, where the blur_radius
# initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
# Renderer for Image-based 3D Reasoning', ICCV 2019
sigma = 1e-4
# raster_settings_soft = RasterizationSettings(
#     image_size=256, 
#     blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
#     faces_per_pixel=50, 
#     perspective_correct=False, 
# )
raster_settings = RasterizationSettings(
        image_size=256,
        faces_per_pixel=1,
        cull_backfaces=True,
        perspective_correct=True
)

# Differentiable soft renderer using per vertex RGB colors for texture
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights,
        blend_params=blend,
        materials=materials,
    )
)

# Number of views to optimize over in each SGD iteration
num_views_per_iteration = 128
# Number of optimization steps
Niter = 50000
# Plot period for the losses
plot_period = 100
src_mesh = meshes[0].clone()

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

# Optimize using rendered RGB image loss, rendered silhouette image loss, mesh 
# edge loss, mesh normal consistency, and mesh laplacian smoothing
losses = {"rgb": {"weight": 1.0, "values": []},
          "silhouette": {"weight": 1.0, "values": []},
          "edge": {"weight": 1.0, "values": []},
          "normal": {"weight": 0.01, "values": []},
          "laplacian": {"weight": 1.0, "values": []},
         }

# src_mesh
verts_shape = src_mesh.verts_packed().shape
# of the mesh
sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
# The optimizer
optimizer = torch.optim.SGD([sphere_verts_rgb], lr=10.0, momentum=0.9)

loop = tqdm(range(Niter))
for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Losses to smooth /regularize the mesh shape
    loss = {k: torch.tensor(0.0, device=device) for k in losses}
    
    for j in np.random.permutation(n_train).tolist()[:num_views_per_iteration]:
        # print(j)
        new_src_mesh = meshes[j].clone()
        new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 
        images_predicted = renderer_textured(new_src_mesh, cameras=camera[j])

        predicted_rgb = images_predicted[..., :3]
        loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
        loss["rgb"] += loss_rgb / num_views_per_iteration
    
    # Weighted sum of the losses
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        losses[k]["values"].append(float(l.detach().cpu()))
    
    # Print the losses
    # loop.set_description("total_loss = %.6f" % sum_loss)
    
    # Plot mesh
    if i % plot_period == 0:
        # print('Plotting!')
        with torch.no_grad():
            idx = np.random.permutation(n_train).tolist()[0]
            new_src_mesh = meshes[idx].clone()
            new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
            # visualize_prediction(new_src_mesh, renderer=renderer_textured, title="iter: %d" % i, silhouette=False)
            # visualize_prediction(new_src_mesh, renderer=renderer_textured, target_image=target_rgb[idx], title="iter: %d" % i, silhouette=False)
            ## visualize
            predicted_image = renderer_textured(new_src_mesh, cameras=camera[idx])
            gen_viz = (predicted_image[0][...,:3].detach().cpu().numpy() *255.).astype(np.uint8)
            # Image.fromarray(gen_viz).save(os.path.join(save_path_images, f'gen_image_{i}.jpg'))
            target_viz = (target_rgb[idx].detach().cpu().numpy() *255.).astype(np.uint8)
            viz = np.concatenate((gen_viz, target_viz), axis=1)
            Image.fromarray(viz).resize((1024,512)).save(os.path.join(save_path_images, f'viz_image_{i}.jpg'))
            
            torch.save(sphere_verts_rgb.detach().cpu(), os.path.join(save_path, 'verts_rgb.pt'))
    # Optimization step
    sum_loss.backward()
    optimizer.step()
