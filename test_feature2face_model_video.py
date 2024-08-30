import os
from options.test_feature2face_options import TestOptions
from datasets import create_dataset
from models import create_model
from util import html
import time
from util.visualizer import Visualizer
from util.visualizer import save_images
import tqdm
import glob
from PIL import Image
import moviepy
import moviepy.editor
import numpy as np
import torch
import pdb
from util.visualizer import VideoWriter


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    print(len(dataset))
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.load_epoch)
    )  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)

    if opt.use_wandb:
        wandb_run = (
            wandb.init(
                project=opt.wandb_project_name,
                name=opt.name,
                config=opt,
                entity="cmu-gil",
            )
            if not wandb.run
            else wandb.run
        )
        wandb_run._label(repo=opt.wandb_project_name, use_wandb=opt.use_wandb)

    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s"
        % (opt.name, opt.phase, opt.load_epoch),
    )
    model.eval()

    writer_vid = VideoWriter(
        path=os.path.join(opt.results_dir, opt.name, "video-audio1-latest.mp4"),
        frame_rate=30,
        bit_rate=int(4 * 1000000),
    )

    writer_vid_com = VideoWriter(
        path=os.path.join(opt.results_dir, opt.name, "video_com-audio1-latest.mp4"),
        frame_rate=30,
        bit_rate=int(4 * 1000000),
    )

    if opt.n_prev_frames > 0:
        name = opt.name
        n_prev_frames = opt.n_prev_frames

        opt.name = opt.single_name  # load model trained with n_prev_frames=0
        opt.n_prev_frames = 4
        single_model = create_model(opt)
        single_model.setup(opt)
        opt.name = name
        opt.n_prev_frames = n_prev_frames
        single_model.eval()

    # generated_frames = []
    # gt_frames = []
    count = 0
    for i, data in enumerate(dataset):
        # if i * opt.batch_size > 100:
        #     break
        if i < 2 or i + 2 > len(dataset) - 1:
            continue
        t = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        print(time.time()-t, visuals["mesh"].shape)

        # red = torch.ones_like(visuals['pred_fake']) * -1.
        # red[:,0,:,:] = 1.

        # generate image overlaped with conditional frame
        # pdb.set_trace()
        # mask = data['mask'].to(red.device)
        # combined = (mask == 1) * red + (mask != 1) * visuals['pred_fake']
        # combined_gt = (mask == 1) * red + (mask != 1) * visuals['pred_real']

        # mask = visuals["feature_map"].repeat(1, 3, 1, 1)

        # depth = visuals["feature_map"][:, :1, ...].repeat(1, 3, 1, 1) * 2 - 1.
        # nerf = visuals["feature_map"][:, :3, ...]

        # white_img = torch.ones_like(visuals["pred_real"])
        # pdb.set_trace()
        # mask_real = (visuals["pred_real"].sum(1, keepdim=True)==0) * 1.
        # visuals["pred_real"] = mask_real * white_img + (1-mask_real) * visuals["pred_real"]
        # pdb.set_trace()
        # combined = torch.cat([visuals["pred_fake"], mask], -2)
        # combined_gt = torch.cat([visuals["pred_real"], mask], -2)
        # com1 = torch.cat([visuals["pred_fake"], visuals["pred_real"]], -1)
        # com2 = torch.cat([combined, combined_gt], -1)

        # visuals["mask"] = data["mask"].to(visuals['pred_fake'].device)
        # pdb.set_trace()
        # com1 = torch.cat([visuals["pred_fake"], visuals["pred_real"]], -1)
        mesh = visuals["mesh"]
        com1 = torch.cat([mesh, visuals["pred_fake"], visuals["pred_real"]], -1)
        com2 = visuals["pred_fake"]

        writer_vid.write(com1)
        writer_vid_com.write(com2)
        count += 1

        if i % 5 == 0:  # save images to an HTML file
            print("processing (%04d)-th image..." % (i * opt.batch_size))

        # if i * opt.batch_size > 1000:
        #     break
        if count * opt.batch_size >= 500:
            break

        save_images(webpage, visuals, i)
    webpage.save()  # save the HTML

    writer_vid.close()
    writer_vid_com.close()

    print("done!")

    # img_list = []
    # feat_list = []
    # gt_img_list = []
    # # for i in range(len(dataset)-1):
    # for i in range(len(generated_frames)-1):
    #     img_path = web_dir + '/images/%06d_pred_fake.png'%i
    #     img_list.append(img_path)
    #     feat_path = web_dir + '/images/%06d_feature_map.png'%i
    #     feat_list.append(feat_path)
    #     gt_img_path = web_dir + '/images/%06d_pred_real.png'%i
    #     gt_img_list.append(gt_img_path)
    # outVid = []
    # for i in tqdm.tqdm(range(len(img_list))):
    #     img = img_list[i]
    #     img_ = Image.open(img).convert("RGB")
    #     img_ = np.array(img_)
    #     gt_img = gt_img_list[i]
    #     gt_img_ = Image.open(gt_img).convert("RGB")
    #     gt_img_ = np.array(gt_img_)
    #     feat = feat_list[i]
    #     feat_ = Image.open(feat).convert("RGB")
    #     feat_ = np.array(feat_)
    #     out = np.concatenate((feat_, img_), axis=1)
    #     out = np.concatenate((out, gt_img_), axis=1)
    #     outVid.append(out)
    # moviepy.editor.ImageSequenceClip(sequence=[(npyFrame).clip(0.0, 255.0).round().astype(np.uint8) for npyFrame in outVid], fps=30).write_videofile(web_dir+'/generated_video.mp4')
