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
from util.visualizer import VideoWriter
import pdb


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 8    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print(len(dataset))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.load_epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)


    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.load_epoch))
    model.eval()

    if opt.n_prev_frames > 0:
        name = opt.name
        n_prev_frames = opt.n_prev_frames

        opt.name = opt.single_name     #load model trained with n_prev_frames=0
        opt.n_prev_frames = 4
        single_model = create_model(opt)
        single_model.setup(opt)
        opt.name = name
        opt.n_prev_frames = n_prev_frames
        single_model.eval()

    # generated_frames = []
    # gt_frames = []
    writer_vid = VideoWriter(
                path=os.path.join(opt.results_dir, opt.name, 'video.mp4'),
                frame_rate=60,
                bit_rate=int(4 * 1000000))
    for i, data in enumerate(dataset):
        # if i * opt.batch_size > 100:
        #     break
        # if i < 2 or i+2 > len(dataset)-1:
        #     continue
        t = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals(mode='test_demo')  # get image results
        # print(time.time()-t)
        # generated_frames.append(visuals['pred_fake'])
        # img_path = model.get_image_paths()
        red = torch.ones_like(visuals['pred_fake']) * -1.
        red[:,0,:,:] = 1.
        # generate image overlaped with conditional frame
        # pdb.set_trace()
        mask = data['mask'].to(red.device)
        combined = (mask == 1) * red + (mask != 1) * visuals['pred_fake']
        writer_vid.write(combined)

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image...' % (i * opt.batch_size))
    #     save_images(webpage, visuals, i)
    # webpage.save()  # save the HTML

    writer_vid.close()

    print('done!')