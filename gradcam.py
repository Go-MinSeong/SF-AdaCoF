import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
import os
import sys
import gc
import numpy as np


parser = argparse.ArgumentParser(description='Feature map Grad CAM')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--training_from',type=str,default='vimeo')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--model_dir', type=str, default='./AdaCoF_all_share')
parser.add_argument('--checkpoint', type=str, default='./AdaCoF_all_share/model50.pth')
parser.add_argument('--log', type=str, default='./test_grad_cam.txt')
parser.add_argument('--out_dir', type=str, default='./output/AdaCoF_all_share')
parser.add_argument('--kernel_size', type=int, default=5) 
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--pr_path', type=str, default='../../test_input/middlebury_others/input/Beanbags/frame10.png')
parser.add_argument('--ne_path', type=str, default='../../test_input/middlebury_others/input/Beanbags/frame11.png')
parser.add_argument('--gt', type=str , default = '../../test_input/middlebury_others/gt/Beanbags/frame10i11.png')
parser.add_argument('--base_img', type=str , default = '../../test_input/middlebury_others/input/Beanbags/frame11.png')
parser.add_argument('--base_img_type', type=str , default = 'ne',help = ['pr','ne'])
parser.add_argument('--target_layer', type=str , default = 'model.get_kernel.moduleDeconv2',help = "Check your model layer")

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.target_layer = None
        self.target_output_list = []
        self.gradients_list = []


        for name, layer in self.model.named_modules():
            if name == self.target_layer_name:
                self.target_layer = layer
                layer.register_forward_hook(self.save_output)
                break
        
        if self.target_layer is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found in the model.")
        
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_output(self, module, input, output):
        self.target_output = output
        self.target_output_list.append(self.target_output)
        print(f'Target Layer Output Feature map shape : {output.shape}')

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        self.gradients_list.append(self.gradients)
        print(f'Gradient Output shape : {self.gradients.shape}')

    def get_gradcam(self, pr_img,ne_img,choice,model_dir):
        self.model.zero_grad()

        output = self.model(pr_img,ne_img)
        output.backward(gradient=output)

        if model_dir != './AdaCoF_all_xshare':
            if choice == 'pr':
                alpha = torch.mean(self.gradients_list[0], dim=(2, 3), keepdim=True) # GAP
                gradcam = torch.sum(alpha * self.target_output_list[0], dim=1, keepdim=True)
            else:
                alpha = torch.mean(self.gradients_list[1], dim=(2, 3), keepdim=True) # GAP
                gradcam = torch.sum(alpha * self.target_output_list[1], dim=1, keepdim=True)
        else:
            alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # GAP
            gradcam = torch.sum(alpha * self.target_output, dim=1, keepdim=True)


        gradcam = F.relu(gradcam)
        gradcam /= torch.max(gradcam) + 1e-9

        return gradcam

def overlay_gradcam_on_image(image, gradcam):
    heatmap = cv2.resize(gradcam, (image.shape[1], image.shape[0]))
    heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9) * 255
    # Convert the heatmap to a 3-channel image with transparency
    heatmap_overlay = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)
    # heatmap_overlay[:, :, :3]  # Set the transparency (adjust as needed)
    overlaid_image = cv2.addWeighted(image, 1, heatmap_overlay, 0.5, 0)
    return overlaid_image

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    sys.path.append('/home/work/capstone/Final')
    sys.path.append(args.model_dir)
    
    import models
    if args.log is not None:
        logfile = open(args.log,'a')
        logfile.write(f'---------------'  + '\n' + f'Model : {args.model_dir.replace(".","")}' +'\n')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        devcie = torch.device('cpu')

    model = models.Model(args,'models.')

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint)
    model.load(checkpoint['state_dict'])
    current_epoch = checkpoint['epoch']

    # for name, layer in model.named_modules():
    #     print(name)

    pr_image = Image.open(args.pr_path)
    ne_image = Image.open(args.ne_path)
    gt_image = Image.open(args.gt)
    base_image = Image.open(args.base_img)

    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    pr_img = preprocess(pr_image).unsqueeze(0).to(device)
    ne_img = preprocess(ne_image).unsqueeze(0).to(device)

    model.eval()

    gradcam_calculator = GradCAM(model, target_layer_name=args.target_layer)  
    gradcam = gradcam_calculator.get_gradcam(pr_img,ne_img,args.base_img_type,args.model_dir)

    gradcam_heatmap = gradcam[0, 0].cpu().detach().numpy()
    overlaid_image = overlay_gradcam_on_image(np.array(base_image), gradcam_heatmap)
    result = Image.fromarray(overlaid_image)
    result.save(args.out_dir + f'/result_{args.base_img_type}.png')

if __name__ == "__main__":
    main()