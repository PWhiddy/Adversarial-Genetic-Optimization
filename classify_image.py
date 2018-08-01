# based on https://discuss.pytorch.org/t/pretrained-resnet-constant-output/2760

from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import time
import random

imx = 224
imy = 224
cx = imx/2
cy = imx/2
#image = np.random.rand(500,500)
#scipy.misc.imsave('image.png', image)
img = Image.new('RGB', (imx, imy), color = 'white')
#img = Image.open('sweetpix/boatblur.jpg')
########

res50 = models.resnet50(pretrained=True, num_classes=1000).cuda()
res50.eval()

vgg19 = models.vgg19(pretrained=True, num_classes=1000).cuda()

densenet = models.densenet161(pretrained=True, num_classes=1000).cuda()
densenet.eval()

#incep_v3 = models.inception_v3(pretrained=True, num_classes=1000).cuda()

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
])

url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
      'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'

imagenet_classes = eval(requests.get(url).content)

images = [('cat', 'https://www.wired.com/wp-content/uploads/2015/02/catinbox_cally_by-helen-haden_4x31-660x495.jpg'),
          ('pomeranian', 'https://c.photoshelter.com/img-get/I0000q_DdkyvP6Xo/s/900/900/Pomeranian-Dog-with-Ball.jpg'),
          ('boat', 'http://gasequipment.com.au/wp-content/uploads/2016/03/Boat-for-Blogpost.jpg')]

'''
for class_name, image_url in images:
    response = requests.get(image_url)
    im = Image.open(BytesIO(response.content))
    #im = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
    tens = Variable(trans(im))
    tens = tens.view(1, 3, 224, 224).cuda()
    res_embedding = res50(tens)
    vgg_embedding = vgg19(tens)
    #print(embedding)
    res_preds = nn.LogSoftmax(dim=1)(res_embedding).data.cpu().numpy()
    res_index = np.argmax(res_preds)
    vgg_preds = nn.LogSoftmax(dim=1)(vgg_embedding).data.cpu().numpy()
    vgg_index = np.argmax(vgg_preds)
    print('index: ', res_index)
    print(res_preds)
    print('true (likely) label:', class_name)
    print('res50 predicted ', imagenet_classes[res_index], ' confidence: ', res_preds[0][res_index], '\n')
    print('vgg19 predicted ', imagenet_classes[vgg_index], ' confidence: ', vgg_preds[0][vgg_index], '\n')

#print( resnet50( torch.rand(1,3,266,266) ) )
'''

def rnd():
    return np.random.random_sample()

def make_score(models, image, index):
    tensor = Variable(trans(image))
    tensor = tensor.view(1,3,224,224).cuda()
    scores = []
    for m in models:
        t = time.time()
        scores.append(score_image(m, tensor, index))
        #print(m.__class__.__name__, ' took ', (time.time()-t), ' seconds')
    torch.cuda.synchronize()
    return min(scores)

def score_image(model, img_t, index):
    r = model(img_t)
    return nn.LogSoftmax(dim=1)(r).data.cpu().numpy()[0][index]

def mutate_image(img):
    mutated = img.copy()
    draw = ImageDraw.Draw(mutated)
    esize = rnd() * 3 + 2 #* (1000/(i*0.2)) + 2
    color = ( int(rnd() * 255), int(rnd() * 255), int(rnd() * 255) )
    rx = rnd()*imx
    ry = rnd()*imy
    #draw.ellipse([rx-esize, ry-esize, rx+esize, ry+esize], fill=color)
    draw.polygon([(rnd()*imx,rnd()*imy), (rnd()*imx,rnd()*imy), 
                  (rnd()*imx,rnd()*imy)], fill=color)
    return mutated

def save_grid(pop):
    grid_size = 5
    dim_x, dim_y = pop[0][0].size
    background = Image.new('RGB',(dim_x*grid_size, dim_y*grid_size), (255, 255, 255))
    for y_off in range(grid_size):
        for x_off in range(grid_size):
            background.paste(pop[x_off+grid_size*y_off][0], (x_off*dim_x, y_off*dim_y))
    background.save('sweetpix/grid.png')


population_size = 300
survivor_rate = 0.10
opt_index = 814
global_best = -100
population = [(img,-99999)]

for generation in range(1,30000):

    # reproduction
    index = 0
    while (len(population) < population_size):
        population.append( ( mutate_image(population[index][0]), -99999) )
        index += 1
        if (index >= len(population)):
            index = 0

    # score fitness
    for indiv_index in range(len(population)):
        population[indiv_index] = (population[indiv_index][0], 
                make_score([res50, vgg19, densenet], population[indiv_index][0], opt_index))
    
    population = sorted(population, key=lambda tup: tup[1], reverse=True)

    best_indiv = population[0][1]
    average_i = 0.0
    for indiv in population:
        average_i += indiv[1]
    average_i /= len(population)
    
    # make histogram?

    # starvation :(
    cutoff = int( population_size*survivor_rate )
    population = population[:cutoff]
    del population[random.randint(0,len(population)-1)]
    #del population[random.randint(0,len(population)-1)]

    if (generation%5==0):
        print("iteration: " + str(generation))
        print("best: " + str(best_indiv))
        print("average: " + str(average_i))
        save_grid(population)
    #next_img = img.copy()
    #t = time.time()
    #draw = ImageDraw.Draw(next_img)
    #for dr in range(10):
    #esize = rnd() * 3 + 2 #* (1000/(i*0.2)) + 2
    #color = ( int(rnd() * 255), int(rnd() * 255), int(rnd() * 255) )
    #rx = rnd()*imx
    #ry = rnd()*imy
    #draw.ellipse([rx-esize, ry-esize, rx+esize, ry+esize], fill=color)
    #draw.polygon([(rnd()*imx,rnd()*imy), (rnd()*imx,rnd()*imy), 
    #              (rnd()*imx,rnd()*imy)], fill=color)
    #del draw
    #print('Drawing on image took ', (t-time.time()), ' seconds')
    #cur_score = make_score([res50, vgg19], img, opt_index)
    #next_score = make_score([res50, vgg19], next_img, opt_index)

    #if (next_score > global_best):
    #    img = next_img
    #    global_best = next_score
    #    print('new best found: ', global_best)
    #    img.save('sweetpix/image.png')

    #if ((i%2000) == 0):
    #    print('step: ' + str(i) + ' progress: ' + str(i/10000.0) + '%')
        
    #global_best -= 0.001
    
    #draw = ImageDraw.Draw(img)
    #esize = 5
    #color = (0,0,255)
    #draw.ellipse([cx-esize, cy-esize, cx+esize, cy+esize], fill=color)
    #del draw
    #img.save('sweetpix/image.png')

