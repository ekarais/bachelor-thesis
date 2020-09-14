import numpy as np
import torch
import matplotlib.pyplot as plt
from vae import Encoder, Decoder, VAE
from datagen import logits_to_RPM, render

'''
To generate a sample from a trained VAE, set the model_name parameter to the name of the VAE.
The generated sample will be saved to output_dir in .npy format, and to the img_dir in rendered
form.
'''


params = {'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 128,        # hidden dimension
          'LATENT_DIM' : 25,         # latent vector dimension
          'output_dir' : '/home/ege/Documents/bthesis/data/vae_generated/',
          'img_dir' : '/home/ege/Documents/bthesis/images/',
          'models_dir' : '/home/ege/Documents/bthesis/models/',
          'model_name': 'mild-violet-8'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder
encoder = Encoder(params['INPUT_DIM'], params['HIDDEN_DIM'], params['LATENT_DIM'])

# decoder
decoder = Decoder(params['LATENT_DIM'], params['HIDDEN_DIM'], params['INPUT_DIM'])

# vae
model = VAE(encoder, decoder).to(device)
model.load_state_dict(torch.load(params['models_dir'] + params['model_name']))
model.eval()

# sample and generate a image
z = torch.randn(1, params['LATENT_DIM']).to(device)
sample = model.dec(z)
sample = sample.view((9,336)).data
sample = sample.cpu().numpy()
np.save(params['output_dir'] + params['model_name'] + '_sample.npy', sample)
img = logits_to_RPM(sample)
render(img, directory=params['img_dir'], name=params['model_name'] + '_sample')
