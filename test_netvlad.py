from models.netvlad import NetVLAD

import torch
import torchvision.models as models

from PIL import Image
from torchvision import transforms

import os

from sklearn.metrics.pairwise import cosine_similarity

from matplotlib import pyplot as plt
import cv2
import numpy as np

import sys
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src = './data/House_Room_Dataset/Bedroom'

def load_model():
    # load pretrained model with netvlad
    encoder = models.vgg16().features[:-2]
    encoder.outchannels = 512

    net_vlad = NetVLAD(num_clusters=64, dim=512)
    model = torch.nn.Sequential(encoder, net_vlad).to(device)
    model.load_state_dict(torch.load("netvlad_model.pth", map_location=device))
    model.eval()
    return model

def extract_features(image_path, model):
    # preprocess the input image
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = trans(image).unsqueeze(0).to(device)

    with torch.no_grad():
        descriptor = model(image)
        descriptor = descriptor / descriptor.norm(p=2) # normalize
        # print("descriptor shape: ", descriptor.shape)
    
    return descriptor.squeeze().cpu()

def create_database():
    # image database
    images = os.listdir(src)
    descriptors = torch.stack([extract_features(os.path.join(src, img)) for img in images])
    return descriptors

# Plotting function
def plot_images(input_image_path, result_image_paths, title):
    num_images = len(result_image_paths)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols + 1  # Adding one row for the input image
    plt.figure(figsize=(15, 3 * num_rows))
    
    # Plot the input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis('off')
    
    for i, image_path in enumerate(result_image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(num_rows, num_cols, i + 6)  # Start from the second subplot
        plt.imshow(image)
        plt.title(f'{os.path.basename(image_path)}')
        plt.axis('off')

    plt.suptitle(title)
    plt.show()

def run_test_netvlad(query_img_path, data_path):

    model = load_model()
    # Code to load the saved descriptors
    checkpoint = torch.load('netvlad_descriptors.pt')
    image_paths = checkpoint['paths']              # List of image paths
    descriptors = checkpoint['descriptors']

    query_vec = extract_features(query_img_path, model).unsqueeze(0)
    # compute similarity with database
    similarities = cosine_similarity(query_vec, descriptors.numpy())
    top_k_indices = similarities.argsort()[0][::-1][:3]  # top 3-k

    result = [image_paths[i] for i in top_k_indices]
    actual_result = []
    query_im = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    for image_path in result:
        im = cv2.imread(os.path.join(data_path, image_path), cv2.IMREAD_GRAYSCALE)
        if im.shape != query_im.shape:
            im = cv2.resize(im, (query_im.shape[1], query_im.shape[0]))
        if ssim(query_im, im) > 0.9:
            actual_result.append(image_path)
    # plot_images(query_img_path, [os.path.join(src, image_paths[i]) for i in top_k_indices], "Top 5 Similar Images")

    return actual_result

if __name__ == "__main__":

    # Example usage: python test_netvlad.py <input_image_path> <data_path>
    # python test_netvlad.py './data/House_Room_Dataset/Bedroom/bed_485.jpg' './data/House_Room_Dataset/Bedroom'
    input_image_path = sys.argv[1]
    data_path = sys.argv[2]
    # Run the test with a query image
    # input_image_path = './data/House_Room_Dataset/Bedroom/bed_485.jpg'
    result = run_test_netvlad(input_image_path, data_path)
    print(result)