import os 
import numpy as np  
from tensorflow.keras.preprocessing import image as kimage              
from feature_extractor import FeatureExtractor

# Define root path for image database and feature database
root_image_path = "./static/image_database/"
root_feature_path = "./static/feature_database/"


# Path of image folder
def folder_to_images(folder):

    list_dir = [folder + "/" + name for name in os.listdir(folder) if name.endswith((".jpeg", ".jpg", ".png"))] 
    
    k = 0
    
    images_np = np.zeros(shape=(len(list_dir), 224, 224, 3))

    images_path = []
   
    for path in list_dir:
        try:
            img = kimage.load_img(path, target_size = (224, 224))  
            images_np[k] = kimage.img_to_array(img, dtype = np.float32)
            images_path.append(path)
            k += 1
            
        except Exception:
            print("error: ", path)         
    images_path = np.array(images_path)
    return images_np, images_path           


# Extract features
if __name__ == "__main__":

    fe = FeatureExtractor()
    
    for folder in os.listdir(root_image_path):
        path = root_image_path + folder
        images_np, images_path = folder_to_images(path)
        np.savez_compressed(root_feature_path + folder, array_1 = np.array(images_path), array_2 = fe.extract(images_np)) # 


imgs_feature = []  
paths_feature = []  


for folder in os.listdir(root_image_path):  
    path = root_image_path + folder          
    images_np, images_path = folder_to_images(path)     
    paths_feature.extend(np.array(images_path))        
    imgs_feature.extend(fe.extract(images_np))          

# Save feature        
np.savez_compressed(root_feature_path + "concat_all_feature", array_1 = np.array(paths_feature), array_2 = np.array(imgs_feature))