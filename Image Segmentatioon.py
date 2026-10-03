

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt



class image_segmentation:
    def __init__(self):  # Initialize the segmentation model and specify the user-defined parameters if needed
        
        self.num_classes  = 5
        self.layer_widths = [64, 32]
        self.lr           = 0.01
        self.epochs       = 60
        self.patch_radius = 2
        self.batch_size   = 1024    
        
        self.weights = []
        self.biases  = []
        self.features_mean = None
        self.features_std  = None
        self.sample_fraction = 0.1


    def extract_features(self, image):
        

        # Step 2: Extract features from the image
        # 2.1: Normalising data between 0 and 1
        image_rgb = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255.0
        image_hsv = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV)).astype(np.float32) / 255.0

        #2.2Filling Channel Indices
        R = image_rgb[:, :, 0]
        G = image_rgb[:, :, 1]
        B = image_rgb[:, :, 2]

        H = image_hsv[:, :, 0]
        S = image_hsv[:, :, 1]
        V = image_hsv[:, :, 2]
        #2.2. Calculating Mean and deviation for each channel for texture
        size = 2 * self.patch_radius + 1   # = 5
        means = []
        stds  = []
        channels = [R, G, B, H, S, V]
        for channel in channels:
            mean = uniform_filter(channel, size=size, mode='reflect')
            std  = np.sqrt(uniform_filter(channel**2, size=size, mode='reflect') - mean**2)
            means.append(mean)
            stds.append(std)
        """R, G, B, H, S, V          → 6 individual arrays
           means                      → list of 6 arrays
           stds                       → list of 6 arrays

           Total: 18 features per pixel"""
        #2.4Stack all the values intoa feature matrix[12e6 , 18]
        # flatten each array from (3000,4000) to (12000000,)
        all_features = [R.flatten(), G.flatten(), B.flatten(),H.flatten(), S.flatten(), V.flatten()]
        for i in range(6):
            all_features.append(means[i].flatten())
            all_features.append(stds[i].flatten())

        features = np.stack(all_features, axis=1)
        return features

    def one_hot(self, label):
        """
        Convert integer labels to one-hot matrix
        y shape: (N,)  → output shape: (N, 5)
        """
        one_hot_matrix = np.zeros((len(label), self.num_classes))
        one_hot_matrix[np.arange(len(label)), label - 1] = 1.0
        return one_hot_matrix

    def _forward_pass(self, X):
        cache = []
        a = X
    
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
        
            if i < len(self.weights) - 1:
                a_next = np.maximum(0, z)    # ReLU
            else:
                exp_z  = np.exp(z - z.max(axis=1, keepdims=True))
                a_next = exp_z / exp_z.sum(axis=1, keepdims=True)  # Softmax
        
            cache.append({'z': z, 'a_in': a})
            a = a_next
    
        return cache, a

    def _backward_pass(self, cache, prob, label_onehot, lr):
    
        n = prob.shape[0]   # batch size
    
        # output layer gradient
        delta = (prob - label_onehot) / n
    
        for i in reversed(range(len(self.weights))):
        
            a_in = cache[i]['a_in']
            z    = cache[i]['z']
        
            # gradients for this layer
            dW = a_in.T @ delta
            db = delta.sum(axis=0, keepdims=True)
        
            # propagate delta to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * (cache[i-1]['z'] > 0)
        
            # update weights and biases
            self.weights[i] -= lr * dW
            self.biases[i]  -= lr * db
        
    
    def model_training(self,Training_Image_Name,Training_Image_Mask_Name): # Train the model with the training image and its mask
        
        # Step 1: Load images
        image = cv2.imread(Training_Image_Name)
        mask  = cv2.imread(Training_Image_Mask_Name)

        # Step2: Extract features
        features = self.extract_features(image)


        # Step 3: Extracting labels from mask
        Pxl_labels = mask[:, :, 0].flatten().astype(np.int32)

        #step 4: Random sampling fraction of each class for training 

        sampled_indices = []

        for cls in range(1,6):
    
            cls_indices = np.where(Pxl_labels == cls)[0]
            n = int(len(cls_indices) * self.sample_fraction)
            chosen = np.random.choice(cls_indices, size=n, replace=False)
            sampled_indices.append(chosen)

        # combine all sampled indices into one array
        sampled_indices = np.concatenate(sampled_indices)

        # select sampled rows from features and labels
        sampled_features = features[sampled_indices]
        sampled_labels = Pxl_labels[sampled_indices]

        #Step5: Normalise the features

        # compute mean and std per feature (per column)
        self.features_mean = sampled_features.mean(axis=0)
        self.features_std  = sampled_features.std(axis=0) + 1e-8

        # normalise
        feature_normalised = (sampled_features - self.features_mean) / self.features_std


        # Step6:Initialise the network weights(18 inputs → 64 → 32 → 5 outputs)
        layer_sizes = [18] + self.layer_widths + [self.num_classes]
        # = [18, 64, 32, 5]

        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            neurons_in  = layer_sizes[i]
            neurons_out = layer_sizes[i+1]
            
            #He initialisation 
            W = np.random.randn(neurons_in, neurons_out) * np.sqrt(2.0 / neurons_in)
            b = np.zeros((1, neurons_out))
            
            self.weights.append(W)
            self.biases.append(b)


        #Step7: Training Loop
        lr = self.lr

        for epoch in range(self.epochs):
    
            # shuffle data each epoch
            shuffle_idx = np.random.permutation(len(feature_normalised))
            shuffled_sample_features  = feature_normalised[shuffle_idx]
            shuffled_sample_labels  = sampled_labels[shuffle_idx]
            
            epoch_loss  = 0.0    #  reset each epoch
            num_batches = 0      #  reset each epoch
    
            for batch_start in range(0, len(shuffled_sample_features), self.batch_size):
        
                # get mini batch
                batch_end = min(batch_start + self.batch_size, len(shuffled_sample_features))
                feature   = shuffled_sample_features[batch_start:batch_end]
                label   = shuffled_sample_labels[batch_start:batch_end]
        
                # convert labels to one-hot
                label_onehot = self.one_hot(label)
                
                # forward pass
                cache, probs = self._forward_pass(feature)
                loss = -np.mean(np.sum(label_onehot * np.log(probs + 1e-8), axis=1))
                
                epoch_loss  += loss    # ← accumulate
                num_batches += 1       # ← count
                
                # backward pass + update weights
                self._backward_pass(cache, probs, label_onehot, lr)
                avg_loss = epoch_loss / num_batches
    
            print(f"Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")
        
        
        print("\nEvaluating on training image...")
        features_n = (features - self.features_mean) / self.features_std
        
        predictions = []
        for start in range(0, len(features_n), 10000):
            end      = min(start + 10000, len(features_n))
            _, probs = self._forward_pass(features_n[start:end])
            predictions.append(np.argmax(probs, axis=1) + 1)
        predictions = np.concatenate(predictions)

        true_labels = mask[:,:,0].flatten()
        recalls = []
        names   = {1:'Building', 2:'Road', 3:'Tree', 4:'Vehicle', 5:'Grass'}
        for cls in range(1, 6):
            tp = np.sum((true_labels == cls) & (predictions == cls))
            fn = np.sum((true_labels == cls) & (predictions != cls))
            recall = tp / (tp + fn + 1e-8)
            recalls.append(recall)
            print(f"Class {cls} ({names[cls]:10s}): recall = {recall:.4f}")

        print(f"\nBalanced Accuracy = {np.mean(recalls)*1001:.2f}%")

        
    def model_testing(self,Testing_Image_Name,Testing_Image_Mask_Name): # Test the model with the testing image and save the prediction as Testing_Image_Mask_Name 
         
        # Step 1: Load test image
        image = cv2.imread(Testing_Image_Name)
    
        # Step 2: Extract features
        features = self.extract_features(image)
    
        # Step 3: Normalise
        features_normalised = (features - self.features_mean) / self.features_std
    
        # Step 4 & 5: Predict in chunks
        chunk_size  = 10000
        predictions = []
        for start in range(0, len(features_normalised), chunk_size):
            end      = min(start + chunk_size, len(features_normalised))
            chunk    = features_normalised[start:end]
            _, probs = self._forward_pass(chunk)
            pred     = np.argmax(probs, axis=1) + 1
            predictions.append(pred)
        predictions = np.concatenate(predictions)
    
        # Step 6: Reshape
        H, W     = image.shape[:2]
        pred_map = predictions.reshape(H, W)
    
        # Step 7: Save mask
        mask_out = np.stack([pred_map, pred_map, pred_map], axis=2).astype(np.uint8)
        cv2.imwrite(Testing_Image_Mask_Name,mask_out) 

Model=image_segmentation() 
Model.model_training("training_image.jpg","training_mask.png")
Model.model_testing("testing_image1.jpg","testing_mask1.png")
Model.model_testing("testing_image2.jpg","testing_mask2.png")
