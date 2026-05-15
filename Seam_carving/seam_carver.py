import numpy as np
import cv2
from numba import jit


@jit(nopython=True)
def compute_forward_energy_map(img_gray, mask_weights):
    """
    calculates the cost of removing pixels using forward energy 
    (looks at neighbors after removal instead of just current edges)
    """
    rows, cols = img_gray.shape
    M = np.zeros((rows, cols), dtype=np.float64)
    backtrack = np.zeros((rows, cols), dtype=np.int32)
    
    for i in range(1, rows):
        for j in range(cols):
            # edge cases for left/right boundaries
            j_l = max(0, j - 1)
            j_r = min(cols - 1, j + 1)
            
            # calculate forward energy costs
            c_u = abs(img_gray[i, j_r] - img_gray[i, j_l])
            c_l = c_u + abs(img_gray[i-1, j] - img_gray[i, j_l])
            c_r = c_u + abs(img_gray[i-1, j] - img_gray[i, j_r])
            
            pixel_energy = mask_weights[i, j]

            # find min cost from top row to current pixel using dp
            if j == 0:
                # if left border, can only come from up or right
                if M[i-1, j] + c_u < M[i-1, j+1] + c_r:
                    M[i, j] = M[i-1, j] + c_u + pixel_energy
                    backtrack[i, j] = j
                else:
                    M[i, j] = M[i-1, j+1] + c_r + pixel_energy
                    backtrack[i, j] = j + 1
            elif j == cols - 1:
                # if right border, can only come from up or left
                if M[i-1, j] + c_u < M[i-1, j-1] + c_l:
                    M[i, j] = M[i-1, j] + c_u + pixel_energy
                    backtrack[i, j] = j
                else:
                    M[i, j] = M[i-1, j-1] + c_l + pixel_energy
                    backtrack[i, j] = j - 1
            else:
                # middle pixels: check left up right
                min_cost = M[i-1, j] + c_u
                backtrack[i, j] = j
                
                cost_left = M[i-1, j-1] + c_l
                if cost_left < min_cost:
                    min_cost = cost_left
                    backtrack[i, j] = j - 1
                
                cost_right = M[i-1, j+1] + c_r
                if cost_right < min_cost:
                    min_cost = cost_right
                    backtrack[i, j] = j + 1
                    
                M[i, j] = min_cost + pixel_energy

    return M, backtrack

@jit(nopython=True)
def backtrack_seam(M, backtrack):
    """
    starts from the bottom and traces the path back to the top
    """
    rows, cols = M.shape
    seam_idx = np.zeros(rows, dtype=np.int32)

    # find the lowest cost pixel in the last row
    j = np.argmin(M[-1])
    seam_idx[-1] = j

    # follow the backtrack pointers up
    for i in range(rows-1, 0, -1):
        j = backtrack[i, j]
        seam_idx[i-1] = j
    return seam_idx

class SeamCarver:
    def __init__(self, image_path):
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError("Could not load image.")
        self.curr_img = self.original_img.copy()
        
        # 0 = Protect, -100000 = Remove
        self.mask = np.zeros((self.curr_img.shape[0], self.curr_img.shape[1]), dtype=np.float64)

    def update_mask(self, new_mask): # placing the pixels where the mask goes as a negative to make them an unimportant seam 
        self.mask = np.where(new_mask > 0, -100000.0, 0.0)

    def get_energy_map_visualization(self):
        gray = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(sobel_x) + np.abs(sobel_y)
        
        # Normalizeation 
        energy = np.clip(energy, 0, 255) 
        norm_energy = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # inferno colormap
        heatmap = cv2.applyColorMap(norm_energy, cv2.COLORMAP_INFERNO)
        
        return heatmap

    # remove logic
    def remove_vertical_seams(self, num_seams, callback=None, status_callback=None):
        print(f"Removing {num_seams} seams...")
        for i in range(num_seams):
            rows, cols, _ = self.curr_img.shape
            gray = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            if self.mask.shape != (rows, cols):
                self.mask = cv2.resize(self.mask, (cols, rows), interpolation=cv2.INTER_NEAREST)

            M, backtrack = compute_forward_energy_map(gray, self.mask) # calc cost using numba function
            seam = backtrack_seam(M, backtrack)

            if status_callback: # visualizing the seam 
                vis_img = self.curr_img.copy()
                for r in range(rows):
                    c = seam[r]
                    if 0 <= c < cols:
                        vis_img[r, c] = [0, 0, 255] # red line for the seam
                status_callback(vis_img)

            mask_bool = np.ones((rows, cols), dtype=bool)
            for r, c in enumerate(seam):
                mask_bool[r, c] = False
            
            self.curr_img = np.dstack([
                self.curr_img[:, :, 0][mask_bool].reshape(rows, cols-1),
                self.curr_img[:, :, 1][mask_bool].reshape(rows, cols-1),
                self.curr_img[:, :, 2][mask_bool].reshape(rows, cols-1)
            ])
            self.mask = self.mask[mask_bool].reshape(rows, cols-1)
            
            if callback:
                callback(i + 1, num_seams)
        return self.curr_img

    # insert logic
    def insert_vertical_seams(self, num_seams, callback=None, status_callback=None):
        print(f"Inserting {num_seams} seams...")
        temp_img = self.curr_img.copy() # clone
        temp_mask = self.mask.copy()
        seams_record = []
        
        # find th best seams
        for i in range(num_seams):
            rows, cols, _ = temp_img.shape
            gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            M, backtrack = compute_forward_energy_map(gray, temp_mask)
            seam = backtrack_seam(M, backtrack) # using backtracking the best seams are found and added to seams_record
            seams_record.append(seam)
            
            mask_bool = np.ones((rows, cols), dtype=bool)
            for r, c in enumerate(seam):
                mask_bool[r, c] = False
                
                # we remove to find other best seams
            temp_img = np.dstack([
                temp_img[:, :, 0][mask_bool].reshape(rows, cols-1),
                temp_img[:, :, 1][mask_bool].reshape(rows, cols-1),
                temp_img[:, :, 2][mask_bool].reshape(rows, cols-1)
            ])
            temp_mask = temp_mask[mask_bool].reshape(rows, cols-1)

        # insert the seams
        for i, seam in enumerate(seams_record):
            rows, cols, _ = self.curr_img.shape
            
            new_img = np.zeros((rows, cols + 1, 3), dtype=np.uint8)
            new_mask = np.zeros((rows, cols + 1), dtype=np.float64)
            
            # live representation
            if status_callback:
                vis_img = self.curr_img.copy() # making a copy of the img
                for r in range(rows):
                    c = seam[r]
                    if 0 <= c < cols:
                        vis_img[r, c] = [0, 255, 0] # painting the seam green
                status_callback(vis_img) # overriding the original image with the seam one

            # the img splitting logic
            for r in range(rows):
                c = seam[r] 
                new_img[r, :c] = self.curr_img[r, :c] # copy of the left side not including the seam
                new_mask[r, :c] = self.mask[r, :c]
                new_img[r, c+1:] = self.curr_img[r, c:] # copy of the right side '' '' '' '' '' '' '' 
                new_mask[r, c+1:] = self.mask[r, c:]
                
                # filling the gaps
                if c > 0 and c < cols:
                    p_left = self.curr_img[r, c-1].astype(int)
                    p_right = self.curr_img[r, c].astype(int)
                    new_pixel = (p_left + p_right) // 2 # the new pixel to blend in both left and right pixel 
                else:
                    new_pixel = self.curr_img[r, c]

                new_img[r, c] = new_pixel
                new_mask[r, c] = self.mask[r, c]

            self.curr_img = new_img
            self.mask = new_mask

            for future_seam in seams_record[i+1:]: # shifitng the seams bec their coords change
                future_seam[np.where(future_seam >= seam)] += 2
            
            if callback:
                callback(i + 1, num_seams)

        return self.curr_img