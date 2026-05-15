import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk  
from PIL import Image, ImageTk, ImageDraw
import cv2
import threading
import queue
import time
import numpy as np
from seam_carver import SeamCarver

# Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue") 

# Colors used
COLOR_BG = "#0F0F0F"        # Main window background 
COLOR_PANEL = "#1E1E1E"     # Card/Sidebar background
COLOR_ACCENT = "#00E096"    # run
COLOR_ACCENT_HOVER = "#00B377"
COLOR_DANGER = "#FF4B4B"    # clear
COLOR_TEXT_MAIN = "#FFFFFF"
COLOR_TEXT_SUB = "#AAAAAA"

class SeamCarvingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup
        self.title("SeamCarve Pro")
        self.geometry("1200x850")
        self.configure(fg_color=COLOR_BG) 

        # fixed sidebar
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # variables
        self.original_image_path = None
        self.carver = None          # Instance of the seamcarving logic class
        self.paint_mode = False     # flag for removing tool
        self.mask_image = None      # stores the user's red drawings 
        self.draw = None            # ImageDraw object for painting on the mask
        self.showing_energy = False # flag for energymap
        
        # canvas size
        self.canvas_w = 800
        self.canvas_h = 700
        self.current_scale = 1.0    
        self.img_x_start = 0        # X-offset (black bar)
        self.img_y_start = 0        # Y-offset (black bar)

        
        # Left sidebar
        self.sidebar = ctk.CTkScrollableFrame(self, width=280, corner_radius=0, fg_color=COLOR_PANEL)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="SeamCarve\n Project ", 
                                       font=ctk.CTkFont(family="Roboto", size=24, weight="bold"),
                                       text_color=COLOR_TEXT_MAIN)
        self.logo_label.pack(pady=(30, 20), padx=20)

        # importing/exporting img
        self.create_card_label("FILE IO")
        self.card_file = self.create_card_frame()
        
        self.btn_load = ctk.CTkButton(self.card_file, text="📂 Import Image", 
                                      fg_color="#333333", hover_color="#444444", 
                                      height=40, font=("Arial", 14),
                                      command=self.load_image)
        self.btn_load.pack(fill="x", pady=5)

        self.btn_save = ctk.CTkButton(self.card_file, text="💾 Export Result", 
                                      fg_color="#333333", hover_color="#444444", 
                                      height=40, font=("Arial", 14),
                                      command=self.save_image)
        self.btn_save.pack(fill="x", pady=5)
        
        # mode and seam counts
        self.create_card_label("CONFIGURATION")
        self.card_config = self.create_card_frame()

        self.lbl_mode = ctk.CTkLabel(self.card_config, text="Operation Mode:", text_color=COLOR_TEXT_SUB, anchor="w")
        self.lbl_mode.pack(fill="x", pady=(5,0))
        
        # toggling between shrink and enlarge
        self.mode_var = ctk.StringVar(value="Shrink")
        self.seg_mode = ctk.CTkSegmentedButton(self.card_config, values=["Shrink", "Enlarge"], 
                                               command=self.change_mode, variable=self.mode_var,
                                               selected_color=COLOR_ACCENT, selected_hover_color=COLOR_ACCENT_HOVER)
        self.seg_mode.pack(fill="x", pady=5)

        self.lbl_seams = ctk.CTkLabel(self.card_config, text="Target Seams:", text_color=COLOR_TEXT_SUB, anchor="w")
        self.lbl_seams.pack(fill="x", pady=(10,0))
        
        self.slider_seams = ctk.CTkSlider(self.card_config, from_=0, to=100, command=self.update_slider_label, 
                                          button_color=COLOR_ACCENT, progress_color=COLOR_ACCENT)
        self.slider_seams.pack(fill="x", pady=5)
        self.slider_seams.set(0)

        self.lbl_slider_val = ctk.CTkLabel(self.card_config, text="0 px", font=("Arial", 16, "bold"), text_color=COLOR_ACCENT)
        self.lbl_slider_val.pack(pady=0)

        # mask used to delete
        self.create_card_label("MASK")
        self.card_mask = self.create_card_frame()
        
        self.btn_paint = ctk.CTkButton(self.card_mask, text="🖌️ Paint Mask (OFF)", 
                                       fg_color="#333333", hover_color="#444444",
                                       command=self.toggle_paint)
        self.btn_paint.pack(fill="x", pady=5)

        self.btn_reset_mask = ctk.CTkButton(self.card_mask, text="✖ Clear Mask", 
                                            fg_color="transparent", border_width=1, border_color=COLOR_DANGER, text_color=COLOR_DANGER,
                                            hover_color="#330000",
                                            command=self.clear_mask)
        self.btn_reset_mask.pack(fill="x", pady=5)

        # visualization
        self.create_card_label("ACTIONS")
        self.card_actions = self.create_card_frame()

        self.btn_energy = ctk.CTkButton(self.card_actions, text="🔥 View Heatmap", 
                                        fg_color="#333333", hover_color="#444444",
                                        command=self.toggle_energy_view)
        self.btn_energy.pack(fill="x", pady=5)

        self.animate_var = ctk.BooleanVar(value=True)
        self.chk_animate = ctk.CTkCheckBox(self.card_actions, text="Live Animation", variable=self.animate_var, 
                                           text_color=COLOR_TEXT_SUB, hover_color=COLOR_ACCENT, fg_color=COLOR_ACCENT)
        self.chk_animate.pack(fill="x", pady=10, padx=5)

        # run
        self.btn_process = ctk.CTkButton(self.sidebar, text="▶ RUN CARVING", 
                                         fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, text_color="black",
                                         height=50, font=("Arial", 16, "bold"),
                                         command=self.start_processing)
        self.btn_process.pack(fill="x", padx=20, pady=(10, 5))

        self.progress = ctk.CTkProgressBar(self.sidebar, progress_color=COLOR_ACCENT)
        self.progress.pack(fill="x", padx=20, pady=5)
        self.progress.set(0)


        # main display
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # frame for the canvas
        self.canvas_container = ctk.CTkFrame(self.main_frame, fg_color=COLOR_PANEL, corner_radius=15)
        self.canvas_container.pack(expand=True, fill="both")

        # using Tkinter canvas for mouse coords
        self.canvas = tk.Canvas(self.canvas_container, width=self.canvas_w, height=self.canvas_h, 
                                bg=COLOR_PANEL, highlightthickness=0)
        self.canvas.pack(expand=True, padx=10, pady=10)
        
        self.canvas.create_text(self.canvas_w//2, self.canvas_h//2, text="Import an Image to Start", fill="#555555", font=("Arial", 20))

        # mouse countrols for drawing
        self.canvas.bind("<B1-Motion>", self.paint_on_canvas)
        self.canvas.bind("<Button-1>", self.paint_on_canvas)

        # Thread queue for live vis 
        self.visual_queue = queue.Queue()
        self.check_queue() 


    # helper for ui creation
    def create_card_label(self, text):
        label = ctk.CTkLabel(self.sidebar, text=text, font=("Arial", 11, "bold"), text_color="#666666", anchor="w")
        label.pack(fill="x", padx=25, pady=(20, 0))

    def create_card_frame(self):
        frame = ctk.CTkFrame(self.sidebar, fg_color="#262626", corner_radius=10)
        frame.pack(fill="x", padx=20, pady=5)
        inner_padding = ctk.CTkFrame(frame, fg_color="transparent")
        inner_padding.pack(padx=10, pady=10, fill="x")
        return inner_padding

    # LOGIC 

    def change_mode(self, value):
        """Updates UI  based on Shrink vs Enlarge mode"""
        if value == "Enlarge":
            self.btn_process.configure(text="▶ EXTEND IMAGE", fg_color="#3B8ED0", hover_color="#36719F")
            # Painting masks is disabled in enlarge mode
            self.btn_paint.configure(state="disabled") 
        else:
            self.btn_process.configure(text="▶ CARVE SEAMS", fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER)
            self.btn_paint.configure(state="normal")

    def toggle_energy_view(self):
        """Switches between showing the normal photo and the gradient heatmap"""
        if not self.carver: return
        
        if not self.showing_energy:
            # Switch to Heatmap
            energy_img = self.carver.get_energy_map_visualization()
            self.display_image(energy_img, is_energy_view=True)
            self.btn_energy.configure(text="⬅ Back to Original", fg_color=COLOR_ACCENT, text_color="black")
            self.showing_energy = True
            
            # Lock controls so you cant edit the heatmap
            self.btn_process.configure(state="disabled")
            self.btn_paint.configure(state="disabled")
        else:
            # Switch back to original
            self.display_image(self.carver.curr_img)
            self.btn_energy.configure(text="🔥 View Heatmap", fg_color="#333333", text_color="white")
            self.showing_energy = False
            self.btn_process.configure(state="normal")
            if self.mode_var.get() == "Shrink": self.btn_paint.configure(state="normal")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.original_image_path = file_path
            self.carver = SeamCarver(file_path)
            self.showing_energy = False
            self.btn_energy.configure(text="🔥 View Heatmap", fg_color="#333333")
            
            # Reset UI controls based on new image size
            h, w, _ = self.carver.original_img.shape
            self.slider_seams.configure(to=w)
            self.slider_seams.set(0)
            self.lbl_slider_val.configure(text="0 px")
            
            # Create a blank mask layer (same size as image)
            self.mask_image = Image.new("L", (w, h), 0)
            self.draw = ImageDraw.Draw(self.mask_image)
            
            self.display_image(self.carver.original_img)

    def save_image(self):
        """Exports whatever is currently visible on the canvas"""
        if not self.carver: return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            img_to_save = self.carver.get_energy_map_visualization() if self.showing_energy else self.carver.curr_img
            cv2.imwrite(file_path, img_to_save)
            print(f"Saved to {file_path}")

    def display_image(self, cv_img, is_energy_view=False):
        """
        Handles scaling and centering the image on the canvas
        Also overlays the red mask if painting mode is active
        """
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Overlay mask drawings  onto the image
        if not is_energy_view and self.mask_image and self.mask_image.size == img_pil.size:
            overlay = Image.new("RGB", img_pil.size, (255, 0, 0))
            img_pil = Image.composite(overlay, img_pil, self.mask_image)

        # Calculate Aspect Ratio Scaling to fit in 800x700 box
        w, h = img_pil.size
        self.current_scale = min(self.canvas_w / w, self.canvas_h / h)
        new_w = int(w * self.current_scale)
        new_h = int(h * self.current_scale)
        
        resized_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_pil)
        
        # Center image on canvas
        self.canvas.delete("all")
        cx, cy = self.canvas_w // 2, self.canvas_h // 2
        self.canvas.create_image(cx, cy, image=self.tk_img, anchor="center")
        
        # Store offsets so we can map mouse clicks back to image coordinates
        self.img_x_start = cx - (new_w // 2)
        self.img_y_start = cy - (new_h // 2)

    def toggle_paint(self):
        self.paint_mode = not self.paint_mode
        if self.paint_mode:
            self.btn_paint.configure(text="🖌️ Painting Active", fg_color="orange", text_color="black")
            self.canvas.configure(cursor="cross")
        else:
            self.btn_paint.configure(text="🖌️ Paint Mask (OFF)", fg_color="#333333", text_color="white")
            self.canvas.configure(cursor="")

    def clear_mask(self):
        if self.mask_image:
            # Fill mask with black (0) to remove all paint
            self.draw.rectangle((0, 0, self.mask_image.width, self.mask_image.height), fill=0)
            self.display_image(self.carver.curr_img)

    def paint_on_canvas(self, event):
        """
        Translates mouse clicks on the Canvas (screen coords) 
        to the actual Image coordinates (math coords)
        """
        if not self.paint_mode or not self.carver or self.showing_energy: return
        
        # Remove the grey sidebar/padding offsets
        x_rel = event.x - self.img_x_start
        y_rel = event.y - self.img_y_start
        
        # Scale back up to original resolution
        real_x = int(x_rel / self.current_scale)
        real_y = int(y_rel / self.current_scale)
        
        # Draw on the invisible mask layer
        w, h = self.mask_image.size
        if 0 <= real_x < w and 0 <= real_y < h:
            radius = int(15 / self.current_scale) # Scale brush size
            self.draw.ellipse((real_x-radius, real_y-radius, real_x+radius, real_y+radius), fill=255)
            self.display_image(self.carver.curr_img)

    def update_slider_label(self, value):
        self.lbl_slider_val.configure(text=f"{int(value)} px")

    def check_queue(self):
        
        #Checks if the background thread has sent a new image frame to display
        
        try:
            last_image = None
            # go thorugh the queue to get the latest frame 
            while True:
                last_image = self.visual_queue.get_nowait()
        except queue.Empty: pass
        
        if last_image is not None: 
            self.display_image(last_image)
        
        # 10ms to check queue
        self.after(10, self.check_queue)

    def start_processing(self):
        num_seams = int(self.slider_seams.get())
        mode = self.mode_var.get()
        if self.carver and num_seams > 0:
            if mode == "Shrink":
                # Convert PIL mask image to NumPy array for the algorithm
                mask_array = np.array(self.mask_image)
                self.carver.update_mask(mask_array)
                
                self.progress.set(0)
                # RUN IN BACKGROUND THREAD so GUI doesn't freeze
                threading.Thread(target=self.process_shrink, args=(num_seams,)).start()
            else:
                self.progress.set(0)
                threading.Thread(target=self.process_enlarge, args=(num_seams,)).start()

    def process_shrink(self, num_seams):
        # Reset image to original before starting new operation
        self.carver.curr_img = self.carver.original_img.copy()
        
        def update_progress(current, total): self.progress.set(current / total)
        
        def send_visual(img): 
            # Send frame to the main thread
            self.visual_queue.put(img)
            time.sleep(0.01) 

        if self.animate_var.get():
            result = self.carver.remove_vertical_seams(num_seams, callback=update_progress, status_callback=send_visual)
        else:
            result = self.carver.remove_vertical_seams(num_seams, callback=update_progress)
        
        self.progress.set(1)
        self.display_image(result)

    def process_enlarge(self, num_seams):
        self.carver.curr_img = self.carver.original_img.copy()
        
        def update_progress(current, total): self.progress.set(current / total)
        
        def send_visual(img): 
            self.visual_queue.put(img)
            time.sleep(0.01)

        if self.animate_var.get():
             result = self.carver.insert_vertical_seams(num_seams, callback=update_progress, status_callback=send_visual)
        else:
             result = self.carver.insert_vertical_seams(num_seams, callback=update_progress)
        
        self.progress.set(1)
        self.display_image(result)

if __name__ == "__main__":
    app = SeamCarvingApp()
    app.mainloop()