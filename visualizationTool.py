import numpy as np
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
from scipy import ndimage


class ImageGui(tk.Tk):
    """
    GUI for iFind1 image sorting. This draws the GUI and handles all the events.
    Useful, for sorting views into sub views or for removing outliers from the data.
    """
    
    def __init__(self, master, SATimages, LIDARimages):
        """
        Initialise GUI
        :param master: The parent window
        :param labels: A list of labels that are associated with the images
        :return:
        """   
        
        # So we can quit the window from within the functions
        self.master = master
        
        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()
        
        # Start at the first file name
        self.index = 0
        self.n_images = SATimages.shape[0]
        self.SATimages = SATimages
        self.LIDARimages = LIDARimages
        
        # Number of labels and paths
        
        # Set empty image container
#        self.image_raw = None
        self.image1 = None
        self.image2 = None
        self.image_panel = tk.Label(frame)
        self.image_panel2 = tk.Label(frame)
        
        # Set image container to first image
        self.set_image()
            
        self.buttons = [] 
                            
        ### added in version 2
        self.buttons.append(tk.Button(frame, text="prev im", width=10, height=2, fg="purple", command=lambda l=0: self.show_prev_image()))
        self.buttons.append(tk.Button(frame, text="next im", width=10, height=2, fg='purple', command=lambda l=0: self.show_next_image()))
        ###
        
        
        # Add progress label
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)  
        self.progress_label.grid(row=2, column=1, sticky='we')
                                                                   
        # Place buttons in grid
        for ll, button in enumerate(self.buttons):
            button.grid(row=0, column=ll, sticky='we')
            frame.grid_columnconfigure(ll, weight=1)        
        
        
         # Place the image in grid
        self.image_panel.grid(row=1, column=0, sticky='we')
        self.image_panel2.grid(row=1, column=1, sticky='we')
                
#******************************************************************************    

    def set_image(self):
        """
        Helper function which sets a new image in the image view
        :param path: path to that image
        """
        image = self.SATimages[self.index,[0,1,2],:,:]
        image = (np.einsum('kij->ijk', image)*255).astype('uint8')
        imagePIL = Image.fromarray(image).resize((200,200), Image.NEAREST)
        self.image1 = ImageTk.PhotoImage(imagePIL, master = self.master)
        self.image_panel.configure(image=self.image1)
        
        #LIDAR
        imageLIDAR = self.LIDARimages[self.index, :,:].astype('uint8')
        imagePIL = Image.fromarray(imageLIDAR).resize((200,200), Image.NEAREST)
        self.image2 = ImageTk.PhotoImage(imagePIL, master = self.master)
        self.image_panel2.configure(image=self.image2)
         
        
    def show_prev_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 1
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string)
        
        if self.index >= 0:
            self.set_image()
        else:
        #            self.master.quit()
            self.master.destroy()

            
    def show_next_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string) 

        if self.index < self.n_images:
            self.set_image()

        else:
#            self.master.quit()
            self.master.destroy()
            
            
    def vote(self, value):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """           
        self.selection[self.index] = value    
            
        self.show_next_image()
 

#******************************************************************************        



SATimages = np.load('Datasets/SAT_species_256revised.npy')
LIDARimages = np.load('Datasets/species_256revisedraw.npy')

# Start the GUI
if __name__ == '__main__':
    root = tk.Tk()
    root.title("SAT - LIDAR Visualization Utility")
    app = ImageGui(root, SATimages, LIDARimages)
    root.mainloop()  
  