import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MouseClickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Click Detection")

        # Create a matplotlib figure and axes
        self.fig, self.ax = plt.subplots()

        # Plot some data
        self.ax.plot([1, 2, 3, 4, 5], [10, 20, 25, 30, 40])

        # Add a title
        self.ax.set_title("Click to get coordinates")

        # Create a canvas to embed the figure in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect the click event to the on_click function
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        """Handle mouse click event."""
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            print(f"Mouse clicked at coordinates: ({x}, {y})")
        else:
            print("Click was outside the axes")

if __name__ == "__main__":
    # Create the Tkinter root window
    root = tk.Tk()

    # Create the application instance
    app = MouseClickApp(root)

    # Run the Tkinter event loop
    root.mainloop()
