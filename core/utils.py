import matplotlib.pyplot as plt

def show_image(x, title=""):
    plt.imshow(x.view(28,28), cmap="gray")
    plt.title(title)
    plt.axis("off")

def plot_grid(images, title):
    n = len(images)
    plt.figure(figsize=(15,3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        show_image(images[i], f"step {i}")
    plt.suptitle(title)
    plt.show()
